use std::ops::Range;

use hashbrown::{HashMap, HashSet};
use icicle_vm::{
    cpu::lifter::{Block, BlockExit, Target},
    BlockTable,
};
use pcode::{Op, Value, VarId};

use crate::stream_relation::AccessContext;

/// Traversal depth when crossing function call boundaries.
const MAX_CALL_LEVELS: usize = 3;

// ──────────────────────────────────────────────────────────────────────────────
// DemandSlice
// ──────────────────────────────────────────────────────────────────────────────

/// Backward demand slice over P-code varnodes.
///
/// Starting from a set of "demanded" varnode IDs (the roots of the slice), we
/// propagate demands backward through block instructions to find all MMIO LOAD
/// instructions that structurally contribute to those root values.
///
/// Temporaries (VarId < 0) are tracked within a block but NOT carried across
/// block boundaries, because P-code temporaries are local to a block.
#[derive(Default, Clone)]
pub struct DemandSlice {
    /// VarNode IDs whose defining instruction we are looking for.
    pub demanded: HashSet<VarId>,
    /// MMIO access contexts (PC, address) found as structural sources for the
    /// demanded varnodes.  Recorded at PC granularity so that the same MMIO
    /// address accessed at different sites produces distinct source contexts.
    pub sources:  HashSet<AccessContext>,
}

impl DemandSlice {
    /// Seed the slice from a single varnode (the address operand of B's LOAD).
    pub fn seed(vn: pcode::VarNode) -> Self {
        let mut s = Self::default();
        if !vn.is_invalid() {
            s.demanded.insert(vn.id);
        }
        s
    }

    /// Process one block backward.  Demands are expanded through P-code ops; MMIO LOADs
    /// are recorded as sources and removed from the demand set.
    ///
    /// After processing, `demanded` may still contain varnode IDs that were not defined in
    /// this block; those should be propagated to predecessor blocks.
    pub fn process_block_backward(&mut self, block: &Block, mmio_ranges: &[Range<u64>]) {
        // Forward pass to build (instruction_index → current_pc) so we can compute
        // StreamKeys that include PC when needed.
        let mut stmt_pcs = vec![block.start; block.pcode.instructions.len()];
        {
            let mut pc = block.start;
            for (i, stmt) in block.pcode.instructions.iter().enumerate() {
                if stmt.op == Op::InstructionMarker {
                    pc = stmt.inputs.first().as_u64();
                }
                stmt_pcs[i] = pc;
            }
        }

        // Backward pass.
        for (idx, stmt) in block.pcode.instructions.iter().enumerate().rev() {
            let out = stmt.output;
            if out.is_invalid() || !self.demanded.contains(&out.id) {
                continue;
            }
            // This instruction defines something we demanded — process it.
            self.demanded.remove(&out.id);

            match stmt.op {
                Op::Load(_) => {
                    let addr_val = stmt.inputs.first();
                    match addr_val {
                        Value::Const(addr, _) => {
                            if mmio_ranges.iter().any(|r| r.contains(&addr)) {
                                // Record the source at (instruction PC, MMIO address) so
                                // that the same peripheral register accessed at different
                                // PCs produces distinct source contexts.
                                self.sources.insert(AccessContext::new(stmt_pcs[idx], addr));
                            }
                            // If addr is not in MMIO range (e.g., a RAM load), demand the
                            // address itself in case it was computed from an MMIO value.
                        }
                        Value::Var(addr_var) if !addr_var.is_invalid() => {
                            // Dynamic address: demand the register holding it so we can
                            // trace how it was computed (it might hold an MMIO value).
                            self.demanded.insert(addr_var.id);
                        }
                        _ => {}
                    }
                }

                Op::Copy | Op::ZeroExtend | Op::SignExtend | Op::Subpiece(_) => {
                    if let Value::Var(src) = stmt.inputs.first() {
                        if !src.is_invalid() {
                            self.demanded.insert(src.id);
                        }
                    }
                }

                // Binary / unary arithmetic and comparison ops: demand all variable inputs.
                Op::IntAdd
                | Op::IntSub
                | Op::IntXor
                | Op::IntOr
                | Op::IntAnd
                | Op::IntMul
                | Op::IntDiv
                | Op::IntSignedDiv
                | Op::IntRem
                | Op::IntSignedRem
                | Op::IntLeft
                | Op::IntRight
                | Op::IntSignedRight
                | Op::IntRotateLeft
                | Op::IntRotateRight
                | Op::IntEqual
                | Op::IntNotEqual
                | Op::IntLess
                | Op::IntSignedLess
                | Op::IntLessEqual
                | Op::IntSignedLessEqual
                | Op::IntCarry
                | Op::IntSignedCarry
                | Op::IntSignedBorrow
                | Op::BoolAnd
                | Op::BoolOr
                | Op::BoolXor => {
                    for v in [stmt.inputs.first(), stmt.inputs.second()] {
                        if let Value::Var(var) = v {
                            if !var.is_invalid() {
                                self.demanded.insert(var.id);
                            }
                        }
                    }
                }

                Op::IntNot | Op::IntNegate | Op::IntCountOnes | Op::IntCountLeadingZeroes
                | Op::BoolNot => {
                    if let Value::Var(src) = stmt.inputs.first() {
                        if !src.is_invalid() {
                            self.demanded.insert(src.id);
                        }
                    }
                }

                // For any store: if the destination address was derived from a demanded value,
                // we don't directly expand — stores are not definitions of their address.
                // Phi / indirection / hooks: conservatively skip.
                _ => {}
            }
        }

        // Seed demands for branch conditions in this block.
        // Control-dependence: if a branch gating the path to B is driven by an MMIO value,
        // that's a Control edge candidate.  Add the condition varnode to the demand set so
        // it propagates backward through predecessor blocks.
        if let Some(cond) = block.exit.cond() {
            if let Value::Var(var) = cond {
                if !var.is_invalid() {
                    self.demanded.insert(var.id);
                }
            }
        }
    }

    /// Strip temporaries from the demanded set.  Call this before carrying demands across a
    /// block boundary: P-code temporaries (VarId < 0) don't survive across blocks.
    pub fn strip_temporaries(&mut self) {
        self.demanded.retain(|id| *id >= 0);
    }

    /// Merge another slice into this one (CFG join point).
    pub fn merge(&mut self, other: &DemandSlice) {
        self.demanded.extend(other.demanded.iter().copied());
        self.sources.extend(other.sources.iter().copied());
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Reverse CFG
// ──────────────────────────────────────────────────────────────────────────────

/// Reverse CFG edge: maps each target block start address to a list of
/// `(predecessor_block_start, is_call)` pairs.
///
/// `is_call = true` means the predecessor's exit was a `BlockExit::Call`, i.e.,
/// the predecessor is in the *caller* function and called into the target.
type ReverseCfg = HashMap<u64, Vec<(u64, bool)>>;

fn build_reverse_cfg(code: &BlockTable) -> ReverseCfg {
    let mut reverse: ReverseCfg = HashMap::new();
    for block in &code.blocks {
        let is_call = matches!(block.exit, BlockExit::Call { .. });
        for (i, target) in block.exit.targets().enumerate() {
            // For Call exits: targets()[0] is the callee, targets()[1] is the fall-through
            // return site. Only the callee edge crosses a function boundary.
            let edge_is_call = is_call && i == 0;
            let target_addr: Option<u64> = match target {
                Target::External(Value::Const(addr, _)) => Some(addr),
                // Internal indices address code.blocks directly (intra-function edges).
                Target::Internal(idx) => code.blocks.get(idx).map(|b| b.start),
                _ => None,
            };
            if let Some(addr) = target_addr {
                reverse.entry(addr).or_default().push((block.start, edge_is_call));
            }
        }
    }
    reverse
}

/// Find the block in `code.blocks` that contains `addr`.
/// Falls back to a linear scan since we don't have ISA mode without the CPU.
fn find_block_containing(code: &BlockTable, addr: u64) -> Option<&Block> {
    code.blocks.iter().find(|b| b.contains_addr(addr))
}

// ──────────────────────────────────────────────────────────────────────────────
// MmioFlowAnalyzer
// ──────────────────────────────────────────────────────────────────────────────

/// Performs Phase A: structural MMIO dependency inference from the P-code IR.
///
/// Called once per newly detected MMIO stream, at the moment the `ReadWatch`
/// exception fires.  No additional execution is needed — all information comes
/// from `vm.code` (already translated blocks).
pub struct MmioFlowAnalyzer {
    /// Address ranges that are memory-mapped I/O (reads from these are MMIO sources).
    pub mmio_ranges: Vec<Range<u64>>,
    /// Maximum number of function-call boundaries to cross when walking backward.
    pub max_call_levels: usize,
}

impl MmioFlowAnalyzer {
    pub fn new(mmio_ranges: Vec<Range<u64>>) -> Self {
        Self { mmio_ranges, max_call_levels: MAX_CALL_LEVELS }
    }

    /// Standard ARM Cortex-M MMIO range (covers peripheral bus + private peripherals).
    pub fn cortexm_default() -> Self {
        Self::new(vec![
            0x4000_0000..0x6000_0000, // APB/AHB peripherals
            0xE000_0000..0xF000_0000, // Private peripheral bus (NVIC, SysTick, etc.)
        ])
    }

    /// Find all MMIO access contexts that structurally govern the access to a new stream B at
    /// `readwatch_pc`.
    ///
    /// Returns an empty vec if no translated blocks are available or no governing contexts
    /// can be identified.  Each returned `AccessContext` carries the PC at which the source
    /// MMIO load appears, enabling the caller to build context-level edges in the graph.
    pub fn candidates_for_new_stream(
        &self,
        readwatch_pc: u64,
        code: &BlockTable,
    ) -> Vec<AccessContext> {
        let Some(block) = find_block_containing(code, readwatch_pc) else {
            return vec![];
        };

        // Seed the demand slice from the LOAD address varnode at `readwatch_pc`.
        let mut slice = seed_demand_from_block(block, readwatch_pc);
        if slice.demanded.is_empty() {
            // Fallback: seed from the block's branch condition — at minimum we record
            // any MMIO value that drives control flow to this block.
            if let Some(cond) = block.exit.cond() {
                if let Value::Var(var) = cond {
                    if !var.is_invalid() {
                        slice.demanded.insert(var.id);
                    }
                }
            }
        }

        if slice.demanded.is_empty() {
            return vec![];
        }

        let reverse_cfg = build_reverse_cfg(code);

        // Backward CFG traversal: intra-function (depth=0) + inter-function up to
        // `max_call_levels` call-stack levels.
        self.backward_traverse(code, &reverse_cfg, block.start, &mut slice, 0);

        slice.sources.into_iter().collect()
    }

    fn backward_traverse(
        &self,
        code: &BlockTable,
        reverse_cfg: &ReverseCfg,
        start_addr: u64,
        slice: &mut DemandSlice,
        call_depth: usize,
    ) {
        // Work-list of (block_start_addr, call_depth_at_which_this_block_was_reached).
        let mut visited: HashSet<u64> = HashSet::new();
        let mut worklist: Vec<(u64, usize)> = vec![(start_addr, call_depth)];

        while let Some((addr, depth)) = worklist.pop() {
            if !visited.insert(addr) {
                continue;
            }

            let Some(block) = find_block_containing(code, addr) else {
                continue;
            };

            // Process the block first so the seed block's P-code temporaries are
            // resolved before they are stripped.  Temporaries don't survive to
            // predecessor blocks, so strip them after processing.
            slice.process_block_backward(block, &self.mmio_ranges);
            slice.strip_temporaries();

            let Some(predecessors) = reverse_cfg.get(&block.start) else {
                continue;
            };

            for &(pred_start, is_call) in predecessors {
                if is_call {
                    // This predecessor is in the caller function — crossing a function
                    // boundary.  Only follow if we have remaining call depth.
                    if depth < self.max_call_levels {
                        worklist.push((pred_start, depth + 1));
                    }
                } else {
                    // Same function — continue at the same call depth.
                    worklist.push((pred_start, depth));
                }
            }
        }
    }
}

/// Seed the demand slice from the LOAD instruction at (or immediately before) `pc`
/// inside `block`.  The LOAD's address input varnode is added to the demand set.
fn seed_demand_from_block(block: &Block, pc: u64) -> DemandSlice {
    let mut current_pc = block.start;
    let mut candidate: Option<pcode::VarNode> = None;

    for stmt in &block.pcode.instructions {
        if stmt.op == Op::InstructionMarker {
            current_pc = stmt.inputs.first().as_u64();
        }
        // Collect LOAD address varnodes for instructions up to and including `pc`.
        if current_pc <= pc {
            if let Op::Load(_) = stmt.op {
                if let Value::Var(addr_var) = stmt.inputs.first() {
                    if !addr_var.is_invalid() {
                        candidate = Some(addr_var);
                    }
                }
            }
        }
    }

    match candidate {
        Some(vn) => DemandSlice::seed(vn),
        None => DemandSlice::default(),
    }
}
