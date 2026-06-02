use std::ops::Range;

use hashbrown::{HashMap, HashSet};
use icicle_vm::{
    cpu::lifter::{Block, BlockExit, Target},
    BlockTable,
};
use pcode::{Op, Value, VarId};

use crate::{
    input::StreamKey,
    stream_relation::{AccessContext, EdgeKind},
};

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
/// Two independent provenance channels are tracked so that the extracted edge
/// can be classified:
///   * `demanded_data` — varnodes that flow into B's MMIO *address/value*.  MMIO
///     loads reached through this channel produce `EdgeKind::Address` candidates.
///   * `demanded_ctrl` — varnodes that flow into a branch condition gating the
///     path to B.  MMIO loads reached through this channel produce
///     `EdgeKind::Control` candidates.
///
/// Temporaries (VarId < 0) are tracked within a block but NOT carried across
/// block boundaries, because P-code temporaries are local to a block.
#[derive(Default, Clone)]
pub struct DemandSlice {
    /// Data/address-provenance demands.
    pub demanded_data: HashSet<VarId>,
    /// Control-provenance demands (branch conditions on the path to B).
    pub demanded_ctrl: HashSet<VarId>,
    /// MMIO contexts reached via the data channel → Address edges.
    pub sources_addr:  HashSet<AccessContext>,
    /// MMIO contexts reached via the control channel → Control edges.
    pub sources_ctrl:  HashSet<AccessContext>,
}

impl DemandSlice {
    /// Seed the slice from a single varnode (the address operand of B's LOAD),
    /// which is data/address provenance.
    pub fn seed(vn: pcode::VarNode) -> Self {
        let mut s = Self::default();
        if !vn.is_invalid() {
            s.demanded_data.insert(vn.id);
        }
        s
    }

    /// Total number of outstanding demands (both channels).  Used to detect when a
    /// block's processing changed the slice (fixpoint progress).
    fn demand_count(&self) -> usize {
        self.demanded_data.len() + self.demanded_ctrl.len()
    }

    /// Total number of discovered sources (both channels).
    fn source_count(&self) -> usize {
        self.sources_addr.len() + self.sources_ctrl.len()
    }

    /// Process one block backward.  Demands are expanded through P-code ops along the
    /// channel they arrived on; MMIO LOADs are recorded as channel-tagged sources.
    ///
    /// `inject_control` adds this block's exit branch condition to the control channel.
    /// It must be `false` for the block that *contains* B (its branch fires after B and
    /// therefore does not gate B), and `true` for predecessor blocks whose branch gates
    /// the path toward B.
    pub fn process_block_backward(
        &mut self,
        block: &Block,
        mmio_ranges: &[Range<u64>],
        read_sites: &HashMap<u64, StreamKey>,
        inject_control: bool,
    ) {
        // Forward pass to map (instruction_index → current_pc) for source contexts.
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
            if out.is_invalid() {
                continue;
            }
            // Which channel(s) demanded this definition?
            let in_data = self.demanded_data.remove(&out.id);
            let in_ctrl = self.demanded_ctrl.remove(&out.id);
            if !in_data && !in_ctrl {
                continue;
            }

            // Input varnodes to propagate backward (along the same channel(s)).
            let mut ins: Vec<VarId> = Vec::new();

            match stmt.op {
                Op::Load(_) => {
                    let load_pc = stmt_pcs[idx];
                    // Register-indirect MMIO reads (`ldr r0,[rN]`) lift to a Load with a *Var*
                    // address — the lifter never folds the literal-pool base into a constant
                    // (Op::Load is opaque to const-propagation).  So recognise MMIO loads by
                    // the runtime-observed (pc → stream address) map, which captures the PC of
                    // every MMIO read site that has triggered a ReadWatch.
                    if let Some(&saddr) = read_sites.get(&load_pc) {
                        let ctx = AccessContext::new(load_pc, saddr);
                        if in_data {
                            self.sources_addr.insert(ctx);
                        }
                        if in_ctrl {
                            self.sources_ctrl.insert(ctx);
                        }
                    }
                    match stmt.inputs.first() {
                        Value::Const(addr, _) => {
                            // Complementary path: absolute-addressed MMIO (rare; e.g. some PPB
                            // accesses) where the address survives as a constant.
                            if mmio_ranges.iter().any(|r| r.contains(&addr)) {
                                let ctx = AccessContext::new(load_pc, addr);
                                if in_data {
                                    self.sources_addr.insert(ctx);
                                }
                                if in_ctrl {
                                    self.sources_ctrl.insert(ctx);
                                }
                            }
                        }
                        Value::Var(addr_var) if !addr_var.is_invalid() => {
                            // Dynamic address: also trace how the pointer was computed (it may
                            // be an indexed access whose offset is itself MMIO-derived).
                            ins.push(addr_var.id);
                        }
                        _ => {}
                    }
                }

                Op::Copy | Op::ZeroExtend | Op::SignExtend | Op::Subpiece(_) => {
                    if let Value::Var(src) = stmt.inputs.first() {
                        if !src.is_invalid() {
                            ins.push(src.id);
                        }
                    }
                }

                // Binary arithmetic and comparison ops: demand all variable inputs.
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
                                ins.push(var.id);
                            }
                        }
                    }
                }

                // Unary arithmetic ops.
                Op::IntNot | Op::IntNegate | Op::IntCountOnes | Op::IntCountLeadingZeroes
                | Op::BoolNot => {
                    if let Value::Var(src) = stmt.inputs.first() {
                        if !src.is_invalid() {
                            ins.push(src.id);
                        }
                    }
                }

                // Stores/phi/indirection/hooks: conservatively skip (no def of `out`).
                _ => {}
            }

            // Propagate each input on the channel(s) that demanded this definition.
            for id in ins {
                if in_data {
                    self.demanded_data.insert(id);
                }
                if in_ctrl {
                    self.demanded_ctrl.insert(id);
                }
            }
        }

        // Control-dependence: a predecessor block's branch condition gates the path to B.
        if inject_control {
            if let Some(cond) = block.exit.cond() {
                if let Value::Var(var) = cond {
                    if !var.is_invalid() {
                        self.demanded_ctrl.insert(var.id);
                    }
                }
            }
        }
    }

    /// Strip temporaries from both demand channels.  Called when carrying demands across
    /// a block boundary: P-code temporaries (VarId < 0) don't survive across blocks.
    pub fn strip_temporaries(&mut self) {
        self.demanded_data.retain(|id| *id >= 0);
        self.demanded_ctrl.retain(|id| *id >= 0);
    }

    /// Merge another slice into this one (CFG join point).
    pub fn merge(&mut self, other: &DemandSlice) {
        self.demanded_data.extend(other.demanded_data.iter().copied());
        self.demanded_ctrl.extend(other.demanded_ctrl.iter().copied());
        self.sources_addr.extend(other.sources_addr.iter().copied());
        self.sources_ctrl.extend(other.sources_ctrl.iter().copied());
    }

    /// Collapse discovered sources into classified `(context, kind)` candidates.
    /// Address provenance takes priority over Control when a context appears on both.
    fn into_candidates(self) -> Vec<(AccessContext, EdgeKind)> {
        let mut map: HashMap<AccessContext, EdgeKind> = HashMap::new();
        for ctx in self.sources_addr {
            map.insert(ctx, EdgeKind::Address);
        }
        for ctx in self.sources_ctrl {
            map.entry(ctx).or_insert(EdgeKind::Control);
        }
        map.into_iter().collect()
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
    /// Runtime-observed MMIO read sites: instruction PC → stream address.  Populated at each
    /// ReadWatch so the structural slice can recognise register-indirect MMIO loads.
    pub mmio_read_sites: HashMap<u64, StreamKey>,
}

impl MmioFlowAnalyzer {
    pub fn new(mmio_ranges: Vec<Range<u64>>) -> Self {
        Self { mmio_ranges, max_call_levels: MAX_CALL_LEVELS, mmio_read_sites: HashMap::new() }
    }

    /// Standard ARM Cortex-M MMIO range (covers peripheral bus + private peripherals).
    pub fn cortexm_default() -> Self {
        Self::new(vec![
            0x4000_0000..0x6000_0000, // APB/AHB peripherals
            0xE000_0000..0xF000_0000, // Private peripheral bus (NVIC, SysTick, etc.)
        ])
    }

    /// Record an observed MMIO read site (called at each ReadWatch).  This is what lets the
    /// backward slice identify register-indirect MMIO loads as dependency sources.
    pub fn record_read_site(&mut self, pc: u64, addr: StreamKey) {
        self.mmio_read_sites.insert(pc, addr);
    }

    /// Find all MMIO access contexts that structurally govern the access to a new stream B at
    /// `readwatch_pc`, each classified as an `Address` or `Control` dependency.
    ///
    /// Returns an empty vec if no translated blocks are available or no governing contexts
    /// can be identified.  Each returned `AccessContext` carries the PC at which the source
    /// MMIO load appears, enabling the caller to build context-level edges in the graph.
    pub fn candidates_for_new_stream(
        &self,
        readwatch_pc: u64,
        code: &BlockTable,
    ) -> Vec<(AccessContext, EdgeKind)> {
        let Some(block) = find_block_containing(code, readwatch_pc) else {
            return vec![];
        };

        // Seed the data channel from the LOAD address varnode at `readwatch_pc`.  The control
        // channel is seeded lazily as predecessor branch conditions are encountered during the
        // backward traversal, so we always traverse even if the data seed is empty.
        let mut slice = seed_demand_from_block(block, readwatch_pc);

        let reverse_cfg = build_reverse_cfg(code);

        // Backward CFG traversal: intra-function (depth=0) + inter-function up to
        // `max_call_levels` call-stack levels.
        self.backward_traverse(code, &reverse_cfg, block.start, &mut slice, 0);

        slice.into_candidates()
    }

    fn backward_traverse(
        &self,
        code: &BlockTable,
        reverse_cfg: &ReverseCfg,
        start_addr: u64,
        slice: &mut DemandSlice,
        call_depth: usize,
    ) {
        /// Bound on how many times a single block may be re-processed.  Allows demands that
        /// arrive late (via another path) to still propagate, while guaranteeing termination.
        const MAX_BLOCK_VISITS: u32 = 8;

        // Work-list of (block_start_addr, call_depth_at_which_this_block_was_reached).
        let mut visits: HashMap<u64, u32> = HashMap::new();
        let mut worklist: Vec<(u64, usize)> = vec![(start_addr, call_depth)];

        while let Some((addr, depth)) = worklist.pop() {
            let count = visits.entry(addr).or_insert(0);
            if *count >= MAX_BLOCK_VISITS {
                continue;
            }
            *count += 1;
            let first_visit = *count == 1;

            let Some(block) = find_block_containing(code, addr) else {
                continue;
            };

            // The block containing B (start_addr) must not inject its own branch condition as
            // a control dependency (that branch fires after B).  Predecessor blocks do.
            let inject_control = addr != start_addr;

            let before = (slice.demand_count(), slice.source_count());
            slice.process_block_backward(
                block,
                &self.mmio_ranges,
                &self.mmio_read_sites,
                inject_control,
            );
            slice.strip_temporaries();
            let after = (slice.demand_count(), slice.source_count());

            // Re-explore predecessors on the first visit, or whenever this block changed the
            // slice (new demands generated or sources found) so late demands keep flowing.
            if first_visit || before != after {
                if let Some(predecessors) = reverse_cfg.get(&block.start) {
                    for &(pred_start, is_call) in predecessors {
                        if is_call {
                            // Crossing a function boundary into the caller — follow only if
                            // call depth remains.
                            if depth < self.max_call_levels {
                                worklist.push((pred_start, depth + 1));
                            }
                        } else {
                            worklist.push((pred_start, depth));
                        }
                    }
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
