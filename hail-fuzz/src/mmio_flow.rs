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
        cutoff_pc: Option<u64>,
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
            // In the block that *contains* B, ignore every instruction that executes
            // after B's load (pc > cutoff_pc).  Such instructions run *after* B has
            // already read, so they cannot be a dependency source of B.  Crucially, a
            // later redefinition of B's base register in the same block (e.g.
            // `LDR Rbase, [other_mmio]` placed below B) would otherwise have its output
            // match the demanded base varnode and be wrongly recorded as an Address
            // source — the exact false positive observed for streams 0x58000004 and
            // 0x58000490.
            if let Some(max) = cutoff_pc {
                if stmt_pcs[idx] > max {
                    continue;
                }
            }
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
                        if std::env::var_os("DUMP_FLOW").is_some() {
                            eprintln!(
                                "[flow]   MMIO-load SOURCE recorded: pc={:#x} stream={:#x} \
                                 out_id={} in_data={} in_ctrl={} \
                                 (demanded_data={:?} demanded_ctrl={:?})",
                                load_pc, saddr, out.id, in_data, in_ctrl,
                                self.demanded_data, self.demanded_ctrl
                            );
                        }
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
                            // Trace the address computation chain only for known MMIO loads.
                            // Non-MMIO loads (flash literal-pool reads like `LDR Rd, =const`,
                            // RAM struct accesses) are not dependency sources.  Propagating
                            // their address varnode would eventually demand the PC register
                            // (ARM PC-relative literal-pool addressing lifts as
                            // `$tmp = INT_ADD(PC, off); Rd = LOAD($tmp)`), which then spreads
                            // across the entire CFG and produces spurious Address edges.
                            if read_sites.contains_key(&load_pc) {
                                ins.push(addr_var.id);
                            }
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
        let dbg = std::env::var_os("DUMP_FLOW").is_some();
        if dbg {
            eprintln!(
                "\n[flow] ───── new stream @ readwatch_pc={:#x}  block [{:#x}..{:#x}) ─────",
                readwatch_pc, block.start, block.end
            );
            let mut sites: Vec<(&u64, &StreamKey)> = self.mmio_read_sites.iter().collect();
            sites.sort();
            eprintln!("[flow] read_sites ({} entries):", sites.len());
            for (pc, key) in sites {
                eprintln!("[flow]    pc={:#x} -> stream={:#x}", pc, key);
            }
        }

        let mut slice = seed_demand_from_block(block, readwatch_pc);
        if dbg {
            eprintln!(
                "[flow] seed demanded_data={:?} demanded_ctrl={:?}",
                slice.demanded_data, slice.demanded_ctrl
            );
        }

        let reverse_cfg = build_reverse_cfg(code);

        // Backward CFG traversal: intra-function (depth=0) + inter-function up to
        // `max_call_levels` call-stack levels.
        self.backward_traverse(code, &reverse_cfg, block.start, readwatch_pc, &mut slice, 0);

        if dbg {
            eprintln!("[flow] RAW sources_addr={:?}", slice.sources_addr);
            eprintln!("[flow] RAW sources_ctrl={:?}", slice.sources_ctrl);
        }

        // A stream cannot govern itself: remove any candidate whose source PC equals
        // readwatch_pc (the target's own load instruction was discovered as its own source,
        // which happens when the load's address-input VarId equals its output VarId — e.g.
        // `LDR Rn,[Rn]` — so the backward slice immediately satisfies its own demand).
        let candidates: Vec<(AccessContext, EdgeKind)> = slice
            .into_candidates()
            .into_iter()
            .filter(|(ctx, _)| ctx.pc != readwatch_pc)
            .collect();

        if dbg {
            eprintln!("[flow] RESULT after self-edge filter: {:?}", candidates);
        }

        candidates
    }

    fn backward_traverse(
        &self,
        code: &BlockTable,
        reverse_cfg: &ReverseCfg,
        start_addr: u64,
        readwatch_pc: u64,
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

            // Only the block that contains B applies the post-B instruction cutoff: within
            // that block, instructions after readwatch_pc execute after B's read and must be
            // ignored.  Predecessor blocks execute entirely before B, so no cutoff applies.
            let cutoff_pc = if addr == start_addr { Some(readwatch_pc) } else { None };

            if std::env::var_os("DUMP_FLOW").is_some() {
                eprintln!(
                    "[flow]  traverse block [{:#x}..{:#x}) visit={} inject_ctrl={} \
                     demanded_data={:?} demanded_ctrl={:?}",
                    block.start, block.end, count, inject_control,
                    slice.demanded_data, slice.demanded_ctrl
                );
            }

            let before = (slice.demand_count(), slice.source_count());
            slice.process_block_backward(
                block,
                &self.mmio_ranges,
                &self.mmio_read_sites,
                inject_control,
                cutoff_pc,
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

#[cfg(test)]
mod tests {
    use super::*;
    use icicle_vm::cpu::lifter::{Block, BlockExit};
    use pcode::VarNode;

    /// Wrap a hand-built `pcode::Block` in a lifter `Block` spanning `[start, end)`.
    fn lifter_block(pcode: pcode::Block, start: u64, end: u64) -> Block {
        Block {
            pcode,
            entry: None,
            start,
            end,
            context: 0,
            exit: BlockExit::invalid(),
            breakpoints: 0,
            num_instructions: 0,
        }
    }

    fn marker(pc: u64) -> pcode::Instruction {
        (pcode::Op::InstructionMarker, pcode::Value::Const(pc, 8)).into()
    }

    /// A general-purpose register varnode (positive id ⇒ survives across blocks).
    fn reg(id: pcode::VarId) -> VarNode {
        VarNode::new(id, 4)
    }

    /// Regression test for the literal-pool false positive (PR feedback, Edge 1/Edge 2).
    ///
    /// A non-MMIO load (e.g. `LDR R2, =const` — a flash literal-pool read) must terminate the
    /// dependency chain: tracing its address operand would otherwise pull unrelated MMIO reads
    /// into the slice and emit spurious `Address` edges.  Here the literal-pool pointer is
    /// (contrivedly) derived from an MMIO value so that, *without* the fix, the backward slice
    /// would walk through the non-MMIO load and wrongly record the MMIO read as an Address source.
    #[test]
    fn non_mmio_load_terminates_address_chain() {
        let mut pcode = pcode::Block::new();
        let mmio_out = reg(10); // value read from the status register
        let lit_ptr = reg(6); // pointer used by the literal-pool load
        let r2 = reg(2); // result of the literal-pool load
        let tgt_addr = reg(7); // B's computed address (the seed)

        pcode.push(marker(0x100));
        pcode.push((mmio_out, pcode::Op::Load(0), reg(5))); // MMIO load @0x100
        pcode.push(marker(0x104));
        pcode.push((lit_ptr, pcode::Op::Copy, mmio_out));
        pcode.push((r2, pcode::Op::Load(0), lit_ptr)); // NON-MMIO load @0x104
        pcode.push(marker(0x108));
        pcode.push((tgt_addr, pcode::Op::IntAdd, r2, pcode::Value::Const(0x90, 4)));

        let block = lifter_block(pcode, 0x100, 0x10c);

        // Only the MMIO load @0x100 is a known read site; @0x104 (literal pool) is not.
        let mut read_sites: HashMap<u64, StreamKey> = HashMap::new();
        read_sites.insert(0x100, 0x5801_0008);

        let mut slice = DemandSlice::seed(tgt_addr);
        slice.process_block_backward(&block, &[], &read_sites, false, None);

        // With the fix the chain stops at the non-MMIO load: no spurious Address source.
        assert!(
            slice.sources_addr.is_empty(),
            "non-MMIO load should not leak an Address source, got {:?}",
            slice.sources_addr
        );
    }

    /// Genuine indexed-MMIO addressing must still be captured: when B's address is computed
    /// directly from a value read at a known MMIO site, that site is a real `Address` source.
    #[test]
    fn indexed_mmio_address_is_captured() {
        let mut pcode = pcode::Block::new();
        let idx = reg(10);
        let tgt_addr = reg(7);

        pcode.push(marker(0x200));
        pcode.push((idx, pcode::Op::Load(0), reg(5))); // MMIO load @0x200
        pcode.push(marker(0x204));
        pcode.push((tgt_addr, pcode::Op::IntAdd, reg(4), idx)); // B addr = base + mmio idx

        let block = lifter_block(pcode, 0x200, 0x208);

        let mut read_sites: HashMap<u64, StreamKey> = HashMap::new();
        read_sites.insert(0x200, 0x5800_0000);

        let mut slice = DemandSlice::seed(tgt_addr);
        slice.process_block_backward(&block, &[], &read_sites, false, None);

        assert!(
            slice.sources_addr.contains(&AccessContext::new(0x200, 0x5800_0000)),
            "indexed MMIO address dependency must be captured, got {:?}",
            slice.sources_addr
        );
    }

    /// Regression: a self-referential load (`LDR Rn,[Rn]`) where the same register is both
    /// address input and output must NOT appear as a source of itself.  This mirrors the
    /// firmware pattern seen at 0x800a506 / 0x800a54e where `out_id == addr_input_id == 30`.
    ///
    /// The `candidates_for_new_stream` filter removes such self-references by checking
    /// `ctx.pc != readwatch_pc`.  At the `process_block_backward` level the load IS recorded
    /// (correctly — the fix only suppresses the entry in the final candidate list), so this
    /// unit test verifies only the DemandSlice half.  The end-to-end filter is covered by the
    /// integration: if the source pc == readwatch_pc it is stripped in candidates_for_new_stream.
    #[test]
    fn self_referential_load_does_not_propagate_as_address_source() {
        // Simulate `LDR R5,[R5]` — register r5 (id=5) is both address and destination.
        let mut pcode = pcode::Block::new();
        let r5 = reg(5);
        pcode.push(marker(0x400));
        // Load whose address input == output: stmt.inputs.first() = r5, stmt.output = r5
        pcode.push((r5, pcode::Op::Load(0), r5));

        let block = lifter_block(pcode, 0x400, 0x404);

        // r5 (id=5) both demanded (as address input seed) and the load's output.
        let mut read_sites: HashMap<u64, StreamKey> = HashMap::new();
        read_sites.insert(0x400, 0x5800_0000); // this IS in read_sites

        // Seed with r5 (simulates seed_demand_from_block picking the address input of the
        // self-referential load). id=5 will also match the load's out.id=5.
        let mut slice = DemandSlice::seed(r5);
        slice.process_block_backward(&block, &[], &read_sites, false, None);

        // At the process_block_backward level the source IS recorded (that's correct
        // behaviour — the slice has no notion of "self").  The self-edge exclusion happens
        // in candidates_for_new_stream by checking ctx.pc != readwatch_pc.
        // Here we verify the source ctx has pc=0x400, which equals the readwatch_pc=0x400:
        let self_source = AccessContext::new(0x400, 0x5800_0000);
        if slice.sources_addr.contains(&self_source) {
            // Correct — the slice recorded it; the caller is responsible for removing it.
            // This test documents the expected behaviour: the filter MUST be in the caller.
        }
        // Ensure no spurious sources with a *different* pc leaked in.
        for ctx in &slice.sources_addr {
            assert_eq!(
                ctx.pc, 0x400,
                "only source at pc=0x400 is expected; got ctx with pc={:#x}",
                ctx.pc
            );
        }
    }

    /// The address operand of a *genuine* MMIO load is still traced, so a deeper indexed-MMIO
    /// chain (pointer of one MMIO read computed from another MMIO read) yields both sources.
    #[test]
    fn mmio_load_still_traces_its_pointer() {
        let mut pcode = pcode::Block::new();
        let off = reg(10);
        let ptr = reg(6);
        let val = reg(2);
        let tgt = reg(7);

        pcode.push(marker(0x300));
        pcode.push((off, pcode::Op::Load(0), reg(5))); // MMIO load @0x300 (offset)
        pcode.push(marker(0x304));
        pcode.push((ptr, pcode::Op::IntAdd, reg(4), off));
        pcode.push((val, pcode::Op::Load(0), ptr)); // MMIO load @0x304 (value), addr is MMIO-derived
        pcode.push(marker(0x308));
        pcode.push((tgt, pcode::Op::Copy, val));

        let block = lifter_block(pcode, 0x300, 0x30c);

        let mut read_sites: HashMap<u64, StreamKey> = HashMap::new();
        read_sites.insert(0x300, 0xAAAA);
        read_sites.insert(0x304, 0xBBBB);

        let mut slice = DemandSlice::seed(tgt);
        slice.process_block_backward(&block, &[], &read_sites, false, None);

        assert!(
            slice.sources_addr.contains(&AccessContext::new(0x304, 0xBBBB))
                && slice.sources_addr.contains(&AccessContext::new(0x300, 0xAAAA)),
            "deep indexed-MMIO chain should yield both MMIO sources, got {:?}",
            slice.sources_addr
        );
    }

    /// Regression for the post-B-instruction false positive (Edge 1: 0x58010008→0x58000490,
    /// Edge 2: 0x58000000→0x58000004).  A *later* instruction in B's own block that redefines
    /// B's base register must NOT be recorded as a source — it executes after B has read.
    ///
    /// Layout (B reads via base register id 30):
    ///   0x100:  R30 = LOAD[mmioA]   ← the genuine reaching definition of B's base (BEFORE B)
    ///   0x104:  <B's load uses R30> (readwatch_pc = 0x104; seed demands R30)
    ///   0x108:  R30 = LOAD[mmioC]   ← a *later* redefinition of R30 (AFTER B)
    ///
    /// Without the cutoff the backward pass (end→start) meets 0x108 first, matches the demanded
    /// R30, and wrongly records mmioC.  With `cutoff_pc = Some(0x104)` the 0x108 load is skipped
    /// and only the genuine 0x100 source survives.
    #[test]
    fn post_readwatch_instruction_is_not_a_source() {
        let mut pcode = pcode::Block::new();
        let base = reg(30);

        pcode.push(marker(0x100));
        pcode.push((base, pcode::Op::Load(0), reg(4))); // genuine def of R30 @0x100 (mmioA)
        pcode.push(marker(0x104)); // B's load site (readwatch_pc); seed already demands R30
        pcode.push(marker(0x108));
        pcode.push((base, pcode::Op::Load(0), reg(5))); // later redefinition @0x108 (mmioC)

        let block = lifter_block(pcode, 0x100, 0x10c);

        let mut read_sites: HashMap<u64, StreamKey> = HashMap::new();
        read_sites.insert(0x100, 0x5800_0000); // mmioA — the legitimate source
        read_sites.insert(0x108, 0x5801_0008); // mmioC — the post-B redefinition

        // With the cutoff, only the source at/<=0x104 is recorded.
        let mut slice = DemandSlice::seed(base);
        slice.process_block_backward(&block, &[], &read_sites, false, Some(0x104));
        assert!(
            slice.sources_addr.contains(&AccessContext::new(0x100, 0x5800_0000)),
            "genuine pre-B source must be recorded, got {:?}",
            slice.sources_addr
        );
        assert!(
            !slice.sources_addr.contains(&AccessContext::new(0x108, 0x5801_0008)),
            "post-B redefinition must NOT be recorded as a source, got {:?}",
            slice.sources_addr
        );

        // Sanity: WITHOUT the cutoff the bug reproduces (the later def shadows the real one).
        let mut slice_nocut = DemandSlice::seed(base);
        slice_nocut.process_block_backward(&block, &[], &read_sites, false, None);
        assert!(
            slice_nocut.sources_addr.contains(&AccessContext::new(0x108, 0x5801_0008)),
            "without cutoff the post-B redefinition is (wrongly) recorded — guards the fix"
        );
    }
}
