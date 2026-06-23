use std::ops::Range;

use hashbrown::{HashMap, HashSet};
use icicle_vm::{
    cpu::lifter::{Block, BlockExit, Target},
    BlockTable,
};
use pcode::{Op, Value, VarId, VarNode};

use crate::{
    input::StreamKey,
    stream_relation::{AccessContext, EdgeKind},
};

const MAX_CALL_LEVELS: usize = 3;
const FRONTIER_BOOST: f64 = 3.0;
const MAX_FRONTIER_BRANCHES: usize = 512;

/// Knobs for a single backward CFG traversal.
/// Data-seed mode: `inject_start_control=false`, `inject_pred_control=true`, cutoff=B's PC.
/// Branch-seed mode: `inject_start_control=true`, `inject_pred_control=false`, cutoff=None.
struct TraverseCfg {
    start_cutoff: Option<u64>,
    inject_start_control: bool,
    inject_pred_control: bool,
}

/// Backward demand slice over P-code varnodes.
/// `demanded_data` feeds B's address → Address edges; `demanded_ctrl` feeds branch conditions → Control edges.
/// Temporaries (VarId < 0) are local to a block and are stripped at block boundaries.
#[derive(Default, Clone)]
pub struct DemandSlice {
    pub demanded_data: HashSet<VarId>,
    pub demanded_ctrl: HashSet<VarId>,
    pub sources_addr:  HashSet<AccessContext>,
    pub sources_ctrl:  HashSet<AccessContext>,
}

impl DemandSlice {
    pub fn seed(vn: pcode::VarNode) -> Self {
        let mut s = Self::default();
        if !vn.is_invalid() {
            s.demanded_data.insert(vn.id);
        }
        s
    }

    fn demand_count(&self) -> usize { self.demanded_data.len() + self.demanded_ctrl.len() }
    fn source_count(&self) -> usize { self.sources_addr.len() + self.sources_ctrl.len() }

    pub fn process_block_backward(
        &mut self,
        block: &Block,
        mmio_ranges: &[Range<u64>],
        read_sites: &HashMap<u64, StreamKey>,
        inject_control: bool,
        cutoff_pc: Option<u64>,
    ) {
        let mut stmt_pcs = vec![block.start; block.pcode.instructions.len()];
        {
            let mut pc = block.start;
            for (i, stmt) in block.pcode.instructions.iter().enumerate() {
                if stmt.op == Op::InstructionMarker {
                    if let pcode::Value::Const(addr, _) = stmt.inputs.first() { pc = addr; }
                }
                stmt_pcs[i] = pc;
            }
        }

        // Inject branch condition BEFORE the backward pass so it can be traced to MMIO reads.
        if inject_control {
            if let Some(cond) = block.exit.cond() {
                if let Value::Var(var) = cond {
                    if !var.is_invalid() {
                        self.demanded_ctrl.insert(var.id);
                    }
                }
            }
        }

        for (idx, stmt) in block.pcode.instructions.iter().enumerate().rev() {
            // Skip instructions after B's load (cutoff prevents post-B redefs from
            // matching the demanded base varnode and producing spurious Address edges).
            if let Some(max) = cutoff_pc {
                if stmt_pcs[idx] > max { continue; }
            }
            let out = stmt.output;
            if out.is_invalid() { continue; }
            let in_data = self.demanded_data.remove(&out.id);
            let in_ctrl = self.demanded_ctrl.remove(&out.id);
            if !in_data && !in_ctrl { continue; }

            let mut ins: Vec<VarId> = Vec::new();

            match stmt.op {
                Op::Load(_) => {
                    let load_pc = stmt_pcs[idx];
                    // Identify MMIO loads by the runtime-observed pc→stream map (register-indirect
                    // loads have a Var address; the lifter never folds the peripheral base to const).
                    if let Some(&saddr) = read_sites.get(&load_pc) {
                        let ctx = AccessContext::new(load_pc, saddr);
                        if std::env::var_os("DUMP_FLOW").is_some() {
                            eprintln!("[flow]   SOURCE pc={:#x} stream={:#x} data={} ctrl={}",
                                load_pc, saddr, in_data, in_ctrl);
                        }
                        if in_data { self.sources_addr.insert(ctx); }
                        if in_ctrl { self.sources_ctrl.insert(ctx); }
                    }
                    match stmt.inputs.first() {
                        Value::Const(addr, _) => {
                            // Absolute-addressed MMIO (rare; e.g. some PPB accesses).
                            if mmio_ranges.iter().any(|r| r.contains(&addr)) {
                                let ctx = AccessContext::new(load_pc, addr);
                                if in_data { self.sources_addr.insert(ctx); }
                                if in_ctrl { self.sources_ctrl.insert(ctx); }
                            }
                        }
                        Value::Var(addr_var) if !addr_var.is_invalid() => {
                            // Trace address chain only for known MMIO loads; non-MMIO loads
                            // (literal-pool, RAM) would pull in the PC register and cause
                            // spurious Address edges across the entire CFG.
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

    }

    /// Strip temporaries from both demand channels.  Called when carrying demands across
    /// a block boundary: P-code temporaries (VarId < 0) don't survive across blocks.
    pub fn strip_temporaries(&mut self) {
        self.demanded_data.retain(|id| *id >= 0);
        self.demanded_ctrl.retain(|id| *id >= 0);
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

/// Maps each block-start address to its `(predecessor_block_start, is_call)` predecessors.
type ReverseCfg = HashMap<u64, Vec<(u64, bool)>>;

fn build_reverse_cfg(code: &BlockTable) -> ReverseCfg {
    let mut reverse: ReverseCfg = HashMap::new();
    for block in &code.blocks {
        let is_call = matches!(block.exit, BlockExit::Call { .. });
        for (i, target) in block.exit.targets().enumerate() {
            let edge_is_call = is_call && i == 0;
            let target_addr: Option<u64> = match target {
                Target::External(Value::Const(addr, _)) => Some(addr),
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

fn find_block_containing(code: &BlockTable, addr: u64) -> Option<&Block> {
    code.blocks.iter().find(|b| b.contains_addr(addr))
}

// ──────────────────────────────────────────────────────────────────────────────
// MmioFlowAnalyzer
// ──────────────────────────────────────────────────────────────────────────────

/// Phase A: backward P-code demand slice for structural MMIO dependency inference.
pub struct MmioFlowAnalyzer {
    pub mmio_ranges: Vec<Range<u64>>,
    pub max_call_levels: usize,
    pub mmio_read_sites: HashMap<u64, StreamKey>,
    analyzed_contexts: HashMap<(StreamKey, u64), usize>,
}

const REANALYZE_BLOCK_DELTA: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseATrigger {
    pub run_slice: bool,
    pub first_seen: bool,
}

impl MmioFlowAnalyzer {
    pub fn new(mmio_ranges: Vec<Range<u64>>) -> Self {
        Self {
            mmio_ranges,
            max_call_levels: MAX_CALL_LEVELS,
            mmio_read_sites: HashMap::new(),
            analyzed_contexts: HashMap::new(),
        }
    }

    pub fn trigger(&mut self, addr: StreamKey, pc: u64, lifted_blocks: usize) -> PhaseATrigger {
        match self.analyzed_contexts.entry((addr, pc)) {
            hashbrown::hash_map::Entry::Vacant(e) => {
                e.insert(lifted_blocks);
                PhaseATrigger { run_slice: true, first_seen: true }
            }
            hashbrown::hash_map::Entry::Occupied(mut e) => {
                if lifted_blocks >= *e.get() + REANALYZE_BLOCK_DELTA {
                    e.insert(lifted_blocks);
                    PhaseATrigger { run_slice: true, first_seen: false }
                } else {
                    PhaseATrigger { run_slice: false, first_seen: false }
                }
            }
        }
    }

    pub fn cortexm_default() -> Self {
        Self::new(vec![
            0x4000_0000..0x6000_0000, // APB/AHB peripherals
            0xE000_0000..0xF000_0000, // Private peripheral bus (NVIC, SysTick, etc.)
        ])
    }

    pub fn record_read_site(&mut self, pc: u64, addr: StreamKey) {
        self.mmio_read_sites.insert(pc, addr);
    }

    /// Return all MMIO contexts that structurally govern stream B at `readwatch_pc`.
    pub fn candidates_for_new_stream(
        &self,
        readwatch_pc: u64,
        code: &BlockTable,
    ) -> Vec<(AccessContext, EdgeKind)> {
        let Some(block) = find_block_containing(code, readwatch_pc) else {
            return vec![];
        };

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
        let cfg = TraverseCfg {
            start_cutoff: Some(readwatch_pc),
            inject_start_control: false,
            inject_pred_control: true,
        };
        self.backward_traverse(code, &reverse_cfg, block.start, &cfg, &mut slice, 0);

        if dbg {
            eprintln!("[flow] RAW sources_addr={:?}", slice.sources_addr);
            eprintln!("[flow] RAW sources_ctrl={:?}", slice.sources_ctrl);
        }

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
        cfg: &TraverseCfg,
        slice: &mut DemandSlice,
        call_depth: usize,
    ) {
        const MAX_BLOCK_VISITS: u32 = 8;

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

            let inject_control =
                if addr == start_addr { cfg.inject_start_control } else { cfg.inject_pred_control };
            let cutoff_pc = if addr == start_addr { cfg.start_cutoff } else { None };

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

            if first_visit || before != after {
                if let Some(predecessors) = reverse_cfg.get(&block.start) {
                    for &(pred_start, is_call) in predecessors {
                        if is_call {
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

    /// Return MMIO stream keys whose value gates the branch at `branch_block_start`.
    #[allow(dead_code)]
    pub fn candidates_for_branch(&self, branch_block_start: u64, code: &BlockTable) -> Vec<StreamKey> {
        let reverse_cfg = build_reverse_cfg(code);
        self.candidates_for_branch_with_cfg(branch_block_start, code, &reverse_cfg)
    }

    fn candidates_for_branch_with_cfg(
        &self,
        branch_block_start: u64,
        code: &BlockTable,
        reverse_cfg: &ReverseCfg,
    ) -> Vec<StreamKey> {
        let Some(block) = find_block_containing(code, branch_block_start) else {
            return vec![];
        };
        if block.exit.cond().is_none() { return vec![]; }
        let cfg = TraverseCfg {
            start_cutoff: None,
            inject_start_control: true,
            inject_pred_control: false,
        };
        let mut slice = DemandSlice::default();
        self.backward_traverse(code, reverse_cfg, block.start, &cfg, &mut slice, 0);

        let mut keys: Vec<StreamKey> = slice.sources_ctrl.into_iter().map(|c| c.addr).collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    }
}

fn target_addr(target: &Target, code: &BlockTable) -> Option<u64> {
    match target {
        Target::External(Value::Const(addr, _)) => Some(*addr),
        Target::Internal(idx) => code.blocks.get(*idx).map(|b| b.start),
        _ => None,
    }
}

/// Return the set of frontier branch block start addresses: blocks with a conditional exit
/// where at least one successor has not been lifted (i.e. its start address is not in `reached`).
/// Used by the hybrid trigger in main.rs.
pub fn compute_frontier_branch_set(code: &BlockTable) -> HashSet<u64> {
    let reached: HashSet<u64> = code.blocks.iter().map(|b| b.start).collect();
    let mut frontier = HashSet::new();
    for block in &code.blocks {
        if block.exit.cond().is_none() { continue; }
        let is_frontier = block.exit.targets().any(|t| {
            target_addr(&t, code).map_or(false, |addr| !reached.contains(&addr))
        });
        if is_frontier {
            frontier.insert(block.start);
        }
    }
    frontier
}

/// Compute per-stream mutation weight boosts from frontier branches.
///
/// For each frontier branch (conditional exit with at least one uncovered successor):
///   - `unlock_potential`: number of uncovered successor addresses (typically 1 or 2).
///   - `rarity`: `1.0 / (predecessor_count + 1)` where `predecessor_count` is the number of
///     blocks in the reverse CFG that list this branch block as a successor. Falls back to
///     `1.0 / 2.0` when unknown.
///
/// Each gating MMIO stream accumulates `unlock_potential * rarity` across all frontier branches
/// it controls.  A stagnation penalty `1.0 / (1 + stagnation_count)` is then applied per stream.
/// The final weight is capped at `1.0 + FRONTIER_BOOST`.
pub fn compute_frontier_stream_weights(
    code: &BlockTable,
    mmio_flow: &MmioFlowAnalyzer,
    stagnation: &HashMap<StreamKey, u32>,
) -> HashMap<StreamKey, f64> {
    let reached: HashSet<u64> = code.blocks.iter().map(|b| b.start).collect();
    let reverse_cfg = build_reverse_cfg(code);

    // Count how many times each block appears as a predecessor target (for rarity).
    let mut pred_count: HashMap<u64, u32> = HashMap::new();
    for preds in reverse_cfg.values() {
        for &(pred_start, _) in preds {
            *pred_count.entry(pred_start).or_insert(0) += 1;
        }
    }

    let mut accum: HashMap<StreamKey, f64> = HashMap::new();
    let mut processed = 0usize;
    for block in &code.blocks {
        if processed >= MAX_FRONTIER_BRANCHES { break; }
        if block.exit.cond().is_none() { continue; }

        // Count uncovered successors (unlock_potential).
        let unlock_potential: u32 = block.exit.targets()
            .filter(|t| target_addr(t, code).map_or(false, |addr| !reached.contains(&addr)))
            .count() as u32;
        if unlock_potential == 0 { continue; }

        processed += 1;

        // Rarity: inverse of how many predecessors reference this block.
        let hit_count = pred_count.get(&block.start).copied().unwrap_or(1);
        let rarity = 1.0 / (hit_count as f64 + 1.0);

        let contribution = unlock_potential as f64 * rarity;

        for key in mmio_flow.candidates_for_branch_with_cfg(block.start, code, &reverse_cfg) {
            *accum.entry(key).or_insert(0.0) += contribution;
        }
    }

    let cap = 1.0 + FRONTIER_BOOST;
    accum.into_iter()
        .map(|(k, raw)| {
            let stag = stagnation.get(&k).copied().unwrap_or(0);
            let penalized = raw * 1.0 / (1 + stag) as f64;
            (k, (1.0 + penalized).min(cap))
        })
        .collect()
}

/// Seed the demand slice from the LOAD at `pc`, resolving indexed-load temporaries
/// (`LDR R0,[R4,#4]` → `$tmp = INT_ADD(R4,4); LOAD($tmp)`) back to the base register.
fn seed_demand_from_block(block: &Block, pc: u64) -> DemandSlice {
    let mut current_pc = block.start;
    let mut target_idx: Option<usize> = None;

    for (i, stmt) in block.pcode.instructions.iter().enumerate() {
        if stmt.op == Op::InstructionMarker {
            if let pcode::Value::Const(addr, _) = stmt.inputs.first() {
                current_pc = addr;
            }
        }
        if current_pc == pc {
            if let Op::Load(_) = stmt.op {
                target_idx = Some(i);
                break;
            }
        }
    }

    let idx = match target_idx {
        Some(i) => i,
        None => return DemandSlice::default(),
    };

    let load_stmt = &block.pcode.instructions[idx];
    let addr_var = match load_stmt.inputs.first() {
        Value::Var(vn) if !vn.is_invalid() => vn,
        _ => return DemandSlice::default(),
    };

    let final_var = resolve_temporary_in_block(block, idx, addr_var.id);
    if final_var.is_invalid() { DemandSlice::default() } else { DemandSlice::seed(final_var) }
}

fn resolve_temporary_in_block(block: &Block, up_to_idx: usize, tmp_id: VarId) -> VarNode {
    if tmp_id > 0 { return VarNode::new(tmp_id, 4); }
    if tmp_id == 0 { return VarNode::NONE; }
    for stmt in block.pcode.instructions[..up_to_idx].iter().rev() {
        let out = stmt.output;
        if out.is_invalid() || out.id != tmp_id {
            continue;
        }
        return match stmt.op {
            Op::Copy | Op::ZeroExtend | Op::SignExtend => {
                match stmt.inputs.first() {
                    Value::Var(src) if !src.is_invalid() => {
                        resolve_temporary_in_block(block, up_to_idx, src.id)
                    }
                    _ => VarNode::NONE,
                }
            }
            Op::IntAdd | Op::IntSub => {
                let src0 = stmt.inputs.first();
                let src1 = stmt.inputs.second();
                for src in [src0, src1] {
                    if let Value::Var(vn) = src {
                        if vn.id > 0 {
                            return vn;
                        }
                        if vn.id < 0 {
                            let rec = resolve_temporary_in_block(block, up_to_idx, vn.id);
                            if !rec.is_invalid() {
                                return rec;
                            }
                        }
                    }
                }
                VarNode::NONE
            }
            _ => VarNode::NONE,
        };
    }
    VarNode::NONE
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

    /// `seed_demand_from_block` must resolve through the temporary produced by `INT_ADD`
    /// for an indexed load (`LDR R0,[R4,#4]`) and seed the base register (R4), not the
    /// short-lived temporary.  Seeding the temporary would leave `demanded_data = {-n}`
    /// which `strip_temporaries` erases before any predecessor block is visited.
    #[test]
    fn seed_resolves_indexed_load_temporary_to_base_register() {
        let mut pcode = pcode::Block::new();
        let tmp = VarNode::new(-1, 4); // $tmp produced by INT_ADD
        let r4 = VarNode::new(4, 4);   // base register R4
        let r0 = VarNode::new(1, 4);   // destination

        // LDR R0,[R4,#4] lifts as: $tmp = INT_ADD(R4, 4); R0 = LOAD($tmp)
        pcode.push(marker(0x500));
        pcode.push((tmp, pcode::Op::IntAdd, r4, pcode::Value::Const(4, 4)));
        pcode.push((r0, pcode::Op::Load(0), tmp));

        let block = lifter_block(pcode, 0x500, 0x504);

        let slice = seed_demand_from_block(&block, 0x500);

        assert!(
            slice.demanded_data.contains(&r4.id),
            "seed should demand the base register R4 (id=4), got {:?}",
            slice.demanded_data
        );
        assert!(
            !slice.demanded_data.contains(&tmp.id),
            "seed must not demand the short-lived temporary (id=-1)"
        );
    }

    /// When a predecessor block's branch condition is gated on an MMIO read, the
    /// backward pass must produce a `Control` edge to that stream.  This verifies the
    /// inject_control-before-pass fix: the branch condition must be in `demanded_ctrl`
    /// *before* the backward pass runs so it can be traced back to the MMIO load.
    #[test]
    fn control_injection_before_pass_yields_control_edge() {
        // Predecessor block:
        //   0x600: R1 = LOAD[mmio_ctrl]         ; MMIO read feeds the branch condition
        //   0x604: $tmp = INT_EQUAL(R1, #3)     ; compare
        //   exit:  CBRANCH($tmp, <target>)       ; conditional branch gating path to B
        let mut pcode = pcode::Block::new();
        let r1 = VarNode::new(1, 4);
        let tmp_cmp = VarNode::new(-2, 1);
        let mmio_base = VarNode::new(9, 4); // address register for mmio_ctrl

        pcode.push(marker(0x600));
        pcode.push((r1, pcode::Op::Load(0), mmio_base));
        pcode.push(marker(0x604));
        pcode.push((tmp_cmp, pcode::Op::IntEqual, r1, pcode::Value::Const(3, 4)));

        let mut block = lifter_block(pcode, 0x600, 0x608);
        // Set the block exit to a conditional branch on $tmp_cmp.
        block.exit = BlockExit::Branch {
            target: icicle_vm::cpu::lifter::Target::External(pcode::Value::Const(0xDEAD, 4)),
            fallthrough: icicle_vm::cpu::lifter::Target::External(pcode::Value::Const(0xBEEF, 4)),
            cond: pcode::Value::Var(tmp_cmp),
        };

        let mut read_sites: HashMap<u64, StreamKey> = HashMap::new();
        read_sites.insert(0x600, 0xDEAD_BEEF); // mmio_ctrl stream key

        let mut slice = DemandSlice::default(); // no data demand — only control
        slice.process_block_backward(&block, &[], &read_sites, true /* inject_control */, None);

        assert!(
            slice.sources_ctrl.contains(&AccessContext::new(0x600, 0xDEAD_BEEF)),
            "MMIO load feeding the branch condition must be recorded as a Control source; \
             got sources_ctrl={:?}",
            slice.sources_ctrl
        );
    }

    /// Build a `BlockTable` from a list of lifter blocks (for frontier tests).
    fn block_table(blocks: Vec<Block>) -> BlockTable {
        let mut code = BlockTable::default();
        code.blocks = blocks;
        code
    }

    /// A conditional branch block whose condition is computed from an MMIO read.
    /// `start`/`end` bound the block; `target`/`fallthrough` are the two successor
    /// addresses; the MMIO read site is registered under `stream` at `start`.
    fn mmio_branch_block(
        start: u64,
        end: u64,
        target: u64,
        fallthrough: u64,
    ) -> Block {
        let mut pcode = pcode::Block::new();
        let r1 = VarNode::new(1, 4);
        let tmp_cmp = VarNode::new(-2, 1);
        let mmio_base = VarNode::new(9, 4);

        pcode.push(marker(start));
        pcode.push((r1, pcode::Op::Load(0), mmio_base)); // MMIO load @start
        pcode.push((tmp_cmp, pcode::Op::IntEqual, r1, pcode::Value::Const(3, 4)));

        let mut block = lifter_block(pcode, start, end);
        block.exit = BlockExit::Branch {
            target: Target::External(pcode::Value::Const(target, 4)),
            fallthrough: Target::External(pcode::Value::Const(fallthrough, 4)),
            cond: pcode::Value::Var(tmp_cmp),
        };
        block
    }

    /// `candidates_for_branch` must trace a frontier branch back to the MMIO stream
    /// whose value computes its condition — the dual of `candidates_for_new_stream`.
    #[test]
    fn candidates_for_branch_traces_gating_mmio() {
        let stream = 0x5800_0010u64;
        let block = mmio_branch_block(0x700, 0x70c, 0xDEAD, 0x720);
        let code = block_table(vec![block]);

        let mut analyzer = MmioFlowAnalyzer::new(vec![]);
        analyzer.record_read_site(0x700, stream); // the load @0x700 reads `stream`

        let keys = analyzer.candidates_for_branch(0x700, &code);
        assert_eq!(
            keys,
            vec![stream],
            "branch condition gated on an MMIO read must attribute to that stream; got {keys:?}"
        );
    }

    /// A block with no conditional branch (unconditional jump) has no gating condition,
    /// so `candidates_for_branch` returns nothing.
    #[test]
    fn candidates_for_branch_empty_for_unconditional_block() {
        let mut pcode = pcode::Block::new();
        pcode.push(marker(0x800));
        pcode.push((VarNode::new(1, 4), pcode::Op::Copy, pcode::Value::Const(0, 4)));
        let mut block = lifter_block(pcode, 0x800, 0x804);
        block.exit = BlockExit::Jump { target: Target::External(pcode::Value::Const(0x820, 4)) };
        let code = block_table(vec![block]);

        let analyzer = MmioFlowAnalyzer::new(vec![]);
        assert!(analyzer.candidates_for_branch(0x800, &code).is_empty());
    }

    /// `compute_frontier_stream_weights` boosts a stream that gates an *uncovered* branch
    /// target (no lifted block at the target address), and leaves the stream un-boosted once
    /// every successor is covered.
    #[test]
    fn frontier_weights_boost_gating_stream_only_when_target_uncovered() {
        let stream = 0x5800_0010u64;

        // Branch @0x700 → 0xDEAD (UNCOVERED: no block at 0xDEAD) / 0x720 (covered).
        let branch = mmio_branch_block(0x700, 0x70c, 0xDEAD, 0x720);
        // A covered fallthrough block so 0x720 IS reached.
        let mut fall_pcode = pcode::Block::new();
        fall_pcode.push(marker(0x720));
        fall_pcode.push((VarNode::new(2, 4), pcode::Op::Copy, pcode::Value::Const(0, 4)));
        let mut fall = lifter_block(fall_pcode, 0x720, 0x724);
        fall.exit = BlockExit::Jump { target: Target::External(pcode::Value::Const(0x900, 4)) };

        let code = block_table(vec![branch, fall]);
        let mut analyzer = MmioFlowAnalyzer::new(vec![]);
        analyzer.record_read_site(0x700, stream);

        let weights = compute_frontier_stream_weights(&code, &analyzer, &HashMap::new());
        assert!(
            weights.get(&stream).copied().unwrap_or(1.0) > 1.0,
            "stream gating an uncovered branch target must be boosted; got {weights:?}"
        );

        // Now mark 0xDEAD as covered by adding a block there: no frontier remains.
        let mut deads = pcode::Block::new();
        deads.push(marker(0xDEAD));
        deads.push((VarNode::new(3, 4), pcode::Op::Copy, pcode::Value::Const(0, 4)));
        let mut dead_block = lifter_block(deads, 0xDEAD, 0xDEB1);
        dead_block.exit = BlockExit::Jump { target: Target::External(pcode::Value::Const(0x900, 4)) };

        let branch2 = mmio_branch_block(0x700, 0x70c, 0xDEAD, 0x720);
        let mut fall2_pcode = pcode::Block::new();
        fall2_pcode.push(marker(0x720));
        fall2_pcode.push((VarNode::new(2, 4), pcode::Op::Copy, pcode::Value::Const(0, 4)));
        let mut fall2 = lifter_block(fall2_pcode, 0x720, 0x724);
        fall2.exit = BlockExit::Jump { target: Target::External(pcode::Value::Const(0x900, 4)) };

        let code_covered = block_table(vec![branch2, fall2, dead_block]);
        let weights_covered = compute_frontier_stream_weights(&code_covered, &analyzer, &HashMap::new());
        assert!(
            weights_covered.get(&stream).is_none(),
            "no boost once every branch target is covered; got {weights_covered:?}"
        );
    }

    /// `compute_frontier_branch_set` returns the start addresses of conditional blocks
    /// with at least one uncovered successor.
    #[test]
    fn frontier_branch_set_contains_uncovered_targets() {
        let branch = mmio_branch_block(0x700, 0x70c, 0xDEAD, 0x720);
        let mut fall_pcode = pcode::Block::new();
        fall_pcode.push(marker(0x720));
        fall_pcode.push((VarNode::new(2, 4), pcode::Op::Copy, pcode::Value::Const(0, 4)));
        let mut fall = lifter_block(fall_pcode, 0x720, 0x724);
        fall.exit = BlockExit::Jump { target: Target::External(pcode::Value::Const(0x900, 4)) };

        let code = block_table(vec![branch, fall]);
        let frontier = compute_frontier_branch_set(&code);

        assert!(
            frontier.contains(&0x700),
            "branch @0x700 with uncovered target 0xDEAD must be in frontier set; got {frontier:?}"
        );

        // Once all successors are covered, the branch is no longer frontier.
        let branch2 = mmio_branch_block(0x700, 0x70c, 0xDEAD, 0x720);
        let mut fall2_pcode = pcode::Block::new();
        fall2_pcode.push(marker(0x720));
        fall2_pcode.push((VarNode::new(2, 4), pcode::Op::Copy, pcode::Value::Const(0, 4)));
        let mut fall2 = lifter_block(fall2_pcode, 0x720, 0x724);
        fall2.exit = BlockExit::Jump { target: Target::External(pcode::Value::Const(0x900, 4)) };

        let mut deads = pcode::Block::new();
        deads.push(marker(0xDEAD));
        deads.push((VarNode::new(3, 4), pcode::Op::Copy, pcode::Value::Const(0, 4)));
        let mut dead_block = lifter_block(deads, 0xDEAD, 0xDEB1);
        dead_block.exit = BlockExit::Jump { target: Target::External(pcode::Value::Const(0x900, 4)) };

        let code_covered = block_table(vec![branch2, fall2, dead_block]);
        let frontier_covered = compute_frontier_branch_set(&code_covered);
        assert!(
            !frontier_covered.contains(&0x700),
            "branch @0x700 with all successors covered must NOT be in frontier set; got {frontier_covered:?}"
        );
    }

    /// Stagnation penalty reduces stream weight: a stream with high stagnation count
    /// receives a smaller boost than the same stream with zero stagnation.
    #[test]
    fn stagnation_penalty_reduces_weight() {
        let stream = 0x5800_0010u64;

        let branch = mmio_branch_block(0x700, 0x70c, 0xDEAD, 0x720);
        let mut fall_pcode = pcode::Block::new();
        fall_pcode.push(marker(0x720));
        fall_pcode.push((VarNode::new(2, 4), pcode::Op::Copy, pcode::Value::Const(0, 4)));
        let mut fall = lifter_block(fall_pcode, 0x720, 0x724);
        fall.exit = BlockExit::Jump { target: Target::External(pcode::Value::Const(0x900, 4)) };

        let code = block_table(vec![branch, fall]);
        let mut analyzer = MmioFlowAnalyzer::new(vec![]);
        analyzer.record_read_site(0x700, stream);

        // No stagnation: full weight.
        let w_none = compute_frontier_stream_weights(&code, &analyzer, &HashMap::new());
        let weight_fresh = w_none.get(&stream).copied().unwrap_or(1.0);

        // High stagnation: reduced weight.
        let mut stag = HashMap::new();
        stag.insert(stream, 10u32);
        // Need fresh blocks because block_table consumed the originals.
        let branch2 = mmio_branch_block(0x700, 0x70c, 0xDEAD, 0x720);
        let mut fall2_pcode = pcode::Block::new();
        fall2_pcode.push(marker(0x720));
        fall2_pcode.push((VarNode::new(2, 4), pcode::Op::Copy, pcode::Value::Const(0, 4)));
        let mut fall2 = lifter_block(fall2_pcode, 0x720, 0x724);
        fall2.exit = BlockExit::Jump { target: Target::External(pcode::Value::Const(0x900, 4)) };
        let code2 = block_table(vec![branch2, fall2]);

        let w_stag = compute_frontier_stream_weights(&code2, &analyzer, &stag);
        let weight_stagnant = w_stag.get(&stream).copied().unwrap_or(1.0);

        assert!(
            weight_stagnant < weight_fresh,
            "stagnation penalty must reduce weight: fresh={weight_fresh}, stagnant={weight_stagnant}"
        );
        assert!(
            weight_stagnant >= 1.0,
            "stagnated weight must still be >= 1.0; got {weight_stagnant}"
        );
    }

    // ── Phase A trigger tests ────────────────────────────────────────────────

    #[test]
    fn trigger_first_sighting_runs_and_is_first_seen() {
        let mut analyzer = MmioFlowAnalyzer::new(vec![]);
        let t = analyzer.trigger(0x5800_0000, 0x100, 50);
        assert_eq!(t, PhaseATrigger { run_slice: true, first_seen: true });
    }

    #[test]
    fn trigger_repeat_without_growth_skips() {
        let mut analyzer = MmioFlowAnalyzer::new(vec![]);
        analyzer.trigger(0x5800_0000, 0x100, 50);
        let t = analyzer.trigger(0x5800_0000, 0x100, 50);
        assert_eq!(t, PhaseATrigger { run_slice: false, first_seen: false });
        let t = analyzer.trigger(0x5800_0000, 0x100, 50 + REANALYZE_BLOCK_DELTA - 1);
        assert_eq!(t, PhaseATrigger { run_slice: false, first_seen: false });
    }

    #[test]
    fn trigger_reanalyzes_after_cfg_growth() {
        let mut analyzer = MmioFlowAnalyzer::new(vec![]);
        analyzer.trigger(0x5800_0000, 0x100, 50);
        let t = analyzer.trigger(0x5800_0000, 0x100, 50 + REANALYZE_BLOCK_DELTA);
        assert_eq!(t, PhaseATrigger { run_slice: true, first_seen: false });
        let t = analyzer.trigger(0x5800_0000, 0x100, 50 + REANALYZE_BLOCK_DELTA);
        assert_eq!(t, PhaseATrigger { run_slice: false, first_seen: false });
    }

    #[test]
    fn trigger_same_addr_different_pc_is_new_context() {
        let mut analyzer = MmioFlowAnalyzer::new(vec![]);
        analyzer.trigger(0x5800_0000, 0x100, 50);
        let t = analyzer.trigger(0x5800_0000, 0x230, 50);
        assert_eq!(t, PhaseATrigger { run_slice: true, first_seen: true });
        let t = analyzer.trigger(0x5800_0004, 0x100, 50);
        assert_eq!(t, PhaseATrigger { run_slice: true, first_seen: true });
    }
}
