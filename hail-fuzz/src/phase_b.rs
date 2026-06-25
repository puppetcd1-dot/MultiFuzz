//! Phase B — dynamic taint pass for typed stream-relationship classification.
//! Interprets cached P-code blocks against a `ShadowState` driven by live `Cpu` values.

use std::cell::{Cell, RefCell};
use std::ops::Range;
use std::rc::Rc;

use hashbrown::{HashMap, HashSet};
use icicle_vm::{
    cpu::lifter::{Block, BlockExit, Target},
    cpu::{BlockGroup, Cpu},
    BlockTable, Vm,
};
use pcode::{Op, Value, VarId, VarNode};

use crate::{
    input::StreamKey,
    stream_relation::{AccessContext, EdgeKind, Relation, StreamRelationGraph},
    taint::{ShadowState, TaintTag},
};

/// Taint summary for a known library function.
///
/// When a `BlockExit::Call` targets a registered address, the engine applies the
/// corresponding summary instead of conservatively killing r0–r3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuncSummary {
    Memcpy,
    Memmove,
    Memset,
    Strlen,
    Passthrough,
}

pub trait ConcreteEnv {
    fn read_value(&mut self, v: Value) -> Option<u64>;
    fn read_mem(&mut self, addr: u64, size: u8) -> Option<u64>;
}

/// Production `ConcreteEnv` backed by the live emulator CPU.
pub struct LiveEnv<'a> {
    pub cpu: &'a mut Cpu,
    pub mmio_ranges: &'a [Range<u64>],
}

impl<'a> ConcreteEnv for LiveEnv<'a> {
    fn read_value(&mut self, v: Value) -> Option<u64> {
        Some(icicle_vm::cpu::read_value_zxt(self.cpu, v))
    }

    fn read_mem(&mut self, addr: u64, size: u8) -> Option<u64> {
        use icicle_vm::cpu::mem::perm;
        if addr > u64::from(u32::MAX) { return None; }
        let len = size.max(1) as u64;
        let end = addr.checked_add(len - 1)?;
        if self.mmio_ranges.iter().any(|r| r.contains(&addr) || r.contains(&end))
            || self.cpu.mem.get_physical_addr(addr).is_none()
            || self.cpu.mem.get_physical_addr(end).is_none()
        {
            return None;
        }
        match size {
            1 => self.cpu.mem.read_u8(addr, perm::NONE).ok().map(|v| v as u64),
            2 => self.cpu.mem.read_u16(addr, perm::NONE).ok().map(|v| v as u64),
            4 => self.cpu.mem.read_u32(addr, perm::NONE).ok().map(|v| v as u64),
            8 => self.cpu.mem.read_u64(addr, perm::NONE).ok(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlEdgeObs {
    pub source: AccessContext,
    pub target: AccessContext,
    pub is_length: bool,
    /// True when the gating condition was an IntEqual/IntNotEqual (not a range check).
    /// Only equality-gated values make useful discriminants.
    pub is_eq: bool,
    pub sample: Option<(u64, u64)>,
}

#[derive(Debug, Default, Clone)]
pub struct PassResult {
    pub address_edges: Vec<(AccessContext, AccessContext)>,
    pub control_edges: Vec<ControlEdgeObs>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Length-relation sample store (accumulates across passes)
// ──────────────────────────────────────────────────────────────────────────────

/// Accumulates `(val_A, count_B)` samples per `Length` edge, keyed by address pair
/// so multiple read-site PCs for the same streams share one pool for better fitting.
#[derive(Default)]
pub struct LengthSampleStore {
    samples: HashMap<(StreamKey, StreamKey), Vec<(u64, u64)>>,
}

impl LengthSampleStore {
    pub fn new() -> Self { Self::default() }

    pub fn record(&mut self, edge: (AccessContext, AccessContext), sample: (u64, u64)) -> Option<Relation> {
        let key = (edge.0.addr, edge.1.addr);
        let v = self.samples.entry(key).or_default();
        if !v.contains(&sample) { v.push(sample); }
        Relation::fit(v)
    }

    pub fn fitted_relations(&self) -> impl Iterator<Item = ((StreamKey, StreamKey), Relation)> + '_ {
        self.samples.iter().filter_map(|(key, v)| Relation::fit(v).map(|r| (*key, r)))
    }
}

/// Upper bound on sub-block steps interpreted per group, guarding against an
/// intra-group back-edge looping forever.  Groups are small (a handful of
/// sub-blocks per guest instruction), so this is never reached in practice.
const MAX_GROUP_STEPS: usize = 256;

#[derive(Debug, Clone, Default)]
struct CtrlObs {
    fires: u64,
    is_loop: bool,
    is_eq: bool,
}

/// Maximum distance (in interpreted blocks) between a gating branch evaluation and
/// a target MMIO read for the gating entry to apply.  Boot-time polling loops
/// (PLL-ready, oscillator-ready) exit thousands of blocks before GPIO/USART code
/// runs; a genuine in-loop read is typically <30 blocks from its controlling branch.
const GATING_WINDOW: u64 = {
    match option_env!("GATING_WINDOW") {
        Some(_) => 128, // placeholder; runtime override below
        None => 128,
    }
};

/// Branch-visit threshold above which a gating branch is considered a spinloop
/// (busy-wait).  Spinloop discriminants (the values that let the loop exit) are
/// not useful selectors — they gate everything downstream equally.
const SPINLOOP_VISIT_THRESHOLD: u64 = 8;

/// Dynamic taint interpreter operating one block at a time.
pub struct PhaseBEngine {
    shadow: ShadowState,
    read_sites: HashMap<u64, StreamKey>,
    mmio_ranges: Vec<Range<u64>>,
    def_concrete: HashMap<VarId, Option<u64>>,
    def_taint: HashMap<VarId, TaintTag>,
    /// VarIds defined by IntEqual/IntNotEqual in current block (gates discriminant capture).
    def_eq_derived: HashSet<VarId>,
    cur_pc: u64,
    /// Branch PCs where the tainted condition was equality-derived (persists across blocks).
    branch_is_eq: HashSet<u64>,
    target_reads: HashMap<AccessContext, u64>,
    source_last_val: HashMap<AccessContext, u64>,
    /// Source loads whose value LiveEnv deferred; captured at next block entry.
    /// Stores the full output `VarNode` (not just its `VarId`) so the deferred
    /// read uses the register's true size.
    pending_src_vals: Vec<(AccessContext, VarNode)>,
    branch_visits: HashMap<u64, u64>,
    gating: Vec<(AccessContext, u64)>,
    gating_set: HashSet<(AccessContext, u64)>,
    /// Last step at which each gating entry's branch was evaluated.  Updated on
    /// every branch re-evaluation (not just the first insertion) so loop iterations
    /// keep the entry "warm".  Target reads outside the recency window are not charged.
    gating_last_step: HashMap<(AccessContext, u64), u64>,
    /// Monotonic block counter incremented at each `begin_block`.
    step: u64,
    /// Runtime-configurable gating window (from `GATING_WINDOW` env var).
    gating_window: u64,
    addr_edges: HashSet<(AccessContext, AccessContext)>,
    ctrl_obs: HashMap<(AccessContext, AccessContext), CtrlObs>,
    /// Per-address function summaries.  Not cleared between passes — populated once
    /// at startup from user configuration (e.g. `PHASE_B_MEMCPY_ADDR` env var).
    pub(crate) summaries: HashMap<u64, FuncSummary>,
}

impl PhaseBEngine {
    pub fn new(read_sites: HashMap<u64, StreamKey>, mmio_ranges: Vec<Range<u64>>) -> Self {
        let gating_window: u64 = std::env::var("GATING_WINDOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(GATING_WINDOW);
        Self {
            shadow: ShadowState::new(),
            read_sites,
            mmio_ranges,
            def_concrete: HashMap::new(),
            def_taint: HashMap::new(),
            def_eq_derived: HashSet::new(),
            cur_pc: 0,
            branch_is_eq: HashSet::new(),
            target_reads: HashMap::new(),
            source_last_val: HashMap::new(),
            pending_src_vals: Vec::new(),
            branch_visits: HashMap::new(),
            gating: Vec::new(),
            gating_set: HashSet::new(),
            gating_last_step: HashMap::new(),
            step: 0,
            gating_window,
            addr_edges: HashSet::new(),
            ctrl_obs: HashMap::new(),
            summaries: HashMap::new(),
        }
    }

    /// Reset all per-pass state and refresh the runtime config for a new pass.
    pub fn reset_pass(&mut self, read_sites: HashMap<u64, StreamKey>, mmio_ranges: Vec<Range<u64>>) {
        self.shadow = ShadowState::new();
        self.read_sites = read_sites;
        self.mmio_ranges = mmio_ranges;
        self.def_concrete.clear();
        self.def_taint.clear();
        self.cur_pc = 0;
        self.target_reads.clear();
        self.source_last_val.clear();
        self.pending_src_vals.clear();
        self.branch_visits.clear();
        self.branch_is_eq.clear();
        self.gating.clear();
        self.gating_set.clear();
        self.gating_last_step.clear();
        self.step = 0;
        self.addr_edges.clear();
        self.ctrl_obs.clear();
    }

    // ── value/taint accessors ─────────────────────────────────────────────────

    /// Concrete value of an input, consulting in-block defs first, then `env`.
    fn concrete_in(&self, v: Value, env: &mut dyn ConcreteEnv) -> Option<u64> {
        match v {
            Value::Const(c, s) => Some(mask_to_size(c, s)),
            Value::Var(vn) => {
                if vn.is_invalid() {
                    return None;
                }
                if let Some(c) = self.def_concrete.get(&vn.id) {
                    return *c;
                }
                if vn.id > 0 {
                    // Mask so JIT host pointers in Regs temp slots can't escape as addresses.
                    env.read_value(Value::Var(vn)).map(|val| mask_to_size(val, vn.size))
                } else {
                    None
                }
            }
        }
    }

    fn taint_in(&self, v: Value) -> TaintTag {
        match v {
            Value::Const(..) => TaintTag::Clean,
            Value::Var(vn) => {
                if vn.is_invalid() {
                    return TaintTag::Clean;
                }
                if let Some(t) = self.def_taint.get(&vn.id) {
                    return t.clone();
                }
                if vn.id > 0 {
                    self.shadow.reg_tag(vn.id)
                } else {
                    TaintTag::Clean
                }
            }
        }
    }

    fn set_def(&mut self, out: pcode::VarNode, concrete: Option<u64>, taint: TaintTag) {
        if out.is_invalid() {
            return;
        }
        self.def_concrete.insert(out.id, concrete);
        self.def_taint.insert(out.id, taint.clone());
        if out.id > 0 {
            self.shadow.set_reg_tag(out.id, taint);
        }
    }

    fn contexts_of(&self, tag: &TaintTag) -> Vec<AccessContext> {
        self.shadow.index.contexts_in_tag(tag)
    }

    #[cfg(test)]
    fn source_value(&self, ctx: AccessContext) -> Option<u64> {
        self.source_last_val.get(&ctx).copied()
    }

    // ── call summary helpers ──────────────────────────────────────────────────

    fn concrete_arg_reg(&self, id: i16, size: u8, env: &mut dyn ConcreteEnv) -> Option<u64> {
        if let Some(c) = self.def_concrete.get(&id) {
            return *c;
        }
        env.read_value(Value::Var(VarNode::new(id, size))).map(|v| mask_to_size(v, size))
    }

    fn taint_arg_reg(&self, id: i16) -> TaintTag {
        if let Some(t) = self.def_taint.get(&id) {
            return t.clone();
        }
        self.shadow.reg_tag(id)
    }

    fn apply_memcpy_summary(&mut self, env: &mut dyn ConcreteEnv) {
        let dst = self.concrete_arg_reg(0, 4, env);
        let src = self.concrete_arg_reg(1, 4, env);
        let n   = self.concrete_arg_reg(2, 4, env).unwrap_or(0) as usize;
        if let (Some(dst), Some(src)) = (dst, src) {
            for i in 0..n.min(4096) {
                let tag = self.shadow.mem_tag(src as u32 + i as u32);
                if tag.is_clean() {
                    self.shadow.mem.remove(&(dst as u32 + i as u32));
                } else {
                    self.shadow.mem.insert(dst as u32 + i as u32, tag);
                }
            }
        }
        self.shadow.kill_call_regs();
    }

    fn apply_memset_summary(&mut self, env: &mut dyn ConcreteEnv) {
        let dst   = self.concrete_arg_reg(0, 4, env);
        let n     = self.concrete_arg_reg(2, 4, env).unwrap_or(0) as usize;
        let c_tag = self.taint_arg_reg(1);
        if let Some(dst) = dst {
            self.shadow.set_mem_tag_range(dst as u32, n.min(4096), c_tag);
        }
        self.shadow.kill_call_regs();
    }

    fn apply_strlen_summary(&mut self, env: &mut dyn ConcreteEnv) {
        let s_ptr = self.concrete_arg_reg(0, 4, env);
        let s_tag = self.taint_arg_reg(0);
        let result_tag = if let Some(s) = s_ptr {
            let mut tag = s_tag;
            for i in 0..4096u32 {
                let addr = (s as u32).wrapping_add(i);
                tag = tag.union(&self.shadow.mem_tag(addr));
                if env.read_mem(addr as u64, 1).map_or(false, |v| v == 0) {
                    break;
                }
            }
            tag
        } else {
            TaintTag::Clean
        };
        self.shadow.kill_call_regs();
        self.shadow.set_reg_tag(0, result_tag);
    }

    fn apply_passthrough_summary(&mut self, env: &mut dyn ConcreteEnv) {
        let in_tag = self.taint_arg_reg(0);
        let _ = self.concrete_arg_reg(0, 4, env);
        self.shadow.kill_call_regs();
        self.shadow.set_reg_tag(0, in_tag);
    }

    fn handle_call_exit(&mut self, target: Value, env: &mut dyn ConcreteEnv) {
        let addr = match target {
            Value::Const(a, _) => a,
            Value::Var(vn) => match self.concrete_arg_reg(vn.id, vn.size, env) {
                Some(a) => a,
                None => {
                    self.shadow.kill_call_regs();
                    return;
                }
            },
        };
        match self.summaries.get(&addr).copied() {
            Some(FuncSummary::Memcpy | FuncSummary::Memmove) => self.apply_memcpy_summary(env),
            Some(FuncSummary::Memset)                        => self.apply_memset_summary(env),
            Some(FuncSummary::Strlen)                        => self.apply_strlen_summary(env),
            Some(FuncSummary::Passthrough)                   => self.apply_passthrough_summary(env),
            None                                             => self.shadow.kill_call_regs(),
        }
    }

    // ── block interpretation ───────────────────────────────────────────────────

    /// Interpret a single basic block, updating shadow state and observations.
    ///
    /// Resets per-block definition state, then interprets the block standalone.
    /// For multi-block groups use [`Self::run_group`], which preserves definition
    /// state across the group's internal sub-blocks.  The production driver always
    /// goes through `run_group`; this single-block entry point is retained for
    /// unit tests and standalone interpretation.
    #[allow(dead_code)]
    pub fn run_block(&mut self, block: &Block, env: &mut dyn ConcreteEnv) {
        self.begin_block(block.start, env);
        self.interpret_block(block, env);
    }

    /// Interpret a whole `BlockGroup` by following its internal control flow.
    ///
    /// The JIT only fires the entry-block hook (injecting `Op::Hook` into a
    /// non-entry sub-block corrupts the JIT), so the engine must itself walk the
    /// group's remaining sub-blocks to observe taint that flows through P-code
    /// generated past an internal branch (ARM IT-blocks, divide zero-checks,
    /// multi-register loads, etc.).
    ///
    /// `base` is the global block index of `blocks[0]` (i.e. `group.blocks.0`),
    /// used to translate `Target::Internal(global_idx)` into a position in
    /// `blocks`.  Definition state (`def_concrete`/`def_taint`) is cleared once at
    /// group entry and then carried across sub-blocks, so a value a sub-block
    /// computes is visible to the sub-blocks that consume it (the live CPU still
    /// holds only the group-entry register state at this point).
    ///
    /// Only the sub-block the real CPU would take is interpreted: each internal
    /// branch is resolved by evaluating its condition concretely.  If a condition
    /// cannot be resolved concretely, or control leaves the group (Call/Return/an
    /// external target), interpretation stops — an accepted under-taint, never a
    /// guess down an untaken path (which would manufacture spurious edges).
    pub fn run_group(&mut self, blocks: &[Block], base: usize, env: &mut dyn ConcreteEnv) {
        let Some(first) = blocks.first() else { return };
        self.begin_block(first.start, env);

        let mut pos = 0usize;
        for _ in 0..MAX_GROUP_STEPS {
            let block = &blocks[pos];
            self.interpret_block(block, env);

            let next = match &block.exit {
                BlockExit::Jump { target } => {
                    Self::internal_pos(target, base, blocks.len())
                }
                BlockExit::Branch { cond, target, fallthrough } => {
                    match self.concrete_in(*cond, env) {
                        Some(0) => Self::internal_pos(fallthrough, base, blocks.len()),
                        Some(_) => Self::internal_pos(target, base, blocks.len()),
                        // Non-concrete condition: we can't know the CPU's path.
                        None => None,
                    }
                }
                BlockExit::Return { .. } => None,
                BlockExit::Call { target, .. } => {
                    self.handle_call_exit(*target, env);
                    None
                }
            };

            match next {
                Some(p) => pos = p,
                None => break,
            }
        }
    }

    /// Translate a `Target::Internal(global_idx)` into a position within the
    /// captured group slice, or `None` if it leaves the group / isn't internal.
    fn internal_pos(target: &Target, base: usize, len: usize) -> Option<usize> {
        match target {
            Target::Internal(global) => {
                let p = global.checked_sub(base)?;
                (p < len).then_some(p)
            }
            _ => None,
        }
    }

    /// Reset per-block definition state and capture any source values deferred
    /// from the previous group/block (their loads have now executed on the CPU).
    fn begin_block(&mut self, start: u64, env: &mut dyn ConcreteEnv) {
        self.def_concrete.clear();
        self.def_taint.clear();
        self.def_eq_derived.clear();
        self.cur_pc = start;
        self.step += 1;

        // Capture source values deferred from previous block (load has now executed).
        if !self.pending_src_vals.is_empty() {
            for (ctx, var_node) in std::mem::take(&mut self.pending_src_vals) {
                if let Some(v) = env.read_value(Value::Var(var_node)) {
                    self.source_last_val.entry(ctx).or_insert(v);
                }
            }
        }
    }

    /// Interpret one block's P-code and its exit gating, WITHOUT resetting the
    /// per-block definition state (so it can be chained across a group's
    /// sub-blocks).  `cur_pc` is advanced only by `InstructionMarker`s, so a
    /// marker-less continuation sub-block keeps the instruction PC of its parent.
    fn interpret_block(&mut self, block: &Block, env: &mut dyn ConcreteEnv) {
        for stmt in &block.pcode.instructions {
            match stmt.op {
                Op::InstructionMarker => {
                    // Guard against synthetic blocks with a VarNode input; `.as_u64()`
                    // would panic through the extern "C" trampoline and abort.
                    if let Value::Const(pc, _) = stmt.inputs.first() {
                        self.cur_pc = pc;
                    }
                }

                Op::Load(_) => {
                    let size = stmt.output.size.max(1);
                    let addr_input = stmt.inputs.first();
                    let addr = self.concrete_in(addr_input, env);

                    if let Some(&key) = self.read_sites.get(&self.cur_pc) {
                        let ctx = AccessContext::new(self.cur_pc, key);
                        *self.target_reads.entry(ctx).or_insert(0) += 1;

                        let addr_tag = self.taint_in(addr_input);
                        for src in self.contexts_of(&addr_tag) {
                            if src.addr != ctx.addr { self.addr_edges.insert((src, ctx)); }
                        }

                        for i in 0..self.gating.len() {
                            let (src, bpc) = self.gating[i];
                            if src.addr == ctx.addr { continue; }
                            // Fix 1: only charge gating entries whose branch was
                            // evaluated recently (within the recency window).
                            let last = self.gating_last_step.get(&(src, bpc)).copied().unwrap_or(0);
                            if self.step.saturating_sub(last) > self.gating_window { continue; }
                            let obs = self.ctrl_obs.entry((src, ctx)).or_default();
                            obs.fires += 1;
                            if self.branch_visits.get(&bpc).copied().unwrap_or(0) >= 2 { obs.is_loop = true; }
                            if self.branch_is_eq.contains(&bpc) { obs.is_eq = true; }
                        }

                        // LiveEnv returns None for MMIO reads (avoids consuming fuzz bytes);
                        // defer value capture to next block entry via pending_src_vals.
                        let loaded = addr.and_then(|a| env.read_mem(a, size));
                        if let Some(v) = loaded {
                            self.source_last_val.insert(ctx, v);
                        } else if stmt.output.id > 0 {
                            self.pending_src_vals.push((ctx, stmt.output));
                        }

                        let tag = self.shadow.source_tag(ctx);
                        self.set_def(stmt.output, loaded, tag);
                    } else {
                        let tag = match addr {
                            Some(a) => self.shadow.mem_tag_range(a as u32, size as usize),
                            None => TaintTag::Clean,
                        };
                        let val = addr.and_then(|a| env.read_mem(a, size));
                        self.set_def(stmt.output, val, tag);
                    }
                }

                Op::Store(_) => {
                    let size = value_size(stmt.inputs.second()).max(1);
                    let addr = self.concrete_in(stmt.inputs.first(), env);
                    if let Some(a) = addr {
                        let val_tag = self.taint_in(stmt.inputs.second());
                        let addr_tag = self.taint_in(stmt.inputs.first());
                        self.shadow
                            .set_mem_tag_range(a as u32, size as usize, val_tag.union(&addr_tag));
                    }
                }

                Op::Copy | Op::ZeroExtend | Op::SignExtend => {
                    let src = stmt.inputs.first();
                    let c = self.concrete_in(src, env);
                    let t = self.taint_in(src);
                    self.set_def(stmt.output, c, t);
                }

                Op::Subpiece(offset) => {
                    let src = stmt.inputs.first();
                    let t = self.taint_in(src);
                    let c = self.concrete_in(src, env).map(|v| {
                        let shifted = v.checked_shr((offset as u32) * 8).unwrap_or(0);
                        mask_to_size(shifted, stmt.output.size)
                    });
                    self.set_def(stmt.output, c, t);
                }

                Op::IntAdd | Op::IntSub | Op::IntAnd | Op::IntOr | Op::IntXor | Op::IntMul
                | Op::IntLeft | Op::IntRight | Op::IntSignedRight => {
                    let a = stmt.inputs.first();
                    let b = stmt.inputs.second();
                    let ta = self.taint_in(a);
                    let tb = self.taint_in(b);
                    let c = match (self.concrete_in(a, env), self.concrete_in(b, env)) {
                        (Some(x), Some(y)) => {
                            Some(mask_to_size(eval_binop(stmt.op, x, y), stmt.output.size))
                        }
                        _ => None,
                    };
                    // Taint narrowing: skip union when both inputs are clean.
                    let t = if ta.is_clean() && tb.is_clean() { TaintTag::Clean } else { ta.union(&tb) };
                    self.set_def(stmt.output, c, t);
                }

                Op::IntEqual | Op::IntNotEqual => {
                    let a = stmt.inputs.first();
                    let b = stmt.inputs.second();
                    let ta = self.taint_in(a);
                    let tb = self.taint_in(b);
                    let size = value_size(a);
                    let c = match (self.concrete_in(a, env), self.concrete_in(b, env)) {
                        (Some(x), Some(y)) => eval_cmp(stmt.op, x, y, size),
                        _ => None,
                    };
                    // Taint narrowing: skip union when both inputs are clean.
                    let t = if ta.is_clean() && tb.is_clean() { TaintTag::Clean } else { ta.union(&tb) };
                    self.set_def(stmt.output, c, t);
                    if !stmt.output.is_invalid() {
                        self.def_eq_derived.insert(stmt.output.id);
                    }
                }

                Op::IntLess | Op::IntSignedLess | Op::IntLessEqual | Op::IntSignedLessEqual
                | Op::IntCarry | Op::IntSignedCarry | Op::IntSignedBorrow | Op::BoolAnd
                | Op::BoolOr | Op::BoolXor => {
                    let a = stmt.inputs.first();
                    let b = stmt.inputs.second();
                    let ta = self.taint_in(a);
                    let tb = self.taint_in(b);
                    let size = value_size(a);
                    // Concretely resolve comparison/boolean ops so internal branch
                    // conditions can be evaluated by `run_group`.
                    let c = match (self.concrete_in(a, env), self.concrete_in(b, env)) {
                        (Some(x), Some(y)) => eval_cmp(stmt.op, x, y, size),
                        _ => None,
                    };
                    // Taint narrowing: skip union when both inputs are clean.
                    let t = if ta.is_clean() && tb.is_clean() { TaintTag::Clean } else { ta.union(&tb) };
                    self.set_def(stmt.output, c, t);
                }

                Op::IntDiv | Op::IntSignedDiv | Op::IntRem | Op::IntSignedRem
                | Op::IntRotateLeft | Op::IntRotateRight => {
                    let ta = self.taint_in(stmt.inputs.first());
                    let tb = self.taint_in(stmt.inputs.second());
                    // Taint narrowing: skip union when both inputs are clean.
                    let t = if ta.is_clean() && tb.is_clean() { TaintTag::Clean } else { ta.union(&tb) };
                    self.set_def(stmt.output, None, t);
                }

                Op::IntNot | Op::IntNegate | Op::IntCountOnes | Op::IntCountLeadingZeroes => {
                    let t = self.taint_in(stmt.inputs.first());
                    self.set_def(stmt.output, None, t);
                }

                Op::BoolNot => {
                    let src = stmt.inputs.first();
                    let t = self.taint_in(src);
                    let c = self.concrete_in(src, env).map(|v| (v == 0) as u64);
                    self.set_def(stmt.output, c, t);
                    if !stmt.output.is_invalid() {
                        let input_is_eq = match stmt.inputs.first() {
                            Value::Var(vn) => self.def_eq_derived.contains(&vn.id),
                            _ => false,
                        };
                        if input_is_eq {
                            self.def_eq_derived.insert(stmt.output.id);
                        }
                    }
                }

                Op::PcodeOp(_) => { self.shadow.kill_call_regs(); }

                _ => {
                    if !stmt.output.is_invalid() {
                        self.set_def(stmt.output, None, TaintTag::Clean);
                    }
                }
            }
        }

        if let Some(Value::Var(cond)) = block.exit.cond() {
            if !cond.is_invalid() {
                let cond_tag = self.taint_in(Value::Var(cond));
                if !cond_tag.is_clean() {
                    let bpc = self.cur_pc;
                    let visits = self.branch_visits.entry(bpc).or_insert(0);
                    *visits += 1;
                    // Fix 3: suppress is_eq for spinloop branches — their exit
                    // values are not useful selectors.
                    if self.def_eq_derived.contains(&cond.id) {
                        if *visits < SPINLOOP_VISIT_THRESHOLD {
                            self.branch_is_eq.insert(bpc);
                        } else {
                            self.branch_is_eq.remove(&bpc);
                        }
                    }
                    for src in self.contexts_of(&cond_tag) {
                        let key = (src, bpc);
                        if self.gating_set.insert(key) {
                            self.gating.push(key);
                        }
                        // Fix 1: always update recency so loop re-evaluations
                        // keep the entry "warm" for targets inside the loop.
                        self.gating_last_step.insert(key, self.step);
                    }
                }
            }
        }
    }

    pub fn finish_pass(&self) -> PassResult {
        let address_edges: Vec<_> = self.addr_edges.iter().copied().collect();

        let mut control_edges = Vec::new();
        for (&(src, tgt), obs) in &self.ctrl_obs {
            let count = self.target_reads.get(&tgt).copied().unwrap_or(0);
            // Require ≥3 (not ≥2) to avoid flip-flopping on the threshold.
            let is_length = obs.is_loop && count >= 3 && obs.fires >= 2;
            let sample = self.source_last_val.get(&src).map(|&v| (v, count));
            control_edges.push(ControlEdgeObs {
                source: src,
                target: tgt,
                is_length,
                is_eq: obs.is_eq,
                sample,
            });
        }

        PassResult { address_edges, control_edges }
    }
}

/// Merge one pass result into the graph, fitting `Length` relations across passes.
pub fn apply_pass_result(
    graph: &mut StreamRelationGraph,
    store: &mut LengthSampleStore,
    result: PassResult,
) {
    for (src, tgt) in result.address_edges {
        graph.insert_or_confirm_edge(src, tgt, EdgeKind::Address, None);
    }
    for obs in result.control_edges {
        if obs.is_length {
            let relation = match obs.sample {
                Some(sample) => store.record((obs.source, obs.target), sample),
                None => None,
            };
            graph.insert_or_confirm_edge(obs.source, obs.target, EdgeKind::Length, relation);
        } else {
            graph.insert_or_confirm_edge(obs.source, obs.target, EdgeKind::Control, None);
            // Only equality-gated (cmd==3, not size<256) observations record discriminants.
            if obs.is_eq {
                if let Some((val, _count)) = obs.sample {
                    graph.record_discriminant(obs.source.addr, obs.target.addr, val);
                }
            }
        }
    }
    graph.backfill_length_relations(store.fitted_relations());
}

fn value_size(v: Value) -> u8 {
    match v {
        Value::Const(_, s) => s,
        Value::Var(vn) => vn.size,
    }
}

fn mask_to_size(v: u64, size: u8) -> u64 {
    if size >= 8 {
        v
    } else {
        let bits = size as u32 * 8;
        v & ((1u64 << bits) - 1)
    }
}

fn eval_binop(op: Op, a: u64, b: u64) -> u64 {
    match op {
        Op::IntAdd => a.wrapping_add(b),
        Op::IntSub => a.wrapping_sub(b),
        Op::IntAnd => a & b,
        Op::IntOr => a | b,
        Op::IntXor => a ^ b,
        Op::IntMul => a.wrapping_mul(b),
        Op::IntLeft => a.checked_shl(b as u32).unwrap_or(0),
        Op::IntRight => a.checked_shr(b as u32).unwrap_or(0),
        Op::IntSignedRight => ((a as i64).checked_shr(b as u32).unwrap_or(0)) as u64,
        _ => 0,
    }
}

/// Sign-extend the low `size` bytes of `v` to a full `i64`.
fn sign_extend(v: u64, size: u8) -> i64 {
    let bits = (size as u32) * 8;
    if bits == 0 || bits >= 64 {
        return v as i64;
    }
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
}

/// Concretely evaluate a comparison/boolean P-code op to `0` or `1`, given the
/// operand `size` (in bytes) needed for signed comparisons and carry/borrow.
/// Returns `None` for ops this evaluator does not model (div/rem/rotate), whose
/// concrete value is left unknown.  Inputs are assumed already masked to `size`.
fn eval_cmp(op: Op, a: u64, b: u64, size: u8) -> Option<u64> {
    let bits = (size as u32) * 8;
    let r = match op {
        Op::IntEqual => a == b,
        Op::IntNotEqual => a != b,
        Op::IntLess => a < b,
        Op::IntLessEqual => a <= b,
        Op::IntSignedLess => sign_extend(a, size) < sign_extend(b, size),
        Op::IntSignedLessEqual => sign_extend(a, size) <= sign_extend(b, size),
        Op::IntCarry => bits < 64 && ((a as u128 + b as u128) >> bits) != 0,
        Op::IntSignedCarry => {
            let s = sign_extend(a, size) as i128 + sign_extend(b, size) as i128;
            let r = sign_extend(a.wrapping_add(b), size) as i128;
            s != r
        }
        Op::IntSignedBorrow => {
            let s = sign_extend(a, size) as i128 - sign_extend(b, size) as i128;
            let r = sign_extend(a.wrapping_sub(b), size) as i128;
            s != r
        }
        Op::BoolAnd => (a != 0) && (b != 0),
        Op::BoolOr => (a != 0) || (b != 0),
        Op::BoolXor => (a != 0) ^ (b != 0),
        _ => return None,
    };
    Some(r as u64)
}

// ──────────────────────────────────────────────────────────────────────────────
// In-VM driver: block cache + block-entry hook
// ──────────────────────────────────────────────────────────────────────────────

/// A cached `BlockGroup`: the entry address maps to all its sub-blocks (clean,
/// pre-hook) plus the global index of the first sub-block, so the engine can
/// follow internal `Target::Internal` edges when the entry-block hook fires.
struct CachedGroup {
    base: usize,
    blocks: Vec<Block>,
}

pub struct PhaseBState {
    blocks: HashMap<u64, CachedGroup>,
    engine: PhaseBEngine,
    mmio_ranges: Vec<Range<u64>>,
    /// Fast-path armed flag checked OUTSIDE the RefCell borrow in the hot hook.
    /// Lives inside PhaseBState but is read via `Rc<Cell<bool>>` by the hook closure
    /// without borrowing the RefCell.  The `Cell<bool>` field here is the ground truth;
    /// `armed_flag` (the Rc clone shared with the hook) points to the same Cell.
    armed: bool,
}

impl PhaseBState {
    pub fn reset_pass(
        &mut self,
        read_sites: HashMap<u64, StreamKey>,
        mmio_ranges: Vec<Range<u64>>,
    ) {
        self.mmio_ranges = mmio_ranges.clone();
        self.engine.reset_pass(read_sites, mmio_ranges);
    }
    pub fn set_armed(&mut self, armed: bool) {
        self.armed = armed;
    }
    pub fn finish_pass(&self) -> PassResult {
        self.engine.finish_pass()
    }
    pub fn register_summary(&mut self, addr: u64, summary: FuncSummary) {
        self.engine.summaries.insert(addr, summary);
    }
}

struct PhaseBInjector {
    state: Rc<RefCell<PhaseBState>>,
    hook: pcode::HookId,
}

/// Cap on the cached-block count.  The original 200k was ~80 MB and created
/// sustained allocation pressure over multi-hour runs; 50k covers the vast
/// majority of Cortex-M firmware while keeping the footprint under ~20 MB.
const MAX_BLOCK_CACHE: usize = 50_000;

impl icicle_vm::CodeInjector for PhaseBInjector {
    /// Inject only into the group's entry block (`group.blocks.0`), mirroring
    /// `BlockHookInjector`.  Injecting into non-entry sub-blocks corrupts the JIT.
    fn inject(&mut self, _cpu: &mut Cpu, group: &BlockGroup, code: &mut BlockTable) {
        let id = group.blocks.0;
        let entry_start = code.blocks[id].start;

        let mut st = self.state.borrow_mut();
        // Skip if already cached (re-lift after snapshot restore) or cache is full.
        if st.blocks.contains_key(&entry_start) || st.blocks.len() >= MAX_BLOCK_CACHE {
            return;
        }

        // Insert the hook AFTER the first InstructionMarker, never at position 0.
        // Icicle's JIT establishes the emulated PC at the InstructionMarker; an
        // Op::Hook placed before it leaves the PC unestablished, so the JIT falls
        // back to `jmp r13` and emits spurious host-memory writes that corrupt the
        // heap/stack (observed as crashes at garbage addresses such as 0x9f9f9f9f).
        // If the entry block has no InstructionMarker (synthetic/empty lifter
        // artefact), skip injection entirely — such blocks carry no MMIO loads the
        // taint engine needs, and position-0 insertion is precisely the crash cause.
        let entry = &mut code.blocks[id];
        let insert_pos = match entry
            .pcode
            .instructions
            .iter()
            .position(|s| s.op == pcode::Op::InstructionMarker)
        {
            Some(p) => p + 1,
            None => return,
        };

        // Cache ALL sub-blocks (clean, pre-hook) so the engine can follow the
        // group's internal control flow when the entry hook fires; taint flowing
        // through P-code past an internal branch would otherwise be missed.
        let base = group.blocks.0;
        let group_blocks: Vec<Block> = group.range().map(|i| code.blocks[i].clone()).collect();
        st.blocks.insert(entry_start, CachedGroup { base, blocks: group_blocks });

        // Inject hook only into the entry block — non-entry injection corrupts the JIT.
        let entry = &mut code.blocks[id];
        entry.pcode.instructions.insert(insert_pos, pcode::Op::Hook(self.hook).into());
        code.modified.insert(id);
    }
}

/// Install the Phase B block cache and hook into `vm`.
/// Must be called before the blocks of interest are translated.
pub fn install(vm: &mut Vm, mmio_ranges: Vec<Range<u64>>) -> (Rc<RefCell<PhaseBState>>, Rc<Cell<bool>>) {
    let armed_flag = Rc::new(Cell::new(false));
    let state = Rc::new(RefCell::new(PhaseBState {
        blocks: HashMap::new(),
        engine: PhaseBEngine::new(HashMap::new(), mmio_ranges.clone()),
        mmio_ranges,
        armed: false,
    }));

    let hook_state = state.clone();
    let hook_armed = armed_flag.clone();
    let hook = vm.cpu.add_hook(move |cpu: &mut Cpu, addr: u64| {
        // Fast path: check the Cell<bool> BEFORE any RefCell borrow or catch_unwind.
        // When disarmed (99%+ of executions), this is a single Cell::get() — essentially free.
        if !hook_armed.get() { return; }

        let cpu_raw: *mut Cpu = cpu;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let cpu = unsafe { &mut *cpu_raw };
            let Ok(mut st) = hook_state.try_borrow_mut() else { return; };
            if !st.armed { return; }
            // Destructure so `blocks` and `engine` can be borrowed independently,
            // eliminating the per-hook Vec<Block> clone that created heavy allocation
            // pressure over long runs.
            let PhaseBState { blocks, engine, mmio_ranges, .. } = &mut *st;
            if let Some(group) = blocks.get(&addr) {
                let mut env = LiveEnv { cpu, mmio_ranges };
                engine.run_group(&group.blocks, group.base, &mut env);
            }
        }));
        if result.is_err() {
            tracing::error!(
                "Phase B: hook panicked at {:#x}; disarming this pass",
                addr
            );
            hook_armed.set(false);
            if let Ok(mut st) = hook_state.try_borrow_mut() {
                st.set_armed(false);
            }
        }
    });

    vm.add_injector(PhaseBInjector { state: state.clone(), hook });
    (state, armed_flag)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Confidence enum removed: all edges are taint-confirmed.
    use icicle_vm::cpu::lifter::{Block, BlockExit, Target};
    use pcode::VarNode;

    struct MockEnv {
        regs: HashMap<i16, u64>,
        mem: HashMap<u64, u64>,
    }
    impl MockEnv {
        fn new() -> Self { Self { regs: HashMap::new(), mem: HashMap::new() } }
    }
    impl ConcreteEnv for MockEnv {
        fn read_value(&mut self, v: Value) -> Option<u64> {
            match v {
                Value::Const(c, _) => Some(c),
                Value::Var(vn) => self.regs.get(&vn.id).copied(),
            }
        }
        fn read_mem(&mut self, addr: u64, _size: u8) -> Option<u64> {
            self.mem.get(&addr).copied()
        }
    }

    fn lifter_block(pcode: pcode::Block, start: u64, end: u64, exit: BlockExit) -> Block {
        Block { pcode, entry: None, start, end, context: 0, exit, breakpoints: 0, num_instructions: 0 }
    }
    fn marker(pc: u64) -> pcode::Instruction {
        (pcode::Op::InstructionMarker, pcode::Value::Const(pc, 8)).into()
    }
    fn reg(id: i16) -> VarNode { VarNode::new(id, 4) }
    fn engine() -> PhaseBEngine {
        PhaseBEngine::new(HashMap::new(), vec![0x4000_0000..0x6000_0000])
    }

    /// An MMIO value used to compute B's load address must yield an `Address` edge.
    #[test]
    fn address_edge_confirmed_from_tainted_load_address() {
        let mut e = engine();
        e.read_sites.insert(0x100, 0x5800_0000);
        e.read_sites.insert(0x108, 0x5800_0004);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(10), Op::Load(0), reg(5)));
        p.push(marker(0x104));
        p.push((reg(7), Op::IntAdd, reg(4), reg(10)));
        p.push(marker(0x108));
        p.push((reg(1), Op::Load(0), reg(7)));

        let block = lifter_block(p, 0x100, 0x10c, BlockExit::invalid());
        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.regs.insert(4, 0x5800_0000);

        e.run_block(&block, &mut env);
        let result = e.finish_pass();

        let src = AccessContext::new(0x100, 0x5800_0000);
        let tgt = AccessContext::new(0x108, 0x5800_0004);
        assert!(result.address_edges.contains(&(src, tgt)), "expected Address edge A→B");
    }

    /// A non-looping tainted equality branch gating B yields Control (not Length).
    #[test]
    fn control_edge_confirmed_from_tainted_branch() {
        let mut e = engine();
        e.read_sites.insert(0x200, 0x5800_0008);
        e.read_sites.insert(0x300, 0x5800_0000);

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));
        p1.push(marker(0x204));
        p1.push((reg(2), Op::IntEqual, reg(1), Value::Const(3, 4)));
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            target: Target::External(Value::Const(0x300, 4)),
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);

        e.run_block(&b1, &mut env);
        e.run_block(&b2, &mut env);
        let result = e.finish_pass();

        let src = AccessContext::new(0x200, 0x5800_0008);
        let tgt = AccessContext::new(0x300, 0x5800_0000);
        let edge = result.control_edges.iter().find(|o| o.source == src && o.target == tgt);
        assert!(edge.is_some(), "expected Control observation A→B");
        let edge = edge.unwrap();
        assert!(!edge.is_length, "single activation must NOT be Length");
        assert!(edge.is_eq, "IntEqual condition must set is_eq");
    }

    /// Fix 3: A range-check (IntLess) condition must NOT set is_eq.
    #[test]
    fn range_check_branch_does_not_set_is_eq() {
        let mut e = engine();
        e.read_sites.insert(0x200, 0x5800_0008);
        e.read_sites.insert(0x300, 0x5800_0000);

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));
        p1.push(marker(0x204));
        // IntLess, not IntEqual — range guard, not discriminant
        p1.push((reg(2), Op::IntLess, reg(1), Value::Const(256, 4)));
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            target: Target::External(Value::Const(0x300, 4)),
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);

        e.run_block(&b1, &mut env);
        e.run_block(&b2, &mut env);
        let result = e.finish_pass();

        let src = AccessContext::new(0x200, 0x5800_0008);
        let tgt = AccessContext::new(0x300, 0x5800_0000);
        let edge = result.control_edges.iter().find(|o| o.source == src && o.target == tgt);
        assert!(edge.is_some());
        assert!(!edge.unwrap().is_eq, "IntLess condition must NOT set is_eq");
    }

    /// Fix 5: A loop where B is read exactly 2 times must NOT be Length (needs ≥3).
    #[test]
    fn count_two_is_not_length() {
        let mut e = engine();
        e.read_sites.insert(0x200, 0x5800_0008);
        e.read_sites.insert(0x300, 0x5800_0000);

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));
        p1.push(marker(0x204));
        p1.push((reg(2), Op::IntLess, reg(3), reg(1)));
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            target: Target::External(Value::Const(0x300, 4)),
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);
        env.regs.insert(3, 0);

        // Only 2 iterations → count=2, below the ≥3 threshold.
        for _ in 0..2 {
            e.run_block(&b1, &mut env);
            e.run_block(&b2, &mut env);
        }
        let result = e.finish_pass();

        let src = AccessContext::new(0x200, 0x5800_0008);
        let tgt = AccessContext::new(0x300, 0x5800_0000);
        let edge = result.control_edges.iter().find(|o| o.source == src && o.target == tgt);
        assert!(edge.is_some());
        assert!(!edge.unwrap().is_length, "count=2 must NOT be classified as Length");
    }

    /// A tainted loop branch (IntLess) where B is read ≥3 times yields `Length`.
    #[test]
    fn length_edge_from_looping_tainted_branch() {
        let mut e = engine();
        e.read_sites.insert(0x200, 0x5800_0008);
        e.read_sites.insert(0x300, 0x5800_0000);

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));
        p1.push(marker(0x204));
        p1.push((reg(2), Op::IntLess, reg(3), reg(1)));
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            target: Target::External(Value::Const(0x300, 4)),
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);
        env.regs.insert(3, 0);

        for _ in 0..3 {
            e.run_block(&b1, &mut env);
            e.run_block(&b2, &mut env);
        }
        let result = e.finish_pass();

        let src = AccessContext::new(0x200, 0x5800_0008);
        let tgt = AccessContext::new(0x300, 0x5800_0000);
        let edge = result
            .control_edges
            .iter()
            .find(|o| o.source == src && o.target == tgt)
            .expect("expected a control/length observation");
        assert!(edge.is_length, "looping tainted branch with ≥3 reads must be Length");
    }

    /// Fix 6: apply_pass_result pools Length samples at address granularity —
    /// two different source PCs for the same addr pair contribute to one fit.
    #[test]
    fn length_samples_pool_at_address_granularity() {
        let mut graph = StreamRelationGraph::new();
        let mut store = LengthSampleStore::new();

        let src_a = AccessContext::new(0x100, 0x5800_0008); // PC 0x100
        let src_b = AccessContext::new(0x200, 0x5800_0008); // same addr, different PC
        let tgt   = AccessContext::new(0x300, 0x5800_0000);

        // Pass 1: sample from PC 0x100 → (4, 4)
        apply_pass_result(&mut graph, &mut store, PassResult {
            address_edges: vec![],
            control_edges: vec![ControlEdgeObs {
                source: src_a, target: tgt, is_length: true, is_eq: false,
                sample: Some((4, 4)),
            }],
        });
        // Pass 2: sample from PC 0x200 → (7, 7)
        // Both should be in the same pool (same stream addresses).
        apply_pass_result(&mut graph, &mut store, PassResult {
            address_edges: vec![],
            control_edges: vec![ControlEdgeObs {
                source: src_b, target: tgt, is_length: true, is_eq: false,
                sample: Some((7, 7)),
            }],
        });

        let edge = graph.confirmed_edges().next().expect("edge must be confirmed");
        assert_eq!(edge.kind, EdgeKind::Length);
        // Two distinct (val, count) pairs from the shared pool → Identity fit.
        assert!(edge.relation.is_some(), "pooled samples must produce a relation fit");
    }

    /// `apply_pass_result` upgrades a structural Control candidate to Length and
    /// fits a relation across repeated samples.
    #[test]
    fn apply_results_confirms_and_fits() {
        let mut graph = StreamRelationGraph::new();
        let mut store = LengthSampleStore::new();

        let src = AccessContext::new(0x200, 0x5800_0008);
        let tgt = AccessContext::new(0x300, 0x5800_0000);

        for v in [4u64, 7u64] {
            apply_pass_result(&mut graph, &mut store, PassResult {
                address_edges: vec![],
                control_edges: vec![ControlEdgeObs {
                    source: src, target: tgt, is_length: true, is_eq: false,
                    sample: Some((v, v)),
                }],
            });
        }

        let edge = graph.confirmed_edges().find(|e| e.source == src && e.target == tgt)
            .expect("edge should be confirmed");
        assert_eq!(edge.kind, EdgeKind::Length);
        assert!(edge.relation.is_some());
    }

    /// Issue #2: when a source both computes B's address AND gates a branch to B,
    /// `apply_pass_result` confirms address_edges before control_edges.  The edge
    /// must end up classified `Address`, not clobbered to `Control` by the later
    /// control confirmation.
    #[test]
    fn apply_results_keeps_address_when_source_also_gates() {
        let mut graph = StreamRelationGraph::new();
        let mut store = LengthSampleStore::new();

        let src = AccessContext::new(0x200, 0x5800_0008);
        let tgt = AccessContext::new(0x300, 0x5800_0000);

        // One pass reports the SAME source on both channels (address + gating).
        apply_pass_result(&mut graph, &mut store, PassResult {
            address_edges: vec![(src, tgt)],
            control_edges: vec![ControlEdgeObs {
                source: src, target: tgt, is_length: false, is_eq: true,
                sample: Some((4, 1)),
            }],
        });

        let edge = graph.confirmed_edges().find(|e| e.source == src && e.target == tgt)
            .expect("edge should be confirmed");
        assert_eq!(edge.kind, EdgeKind::Address, "address role must win over gating role");
        // An Address edge carries no discriminant value_set.
        assert!(edge.value_set.is_empty(), "Address edge must not accumulate discriminants");
    }

    /// Fix 3: Discriminant is captured only for equality-gated Control edges.
    #[test]
    fn apply_results_captures_control_discriminant_only_for_equality_gate() {
        let mut graph = StreamRelationGraph::new();
        let mut store = LengthSampleStore::new();

        let src = AccessContext::new(0x200, 0x5800_0008);
        let tgt = AccessContext::new(0x300, 0x5800_0000);

        // Pass 1: is_eq=true → value 3 must be recorded.
        apply_pass_result(&mut graph, &mut store, PassResult {
            address_edges: vec![],
            control_edges: vec![ControlEdgeObs {
                source: src, target: tgt, is_length: false, is_eq: true,
                sample: Some((3, 1)),
            }],
        });
        // Pass 2: is_eq=false (range check) → value 42 must NOT be recorded.
        apply_pass_result(&mut graph, &mut store, PassResult {
            address_edges: vec![],
            control_edges: vec![ControlEdgeObs {
                source: src, target: tgt, is_length: false, is_eq: false,
                sample: Some((42, 1)),
            }],
        });

        let edge = graph.confirmed_edges().find(|e| e.source == src && e.target == tgt)
            .expect("Control edge should be confirmed");
        assert_eq!(edge.kind, EdgeKind::Control);
        let vals: Vec<u64> = edge.value_set.iter().map(|&(v, _)| v).collect();
        assert_eq!(vals, vec![3], "only the equality-gated value must be captured");
        assert!(!vals.contains(&42), "range-check value must not be captured as discriminant");
    }

    /// A Control observation without a prior structural candidate now directly
    /// inserts an edge via `insert_or_confirm_edge`.
    #[test]
    fn apply_results_inserts_edge_without_structural_backing() {
        let mut graph = StreamRelationGraph::new();
        let mut store = LengthSampleStore::new();
        let src = AccessContext::new(0x200, 0x5800_0008);
        let tgt = AccessContext::new(0x300, 0x5800_0000);
        apply_pass_result(&mut graph, &mut store, PassResult {
            address_edges: vec![],
            control_edges: vec![ControlEdgeObs {
                source: src, target: tgt, is_length: false, is_eq: true,
                sample: Some((9, 1)),
            }],
        });
        assert!(graph.edge_count() > 0, "observation must directly insert an edge");
    }

    /// Deferred source-value capture from the next block entry (production LiveEnv path).
    #[test]
    fn deferred_source_value_capture_from_register() {
        let mut e = engine();
        e.read_sites.insert(0x100, 0x5800_0008);

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x100));
        p1.push((reg(5), Op::Load(0), reg(9)));
        let b1 = lifter_block(p1, 0x100, 0x104, BlockExit::invalid());

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x104));
        let b2 = lifter_block(p2, 0x104, 0x108, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);

        e.run_block(&b1, &mut env);
        let src = AccessContext::new(0x100, 0x5800_0008);
        assert_eq!(e.source_value(src), None);

        env.regs.insert(5, 0x2a);
        e.run_block(&b2, &mut env);
        assert_eq!(e.source_value(src), Some(0x2a));
    }

    /// `run_block` must not panic when an `InstructionMarker` has a non-constant
    /// first input.  This can occur with synthetic ARM/Thumb blocks where the SLEIGH
    /// lifter or a rewriter emits an InstructionMarker with a `VarNode` address.
    /// The safe `if let Value::Const` guard must keep the previous `cur_pc` value
    /// (initialized to `block.start`) rather than calling `as_u64()` and panicking.
    #[test]
    fn run_block_does_not_panic_on_non_const_instruction_marker() {
        let mut e = engine();

        // Build a block where InstructionMarker has a VarNode input (synthesised via
        // the low-level Instruction constructor) instead of a constant PC.
        let malformed_marker = pcode::Instruction {
            op: pcode::Op::InstructionMarker,
            inputs: pcode::Inputs::new(pcode::Value::Var(VarNode::new(5, 8)), pcode::Value::Const(4, 8)),
            output: VarNode::NONE,
        };
        let mut p = pcode::Block::new();
        p.instructions.push(malformed_marker);
        let block = lifter_block(p, 0xdead_0000, 0xdead_0004, BlockExit::invalid());
        let mut env = MockEnv::new();

        // Must not panic; cur_pc should stay at block.start (0xdead_0000).
        e.run_block(&block, &mut env);
    }

    /// A call (PcodeOp) clears r0–r3 taint so it cannot leak across opaque calls.
    #[test]
    fn call_kills_argument_register_taint() {
        let mut e = engine();
        e.read_sites.insert(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(0), Op::Load(0), reg(5)));
        p.push(marker(0x104));
        p.push((VarNode::NONE, Op::PcodeOp(0), reg(0)));
        let block = lifter_block(p, 0x100, 0x108, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        e.run_block(&block, &mut env);

        assert!(e.shadow.reg_tag(0).is_clean(), "r0 taint must be killed after an opaque call");
    }

    /// Build a 3-sub-block group mirroring an internal-branch instruction (e.g. an
    /// ARM IT-block / divide zero-check):
    ///   L0 (entry, marker 0x100): r10 = LOAD[mmioA];  r2 = (r10 == 5)
    ///       Branch r2 → L1 (taken)  else → L2
    ///   L1 (marker 0x104):  r7 = r4 + r10;  r1 = LOAD[r7]   (B's MMIO read)
    ///   L2 (marker 0x108):  (fallthrough, empty)
    /// `mem[mmioA] = a_val` makes the branch condition concrete so `run_group`
    /// knows which sub-block the CPU took.
    fn it_block_group(a_val: u64) -> (Vec<Block>, MockEnv) {
        let mmio_a = 0x5800_0000u64;
        let b_base = 0x2000_0000u64; // regular RAM, not MMIO

        let mut p0 = pcode::Block::new();
        p0.push(marker(0x100));
        p0.push((reg(10), Op::Load(0), reg(5)));        // r10 = LOAD[mmioA]
        p0.push((reg(2), Op::IntEqual, reg(10), Value::Const(5, 4)));
        let b0 = lifter_block(p0, 0x100, 0x104, BlockExit::Branch {
            cond: Value::Var(reg(2)),
            target: Target::Internal(1),
            fallthrough: Target::Internal(2),
        });

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x104));
        p1.push((reg(7), Op::IntAdd, reg(4), reg(10)));  // B addr = base + mmioA value
        p1.push((reg(1), Op::Load(0), reg(7)));          // r1 = LOAD[r7]  (B's MMIO read)
        let b1 = lifter_block(p1, 0x104, 0x108, BlockExit::Jump { target: Target::Internal(2) });

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x108));
        let b2 = lifter_block(p2, 0x108, 0x10c, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, mmio_a);
        env.regs.insert(4, b_base);
        env.mem.insert(mmio_a, a_val); // makes r10 (and thus the branch) concrete
        (vec![b0, b1, b2], env)
    }

    /// `run_group` must interpret a non-entry sub-block reached by a taken internal
    /// branch, so taint flowing through it (B's address derived from A) is observed.
    #[test]
    fn group_replay_observes_taken_internal_subblock() {
        let mut e = engine();
        e.read_sites.insert(0x100, 0x5800_0000); // stream A
        e.read_sites.insert(0x104, 0x5800_0004); // stream B (in non-entry sub-block L1)

        let (blocks, mut env) = it_block_group(5); // 5 == 5 → branch taken → L1 runs
        e.run_group(&blocks, 0, &mut env);
        let result = e.finish_pass();

        let src = AccessContext::new(0x100, 0x5800_0000);
        let tgt = AccessContext::new(0x104, 0x5800_0004);
        assert!(
            result.address_edges.contains(&(src, tgt)),
            "Address edge A→B from the taken sub-block must be observed, got {:?}",
            result.address_edges
        );
    }

    /// `run_group` must NOT interpret the untaken sub-block: doing so would
    /// manufacture a spurious edge from P-code the CPU never executed.
    #[test]
    fn group_replay_skips_untaken_internal_subblock() {
        let mut e = engine();
        e.read_sites.insert(0x100, 0x5800_0000); // stream A
        e.read_sites.insert(0x104, 0x5800_0004); // stream B (in non-entry sub-block L1)

        let (blocks, mut env) = it_block_group(7); // 7 != 5 → fallthrough → L1 skipped
        e.run_group(&blocks, 0, &mut env);
        let result = e.finish_pass();

        let src = AccessContext::new(0x100, 0x5800_0000);
        let tgt = AccessContext::new(0x104, 0x5800_0004);
        assert!(
            !result.address_edges.contains(&(src, tgt)),
            "untaken sub-block must not be interpreted, but produced edge {:?}",
            result.address_edges
        );
    }

    /// `BlockExit::Call` to an unknown function kills r0–r3.
    #[test]
    fn call_exit_kills_regs_for_unknown_function() {
        let mut e = engine();
        let mut t0 = TaintTag::Clean; t0.set_stream(0);
        let mut t1 = TaintTag::Clean; t1.set_stream(1);
        e.shadow.set_reg_tag(0, t0);
        e.shadow.set_reg_tag(1, t1);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        let block = lifter_block(p, 0x100, 0x104, BlockExit::Call {
            target: Value::Const(0x1234_5678, 4),
            fallthrough: 0x104,
        });
        let mut env = MockEnv::new();
        e.run_group(&[block], 0, &mut env);

        for r in 0..4i16 {
            assert!(e.shadow.reg_tag(r).is_clean(), "r{r} must be killed for unknown call");
        }
    }

    #[test]
    fn strlen_summary_propagates_string_buffer_taint_to_r0() {
        let mut e = engine();
        e.summaries.insert(0xDEAD_0001, FuncSummary::Strlen);

        let mut buf_tag = TaintTag::Clean;
        buf_tag.set_stream(2);
        e.shadow.set_mem_tag_range(0x2000_2000, 4, buf_tag);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        let block = lifter_block(p, 0x100, 0x104, BlockExit::Call {
            target: Value::Const(0xDEAD_0001, 4),
            fallthrough: 0x104,
        });

        let mut env = MockEnv::new();
        env.regs.insert(0, 0x2000_2000u64);

        e.run_group(&[block], 0, &mut env);

        let r0_tag = e.shadow.reg_tag(0);
        assert!(!r0_tag.is_clean(), "strlen summary must propagate string buffer taint to r0");
        for r in 1..4i16 {
            assert!(e.shadow.reg_tag(r).is_clean(), "r{r} must be killed after strlen call");
        }
    }

    #[test]
    fn strlen_summary_stops_at_nul_byte() {
        let mut e = engine();
        e.summaries.insert(0xDEAD_0001, FuncSummary::Strlen);

        let mut t0 = TaintTag::Clean; t0.set_stream(0);
        let mut t_past = TaintTag::Clean; t_past.set_stream(7);
        e.shadow.mem.insert(0x2000_3000u32, t0);
        e.shadow.mem.insert(0x2000_3002u32, t_past);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        let block = lifter_block(p, 0x100, 0x104, BlockExit::Call {
            target: Value::Const(0xDEAD_0001, 4),
            fallthrough: 0x104,
        });

        let mut env = MockEnv::new();
        env.regs.insert(0, 0x2000_3000u64);
        env.mem.insert(0x2000_3000, 0x41);
        env.mem.insert(0x2000_3001, 0x00);

        e.run_group(&[block], 0, &mut env);

        let r0_tag = e.shadow.reg_tag(0);
        let streams: Vec<usize> = r0_tag.iter_streams().collect();
        assert!(streams.contains(&0), "byte before NUL must contribute its taint");
        assert!(!streams.contains(&7), "bytes past NUL must not contribute taint");
    }

    #[test]
    fn passthrough_summary_preserves_r0_taint() {
        let mut e = engine();
        e.summaries.insert(0xDEAD_0002, FuncSummary::Passthrough);

        let mut t = TaintTag::Clean; t.set_stream(5);
        e.shadow.set_reg_tag(0, t);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        let block = lifter_block(p, 0x100, 0x104, BlockExit::Call {
            target: Value::Const(0xDEAD_0002, 4),
            fallthrough: 0x104,
        });

        let mut env = MockEnv::new();
        env.regs.insert(0, 0x1234u64);

        e.run_group(&[block], 0, &mut env);

        let r0_tag = e.shadow.reg_tag(0);
        assert!(!r0_tag.is_clean(), "passthrough summary must preserve r0 taint tag");
        let streams: Vec<usize> = r0_tag.iter_streams().collect();
        assert!(streams.contains(&5), "original stream 5 taint must survive passthrough");
        for r in 1..4i16 {
            assert!(e.shadow.reg_tag(r).is_clean(), "r{r} must be killed by passthrough call");
        }
    }

    #[test]
    fn memcpy_summary_propagates_shadow_memory() {
        let mut e = engine();
        e.summaries.insert(0xDEAD_0000, FuncSummary::Memcpy);

        let mut src_tag = TaintTag::Clean;
        src_tag.set_stream(3);
        e.shadow.set_mem_tag_range(0x2000_1000, 4, src_tag);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        let block = lifter_block(p, 0x100, 0x104, BlockExit::Call {
            target: Value::Const(0xDEAD_0000, 4),
            fallthrough: 0x104,
        });

        let mut env = MockEnv::new();
        env.regs.insert(0, 0x2000_0000u64);
        env.regs.insert(1, 0x2000_1000u64);
        env.regs.insert(2, 4u64);

        e.run_group(&[block], 0, &mut env);

        let dst_tag = e.shadow.mem_tag_range(0x2000_0000, 4);
        assert!(!dst_tag.is_clean(), "memcpy summary must propagate taint from src to dst buffer");
        for r in 0..4i16 {
            assert!(e.shadow.reg_tag(r).is_clean(), "r{r} must be killed after memcpy call");
        }
    }

    // ── Fix 1: gating recency window ────────────────────────────────────────

    /// Simulate the stmclk_init_sysclk → led_init false-positive pattern:
    ///
    ///   Block 1 (PLL poll): reads MMIO_A in a tight loop (branch visited ≥2 times)
    ///   ...many blocks of gap (simulated by running empty blocks)...
    ///   Block N (GPIO init): reads MMIO_B ≥3 times
    ///
    /// Without the recency window, MMIO_A → MMIO_B would be classified as Length.
    /// With the window, the gap exceeds GATING_WINDOW and no Length edge is emitted.
    #[test]
    fn recency_window_kills_boot_to_gpio_false_positive() {
        let mut e = engine();
        // Set a small window for the test (the production default is 128).
        e.gating_window = 16;

        e.read_sites.insert(0x200, 0x4002_1000); // RCC_CR (source)
        e.read_sites.insert(0x800, 0x4800_0004); // GPIOA_OTYPER (target)

        // Block 1: tainted polling loop (PLL-ready).  Read RCC_CR, branch on it.
        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));
        p1.push(marker(0x204));
        p1.push((reg(2), Op::IntEqual, reg(1), Value::Const(0x2000000, 4)));
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            target: Target::External(Value::Const(0x200, 4)), // back-edge
            fallthrough: Target::External(Value::Const(0x300, 4)),
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        // Simulate the loop body: run b1 three times (is_loop=true, fires≥2).
        let mut env = MockEnv::new();
        env.regs.insert(9, 0x4002_1000);
        env.regs.insert(8, 0x4800_0004);
        for _ in 0..3 {
            e.run_block(&b1, &mut env);
        }

        // Gap: 30 empty blocks (exceeds window of 16).
        for i in 0..30u64 {
            let mut pg = pcode::Block::new();
            pg.push(marker(0x300 + i * 4));
            let bg = lifter_block(pg, 0x300 + i * 4, 0x304 + i * 4, BlockExit::invalid());
            e.run_block(&bg, &mut env);
        }

        // Block N: GPIO init — read MMIO_B ≥3 times.
        for _ in 0..4 {
            let mut p2 = pcode::Block::new();
            p2.push(marker(0x800));
            p2.push((reg(5), Op::Load(0), reg(8)));
            let b2 = lifter_block(p2, 0x800, 0x804, BlockExit::invalid());
            e.run_block(&b2, &mut env);
        }

        let result = e.finish_pass();
        let src = AccessContext::new(0x200, 0x4002_1000);
        let tgt = AccessContext::new(0x800, 0x4800_0004);
        let edge = result.control_edges.iter().find(|o| o.source == src && o.target == tgt);
        assert!(
            edge.is_none(),
            "gating recency window must suppress boot-to-GPIO false positive, but got {edge:?}"
        );
    }

    /// Positive counterpart: a genuine in-loop read (target inside the loop body,
    /// within the recency window) is still classified as Length.
    #[test]
    fn recency_window_preserves_genuine_in_loop_length() {
        let mut e = engine();
        e.gating_window = 128; // default

        e.read_sites.insert(0x200, 0x5800_0008); // source (loop counter)
        e.read_sites.insert(0x300, 0x5800_0000); // target (buffer read inside loop)

        // Block 1: tainted branch (loop header).
        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));
        p1.push(marker(0x204));
        p1.push((reg(2), Op::IntLess, reg(3), reg(1)));
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            target: Target::External(Value::Const(0x300, 4)),
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        // Block 2: target read (loop body) — immediately after the branch.
        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);
        env.regs.insert(3, 0);

        // 4 iterations: branch → body → branch → body → ...
        for _ in 0..4 {
            e.run_block(&b1, &mut env);
            e.run_block(&b2, &mut env);
        }

        let result = e.finish_pass();
        let src = AccessContext::new(0x200, 0x5800_0008);
        let tgt = AccessContext::new(0x300, 0x5800_0000);
        let edge = result.control_edges.iter().find(|o| o.source == src && o.target == tgt);
        assert!(edge.is_some(), "genuine in-loop read must still produce an edge");
        assert!(edge.unwrap().is_length, "genuine in-loop read must be classified as Length");
    }

    // ── Fix 3: spinloop discriminant suppression ────────────────────────────

    /// A high-iteration spinloop (≥ SPINLOOP_VISIT_THRESHOLD visits) must NOT
    /// set is_eq, even if the condition is derived from IntEqual.  The exit
    /// values of a busy-wait are not useful selectors.
    #[test]
    fn spinloop_does_not_set_is_eq() {
        let mut e = engine();
        e.read_sites.insert(0x200, 0x5800_0008);
        e.read_sites.insert(0x300, 0x5800_0000);

        // Polling block: MMIO_A == 0x2000000 (e.g. PLL ready flag)
        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));
        p1.push(marker(0x204));
        p1.push((reg(2), Op::IntEqual, reg(1), Value::Const(0x2000000, 4)));
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            target: Target::External(Value::Const(0x300, 4)),
            fallthrough: Target::External(Value::Const(0x200, 4)), // back to self
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        // Target block (downstream read).
        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);

        // Iterate the spinloop many times past SPINLOOP_VISIT_THRESHOLD.
        for _ in 0..20 {
            e.run_block(&b1, &mut env);
        }
        e.run_block(&b2, &mut env);

        let result = e.finish_pass();
        let src = AccessContext::new(0x200, 0x5800_0008);
        let tgt = AccessContext::new(0x300, 0x5800_0000);
        let edge = result.control_edges.iter().find(|o| o.source == src && o.target == tgt);
        // The edge may or may not exist (depends on window), but if it exists
        // its is_eq must be false.
        if let Some(obs) = edge {
            assert!(
                !obs.is_eq,
                "spinloop (>={SPINLOOP_VISIT_THRESHOLD} visits) must suppress is_eq, \
                 even for IntEqual conditions"
            );
        }
    }

    /// A low-iteration equality branch (< SPINLOOP_VISIT_THRESHOLD) still sets is_eq.
    #[test]
    fn low_iteration_equality_branch_sets_is_eq() {
        let mut e = engine();
        e.read_sites.insert(0x200, 0x5800_0008);
        e.read_sites.insert(0x300, 0x5800_0000);

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));
        p1.push(marker(0x204));
        p1.push((reg(2), Op::IntEqual, reg(1), Value::Const(3, 4)));
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            target: Target::External(Value::Const(0x300, 4)),
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);

        // Only 2 iterations — well below SPINLOOP_VISIT_THRESHOLD.
        for _ in 0..2 {
            e.run_block(&b1, &mut env);
            e.run_block(&b2, &mut env);
        }

        let result = e.finish_pass();
        let src = AccessContext::new(0x200, 0x5800_0008);
        let tgt = AccessContext::new(0x300, 0x5800_0000);
        let edge = result.control_edges.iter().find(|o| o.source == src && o.target == tgt);
        assert!(edge.is_some(), "low-iteration branch must produce an edge");
        assert!(edge.unwrap().is_eq, "low-iteration IntEqual branch must set is_eq");
    }
}
