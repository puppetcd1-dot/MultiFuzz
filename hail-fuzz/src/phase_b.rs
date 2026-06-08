//! Phase B — dynamic taint pass for typed stream-relationship classification.
//!
//! Phase A (`mmio_flow.rs`) produces *structural* candidates by backward demand
//! slicing the P-code IR.  It over-approximates control dependence (any MMIO read
//! feeding any branch on a path to B) and cannot distinguish `Control` from
//! `Length`/`Stride`, nor confirm that a value genuinely flows.
//!
//! Phase B re-executes a seed with byte-granular dynamic taint tracking and
//! classifies / confirms edges from observed runtime behaviour:
//!   * the loaded value of an MMIO read reaching **B's load-address register**
//!     → `Address` (precise value flow);
//!   * an MMIO value reaching a **branch condition** that gates B → `Control`
//!     (taint-confirmed; prunes Phase A false positives where taint never
//!     actually reached the condition);
//!   * a `Control` whose gating branch is a runtime **loop** (its PC is revisited)
//!     under which B is read **≥2 times** → `Length` (loop trip count), with an
//!     optional fitted `Relation` extracted from `(value_A, count_B)` samples.
//!
//! ## Engine design
//!
//! The engine interprets a block's P-code against a [`ShadowState`], driven by a
//! [`ConcreteEnv`] that supplies runtime register/memory values.  Decoupling the
//! concrete value source behind a trait makes the dataflow logic unit-testable
//! with a mock environment (no live VM required) and backed by the real `Cpu` in
//! production via [`LiveEnv`].
//!
//! The in-VM driver caches each translated block (via a [`CodeInjector`]) and
//! installs a block-entry `Op::Hook` that runs the engine over the cached block
//! using the live CPU state.  Because block-entry hooks fire *before* the block
//! executes (and after all prior blocks have executed), the shadow state stays
//! consistent with sequential block execution; register reads at block entry are
//! the correct pre-block values.

use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;

use hashbrown::{HashMap, HashSet};
use icicle_vm::{
    cpu::lifter::Block,
    cpu::{BlockGroup, Cpu},
    BlockTable, Vm,
};
use pcode::{Op, Value, VarId, VarNode};

use crate::{
    input::StreamKey,
    stream_relation::{AccessContext, EdgeKind, Relation, StreamRelationGraph},
    taint::{ShadowState, TaintTag},
};

// ──────────────────────────────────────────────────────────────────────────────
// Concrete value environment
// ──────────────────────────────────────────────────────────────────────────────

/// Supplies concrete runtime values for the taint interpreter.  Implemented by
/// [`LiveEnv`] over a real `Cpu` and by a mock in unit tests.
pub trait ConcreteEnv {
    /// Concrete value of a register/const at block entry.  `None` if unavailable.
    fn read_value(&mut self, v: Value) -> Option<u64>;
    /// Read `size` bytes of *non-MMIO* memory at `addr`.  Implementations MUST return
    /// `None` for MMIO-mapped addresses to avoid input-consuming side effects.
    fn read_mem(&mut self, addr: u64, size: u8) -> Option<u64>;
}

/// Production `ConcreteEnv` backed by the live emulator CPU.
pub struct LiveEnv<'a> {
    pub cpu: &'a mut Cpu,
    /// MMIO ranges — reads from these are suppressed (would consume fuzz input).
    pub mmio_ranges: &'a [Range<u64>],
}

impl<'a> ConcreteEnv for LiveEnv<'a> {
    fn read_value(&mut self, v: Value) -> Option<u64> {
        Some(icicle_vm::cpu::read_value_zxt(self.cpu, v))
    }

    fn read_mem(&mut self, addr: u64, size: u8) -> Option<u64> {
        use icicle_vm::cpu::mem::perm;
        if self.mmio_ranges.iter().any(|r| r.contains(&addr)) {
            // Reading MMIO would trigger the IoMemory handler and consume a fuzz byte.
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

// ──────────────────────────────────────────────────────────────────────────────
// Pass results
// ──────────────────────────────────────────────────────────────────────────────

/// A single control-channel observation collapsed from one taint pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlEdgeObs {
    pub source: AccessContext,
    pub target: AccessContext,
    /// The gating branch was a runtime loop and B was read ≥3 times → `Length`.
    pub is_length: bool,
    /// The condition that gated the branch to B was derived from an equality
    /// comparison (`IntEqual`/`IntNotEqual`), not a range/arithmetic check.
    /// Only equality-gated sources produce meaningful discriminants: a value that
    /// passes `cmd == 3` IS a discriminant; a value that passes `size < 256` is not.
    pub is_eq: bool,
    /// `(source_value, target_read_count)` for relation fitting, when the source
    /// value could be observed without MMIO side effects.
    pub sample: Option<(u64, u64)>,
}

/// Everything one taint pass discovered, ready to be merged into the graph.
#[derive(Debug, Default, Clone)]
pub struct PassResult {
    /// Confirmed `Address` edges (source value reached B's load address).
    pub address_edges: Vec<(AccessContext, AccessContext)>,
    /// `Control`/`Length` observations (source value reached a branch gating B).
    pub control_edges: Vec<ControlEdgeObs>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Length-relation sample store (accumulates across passes)
// ──────────────────────────────────────────────────────────────────────────────

/// Accumulates `(value_A, count_B)` samples per `Length` edge across multiple
/// taint passes so a `Relation` can be fitted once enough distinct points exist.
///
/// Keyed by `(source_addr, target_addr)` rather than by full `AccessContext` so
/// that multiple read PCs for the same source→target stream pair pool their samples
/// into a single list.  Five distinct read sites accessing the same DMA register all
/// contribute to the same fit, giving the estimator much more statistical power.
#[derive(Default)]
pub struct LengthSampleStore {
    samples: HashMap<(StreamKey, StreamKey), Vec<(u64, u64)>>,
}

impl LengthSampleStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a sample and return the best `Relation` fit so far (if any).
    pub fn record(
        &mut self,
        edge: (AccessContext, AccessContext),
        sample: (u64, u64),
    ) -> Option<Relation> {
        // Project to address granularity: multiple read-site PCs for the same
        // stream-level edge share one sample list for better fitting power.
        let key = (edge.0.addr, edge.1.addr);
        let v = self.samples.entry(key).or_default();
        if !v.contains(&sample) {
            v.push(sample);
        }
        Relation::fit(v)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Taint engine
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
struct CtrlObs {
    /// Number of times B was read while this source was a known gating source.
    fires: u64,
    /// The gating branch was observed as a runtime loop (its PC was revisited).
    is_loop: bool,
    /// At least one activating branch for this source→target pair was derived from
    /// an equality comparison (`IntEqual`/`IntNotEqual`).  Only equality-gated
    /// observations produce useful discriminants; inequality checks (range guards)
    /// produce noise.
    is_eq: bool,
}

/// Dynamic taint interpreter operating one block at a time.
pub struct PhaseBEngine {
    shadow: ShadowState,
    /// Observed MMIO read sites: load PC → stream key.  Refreshed at arm time.
    read_sites: HashMap<u64, StreamKey>,
    mmio_ranges: Vec<Range<u64>>,

    // ── per-block scratch (negative ids = temporaries, cleared per block) ──
    def_concrete: HashMap<VarId, Option<u64>>,
    def_taint: HashMap<VarId, TaintTag>,
    /// Varnodes whose value was produced by an `IntEqual` / `IntNotEqual` op
    /// (or `BoolNot` of one) in the current block.  Used to gate discriminant
    /// capture on equality-type conditions at block exit.
    def_eq_derived: HashSet<VarId>,
    cur_pc: u64,

    // ── per-pass state ──
    /// Branch PCs at which the tainted condition was derived from an equality
    /// comparison this pass.  Persists across blocks so MMIO reads attributed
    /// to that branch later in the pass correctly inherit is_eq.
    branch_is_eq: HashSet<u64>,

    // ── per-pass observations ──
    /// How many times each MMIO context was read this pass.
    target_reads: HashMap<AccessContext, u64>,
    /// Last concrete value loaded at each MMIO context (for relation samples).
    source_last_val: HashMap<AccessContext, u64>,
    /// MMIO source reads whose value was not yet observable (production: `LiveEnv`
    /// suppresses MMIO reads to avoid consuming a fuzz byte).  Each entry is the
    /// `(context, destination register)` of a source load; the value is captured at
    /// the *next* block entry, by which point the real CPU has executed the load and
    /// the register holds the loaded value (block-entry hooks fire pre-execution).
    pending_src_vals: Vec<(AccessContext, VarId)>,
    /// Tainted branch PCs and how many times each was evaluated tainted (loop ⇒ ≥2).
    branch_visits: HashMap<u64, u64>,
    /// Accumulated gating `(source, branch_pc)` pairs (taint-confirmed).
    gating: Vec<(AccessContext, u64)>,
    gating_set: HashSet<(AccessContext, u64)>,
    /// Address-channel confirmations.
    addr_edges: HashSet<(AccessContext, AccessContext)>,
    /// Control-channel observations keyed by `(source, target)`.
    ctrl_obs: HashMap<(AccessContext, AccessContext), CtrlObs>,
}

impl PhaseBEngine {
    pub fn new(read_sites: HashMap<u64, StreamKey>, mmio_ranges: Vec<Range<u64>>) -> Self {
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
            addr_edges: HashSet::new(),
            ctrl_obs: HashMap::new(),
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
        self.addr_edges.clear();
        self.ctrl_obs.clear();
    }

    // ── value/taint accessors ─────────────────────────────────────────────────

    /// Concrete value of an input, consulting in-block defs first, then `env`.
    fn concrete_in(&self, v: Value, env: &mut dyn ConcreteEnv) -> Option<u64> {
        match v {
            Value::Const(c, _) => Some(c),
            Value::Var(vn) => {
                if vn.is_invalid() {
                    return None;
                }
                if let Some(c) = self.def_concrete.get(&vn.id) {
                    return *c;
                }
                if vn.id > 0 {
                    env.read_value(Value::Var(vn))
                } else {
                    None // undefined temporary
                }
            }
        }
    }

    /// Taint of an input, consulting in-block defs first, then the shadow registers.
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

    // ── block interpretation ───────────────────────────────────────────────────

    /// Interpret a single basic block, updating shadow state and observations.
    pub fn run_block(&mut self, block: &Block, env: &mut dyn ConcreteEnv) {
        // Temporaries do not survive block boundaries; register defs are re-seeded
        // from `env` (the live block-entry state) on first use within the block.
        self.def_concrete.clear();
        self.def_taint.clear();
        self.def_eq_derived.clear();
        self.cur_pc = block.start;

        // Capture the concrete value of any source MMIO read from the previous block:
        // its load has now executed, so the destination register holds the loaded value.
        if !self.pending_src_vals.is_empty() {
            for (ctx, reg) in std::mem::take(&mut self.pending_src_vals) {
                if let Some(v) = env.read_value(Value::Var(VarNode::new(reg, 4))) {
                    self.source_last_val.entry(ctx).or_insert(v);
                }
            }
        }

        for stmt in &block.pcode.instructions {
            match stmt.op {
                Op::InstructionMarker => {
                    self.cur_pc = stmt.inputs.first().as_u64();
                }

                Op::Load(_) => {
                    let size = stmt.output.size.max(1);
                    let addr_input = stmt.inputs.first();
                    let addr = self.concrete_in(addr_input, env);

                    if let Some(&key) = self.read_sites.get(&self.cur_pc) {
                        // This load is an MMIO source/target context.
                        let ctx = AccessContext::new(self.cur_pc, key);
                        *self.target_reads.entry(ctx).or_insert(0) += 1;

                        // Address channel: did a prior MMIO value compute this address?
                        let addr_tag = self.taint_in(addr_input);
                        for src in self.contexts_of(&addr_tag) {
                            if src.addr != ctx.addr {
                                self.addr_edges.insert((src, ctx));
                            }
                        }

                        // Control channel: attribute every accumulated gating source.
                        let gating: Vec<(AccessContext, u64)> = self.gating.clone();
                        for (src, bpc) in gating {
                            if src.addr == ctx.addr {
                                continue;
                            }
                            let obs = self.ctrl_obs.entry((src, ctx)).or_default();
                            obs.fires += 1;
                            if self.branch_visits.get(&bpc).copied().unwrap_or(0) >= 2 {
                                obs.is_loop = true;
                            }
                            if self.branch_is_eq.contains(&bpc) {
                                obs.is_eq = true;
                            }
                        }

                        // Capture the loaded value for relation samples.  In production
                        // `LiveEnv` suppresses MMIO reads (returns `None`) to avoid consuming
                        // a fuzz byte, so defer to the next block entry where the register
                        // holds the value; under the mock environment the value is available
                        // immediately.
                        let loaded = addr.and_then(|a| env.read_mem(a, size));
                        if let Some(v) = loaded {
                            self.source_last_val.insert(ctx, v);
                        } else if stmt.output.id > 0 {
                            self.pending_src_vals.push((ctx, stmt.output.id));
                        }

                        // Inject the source taint onto the destination register.
                        let tag = self.shadow.source_tag(ctx);
                        self.set_def(stmt.output, loaded, tag);
                    } else {
                        // Ordinary (non-MMIO) load: dst taint = shadow over the loaded bytes.
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
                        let shifted = v >> (offset as u64 * 8);
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
                    self.set_def(stmt.output, c, ta.union(&tb));
                }

                // Equality comparisons: taint propagates AND the output is marked as
                // equality-derived for discriminant-capture gating at block exit.
                Op::IntEqual | Op::IntNotEqual => {
                    let ta = self.taint_in(stmt.inputs.first());
                    let tb = self.taint_in(stmt.inputs.second());
                    self.set_def(stmt.output, None, ta.union(&tb));
                    if !stmt.output.is_invalid() {
                        self.def_eq_derived.insert(stmt.output.id);
                    }
                }

                // Other binary/comparison ops: taint propagates, concrete is dropped.
                Op::IntDiv | Op::IntSignedDiv | Op::IntRem | Op::IntSignedRem
                | Op::IntRotateLeft | Op::IntRotateRight
                | Op::IntLess | Op::IntSignedLess | Op::IntLessEqual | Op::IntSignedLessEqual
                | Op::IntCarry | Op::IntSignedCarry | Op::IntSignedBorrow | Op::BoolAnd
                | Op::BoolOr | Op::BoolXor => {
                    let ta = self.taint_in(stmt.inputs.first());
                    let tb = self.taint_in(stmt.inputs.second());
                    self.set_def(stmt.output, None, ta.union(&tb));
                }

                Op::IntNot | Op::IntNegate | Op::IntCountOnes | Op::IntCountLeadingZeroes => {
                    let t = self.taint_in(stmt.inputs.first());
                    self.set_def(stmt.output, None, t);
                }

                // BoolNot of an equality-derived condition is still equality-derived
                // (it is just the negation of the same equality test).
                Op::BoolNot => {
                    let t = self.taint_in(stmt.inputs.first());
                    self.set_def(stmt.output, None, t);
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

                // Calls: conservative kill of r0–r3 (accept under-tainting; see plan Fix 4).
                Op::PcodeOp(_) => {
                    self.shadow.kill_call_regs();
                }

                // Unmodeled ops define their output as untainted/unknown (conservative).
                _ => {
                    if !stmt.output.is_invalid() {
                        self.set_def(stmt.output, None, TaintTag::Clean);
                    }
                }
            }
        }

        // Control dependence: record a tainted branch condition as a gating source.
        if let Some(Value::Var(cond)) = block.exit.cond() {
            if !cond.is_invalid() {
                let cond_tag = self.taint_in(Value::Var(cond));
                if !cond_tag.is_clean() {
                    let bpc = self.cur_pc;
                    *self.branch_visits.entry(bpc).or_insert(0) += 1;
                    // Track whether this branch condition was derived from an equality
                    // comparison so MMIO reads that reach here can be flagged is_eq.
                    if self.def_eq_derived.contains(&cond.id) {
                        self.branch_is_eq.insert(bpc);
                    }
                    for src in self.contexts_of(&cond_tag) {
                        let key = (src, bpc);
                        if self.gating_set.insert(key) {
                            self.gating.push(key);
                        }
                    }
                }
            }
        }
    }

    /// Collapse per-pass observations into a [`PassResult`].
    pub fn finish_pass(&self) -> PassResult {
        let address_edges: Vec<_> = self.addr_edges.iter().copied().collect();

        let mut control_edges = Vec::new();
        for (&(src, tgt), obs) in &self.ctrl_obs {
            let count = self.target_reads.get(&tgt).copied().unwrap_or(0);
            // Require ≥3 reads (not ≥2) to classify as Length: a count of exactly 2
            // sits on the threshold and flip-flops between passes, leaking stale
            // relations onto Control edges.  Three reads confirm a genuine loop.
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
        graph.confirm_edge(src, tgt, EdgeKind::Address, None);
    }
    for obs in result.control_edges {
        if obs.is_length {
            let relation = match obs.sample {
                Some(sample) => store.record((obs.source, obs.target), sample),
                None => None,
            };
            graph.confirm_edge(obs.source, obs.target, EdgeKind::Length, relation);
        } else if graph.confirm_edge(obs.source, obs.target, EdgeKind::Control, None) {
            // Capture the gating source value as a discriminant ONLY when:
            //   (a) the Control edge is structurally backed (confirm_edge returned
            //       true) — unbacked observations are taint over-approximation noise;
            //   (b) the condition was derived from an equality comparison (is_eq) —
            //       a value that passes `cmd == 3` is a real discriminant; a value
            //       that passes `size < 256` is an arbitrary data point, not useful.
            if obs.is_eq {
                if let Some((val, _count)) = obs.sample {
                    graph.record_discriminant(obs.source.addr, obs.target.addr, val);
                }
            }
        }
    }
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

// ──────────────────────────────────────────────────────────────────────────────
// In-VM driver: block cache + block-entry hook
// ──────────────────────────────────────────────────────────────────────────────

/// Shared state between the block-caching [`CodeInjector`] and the block-entry hook.
pub struct PhaseBState {
    /// Cached clean P-code blocks keyed by start address (no injected hook ops).
    blocks: HashMap<u64, Block>,
    engine: PhaseBEngine,
    /// MMIO ranges, held here (not only in the engine) so the block-entry hook can
    /// build a `LiveEnv` borrowing this field disjointly from `&mut engine`.
    mmio_ranges: Vec<Range<u64>>,
    /// When false the hook is a no-op (Phase B is sampled, not always-on).
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
}

struct PhaseBInjector {
    state: Rc<RefCell<PhaseBState>>,
    hook: pcode::HookId,
}

impl icicle_vm::CodeInjector for PhaseBInjector {
    fn inject(&mut self, _cpu: &mut Cpu, group: &BlockGroup, code: &mut BlockTable) {
        for id in group.range() {
            let block = &mut code.blocks[id];
            // Cache the *clean* block before inserting the hook op into the live copy.
            self.state.borrow_mut().blocks.insert(block.start, block.clone());
            block.pcode.instructions.insert(0, pcode::Op::Hook(self.hook).into());
            code.modified.insert(id);
        }
    }
}

/// Install the Phase B block cache + hook into the VM.  Must be called *before* the
/// blocks of interest are translated (the injector only runs on newly-lifted blocks).
///
/// Returns the shared state handle used to arm/disarm passes and read results.
pub fn install(vm: &mut Vm, mmio_ranges: Vec<Range<u64>>) -> Rc<RefCell<PhaseBState>> {
    let state = Rc::new(RefCell::new(PhaseBState {
        blocks: HashMap::new(),
        engine: PhaseBEngine::new(HashMap::new(), mmio_ranges.clone()),
        mmio_ranges,
        armed: false,
    }));

    let hook_state = state.clone();
    let hook = vm.cpu.add_hook(move |cpu: &mut Cpu, addr: u64| {
        let mut st = hook_state.borrow_mut();
        if !st.armed {
            return;
        }
        // Field-disjoint borrows: `blocks`, `engine`, and `mmio_ranges` are distinct fields
        // of `st`, so they can be borrowed simultaneously.  The block is cloned (it is small)
        // to release the `blocks` borrow before the `&mut engine` call.
        let st = &mut *st;
        if let Some(block) = st.blocks.get(&addr) {
            let block = block.clone();
            let mut env = LiveEnv { cpu, mmio_ranges: &st.mmio_ranges };
            st.engine.run_block(&block, &mut env);
        }
    });

    vm.add_injector(PhaseBInjector { state: state.clone(), hook });
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream_relation::Confidence;
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

        graph.add_structural_candidates(tgt, &[(src_a, EdgeKind::Control)]);

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
        graph.add_structural_candidates(tgt, &[(src, EdgeKind::Control)]);

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
        assert_eq!(edge.confidence, Confidence::TaintConfirmed);
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
        graph.add_structural_candidates(tgt, &[(src, EdgeKind::Control)]);

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
        graph.add_structural_candidates(tgt, &[(src, EdgeKind::Control)]);

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

    /// An unbacked Control observation records no discriminant.
    #[test]
    fn apply_results_no_discriminant_without_structural_backing() {
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
        assert_eq!(graph.edge_count(), 0, "unbacked observation invents no edge");
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
}
