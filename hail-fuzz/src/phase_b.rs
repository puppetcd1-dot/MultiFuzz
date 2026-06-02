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
use pcode::{Op, Value, VarId};

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
    /// The gating branch was a runtime loop and B was read ≥2 times → `Length`.
    pub is_length: bool,
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
#[derive(Default)]
pub struct LengthSampleStore {
    samples: HashMap<(AccessContext, AccessContext), Vec<(u64, u64)>>,
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
        let v = self.samples.entry(edge).or_default();
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
    cur_pc: u64,

    // ── per-pass observations ──
    /// How many times each MMIO context was read this pass.
    target_reads: HashMap<AccessContext, u64>,
    /// Last concrete value loaded at each MMIO context (for relation samples).
    source_last_val: HashMap<AccessContext, u64>,
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
            cur_pc: 0,
            target_reads: HashMap::new(),
            source_last_val: HashMap::new(),
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
        self.branch_visits.clear();
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

    // ── block interpretation ───────────────────────────────────────────────────

    /// Interpret a single basic block, updating shadow state and observations.
    pub fn run_block(&mut self, block: &Block, env: &mut dyn ConcreteEnv) {
        // Temporaries do not survive block boundaries; register defs are re-seeded
        // from `env` (the live block-entry state) on first use within the block.
        self.def_concrete.clear();
        self.def_taint.clear();
        self.cur_pc = block.start;

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
                        }

                        // Capture the loaded value for relation samples (non-MMIO read only;
                        // LiveEnv suppresses MMIO reads, so this is `None` in production but
                        // available under the mock environment used in tests).
                        let loaded = addr.and_then(|a| env.read_mem(a, size));
                        if let Some(v) = loaded {
                            self.source_last_val.insert(ctx, v);
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

                // Other binary/comparison ops: taint propagates, concrete is dropped
                // (their result is not used to compute memory addresses we model).
                Op::IntDiv | Op::IntSignedDiv | Op::IntRem | Op::IntSignedRem
                | Op::IntRotateLeft | Op::IntRotateRight | Op::IntEqual | Op::IntNotEqual
                | Op::IntLess | Op::IntSignedLess | Op::IntLessEqual | Op::IntSignedLessEqual
                | Op::IntCarry | Op::IntSignedCarry | Op::IntSignedBorrow | Op::BoolAnd
                | Op::BoolOr | Op::BoolXor => {
                    let ta = self.taint_in(stmt.inputs.first());
                    let tb = self.taint_in(stmt.inputs.second());
                    self.set_def(stmt.output, None, ta.union(&tb));
                }

                Op::IntNot | Op::IntNegate | Op::IntCountOnes | Op::IntCountLeadingZeroes
                | Op::BoolNot => {
                    let t = self.taint_in(stmt.inputs.first());
                    self.set_def(stmt.output, None, t);
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
            let is_length = obs.is_loop && count >= 2 && obs.fires >= 2;
            let sample = self.source_last_val.get(&src).map(|&v| (v, count));
            control_edges.push(ControlEdgeObs { source: src, target: tgt, is_length, sample });
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
        } else {
            graph.confirm_edge(obs.source, obs.target, EdgeKind::Control, None);
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

    /// Mock concrete environment: a register map + a flat memory map.
    struct MockEnv {
        regs: HashMap<i16, u64>,
        mem: HashMap<u64, u64>, // addr → value (size-agnostic for tests)
    }
    impl MockEnv {
        fn new() -> Self {
            Self { regs: HashMap::new(), mem: HashMap::new() }
        }
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
        Block {
            pcode,
            entry: None,
            start,
            end,
            context: 0,
            exit,
            breakpoints: 0,
            num_instructions: 0,
        }
    }

    fn marker(pc: u64) -> pcode::Instruction {
        (pcode::Op::InstructionMarker, pcode::Value::Const(pc, 8)).into()
    }
    fn reg(id: i16) -> VarNode {
        VarNode::new(id, 4)
    }

    fn engine() -> PhaseBEngine {
        PhaseBEngine::new(HashMap::new(), vec![0x4000_0000..0x6000_0000])
    }

    /// An MMIO value used to compute B's load address must yield an `Address` edge.
    #[test]
    fn address_edge_confirmed_from_tainted_load_address() {
        let mut e = engine();
        e.read_sites.insert(0x100, 0x5800_0000); // source A
        e.read_sites.insert(0x108, 0x5800_0004); // target B

        // 0x100: r10 = LOAD[r5]        (MMIO A → r10 tainted by A)
        // 0x104: r7  = ADD r4, r10     (B's address = base + A)
        // 0x108: r1  = LOAD[r7]        (MMIO B; address tainted by A)
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
        assert!(
            result.address_edges.contains(&(src, tgt)),
            "expected Address edge A→B, got {:?}",
            result.address_edges
        );
    }

    /// A non-looping tainted branch gating B yields a `Control` edge (not `Length`).
    #[test]
    fn control_edge_confirmed_from_tainted_branch() {
        let mut e = engine();
        e.read_sites.insert(0x200, 0x5800_0008); // source A (gates branch)
        e.read_sites.insert(0x300, 0x5800_0000); // target B

        // Block 1 @0x200: r1 = LOAD[r9]; r2 = EQ(r1, #3); branch on r2.
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

        // Block 2 @0x300: r5 = LOAD[r8]  (B)
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
        assert!(edge.is_some(), "expected Control observation A→B, got {:?}", result.control_edges);
        assert!(!edge.unwrap().is_length, "single activation must NOT be Length");
    }

    /// A tainted loop branch under which B is read multiple times yields `Length`.
    #[test]
    fn length_edge_from_looping_tainted_branch() {
        let mut e = engine();
        e.read_sites.insert(0x200, 0x5800_0008); // source A (loop bound)
        e.read_sites.insert(0x300, 0x5800_0000); // target B (read each iteration)

        // Loop header @0x200 (tainted branch). Re-evaluated each iteration → revisited.
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

        // Loop body @0x300: reads B.
        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);
        env.regs.insert(3, 0);

        // Three iterations: header, body, header, body, header, body.
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
        assert!(edge.is_length, "looping tainted branch with ≥2 reads must be Length");
    }

    /// `apply_pass_result` upgrades a structural Control candidate to a taint-confirmed
    /// Address edge, and fits a Length relation across repeated samples.
    #[test]
    fn apply_results_confirms_and_fits() {
        let mut graph = StreamRelationGraph::new();
        let mut store = LengthSampleStore::new();

        let src = AccessContext::new(0x200, 0x5800_0008);
        let tgt = AccessContext::new(0x300, 0x5800_0000);

        // Two passes with identity samples (count == value) → Identity relation fit.
        for v in [4u64, 7u64] {
            let result = PassResult {
                address_edges: vec![],
                control_edges: vec![ControlEdgeObs {
                    source: src,
                    target: tgt,
                    is_length: true,
                    sample: Some((v, v)),
                }],
            };
            apply_pass_result(&mut graph, &mut store, result);
        }

        let edge = graph
            .confirmed_edges()
            .find(|e| e.source == src && e.target == tgt)
            .expect("edge should be confirmed");
        assert_eq!(edge.kind, EdgeKind::Length);
        assert_eq!(edge.confidence, Confidence::TaintConfirmed);
        assert!(edge.relation.is_some(), "a relation should be fitted from the samples");
    }

    /// A call (PcodeOp) clears r0–r3 taint so it cannot leak across an opaque call.
    #[test]
    fn call_kills_argument_register_taint() {
        let mut e = engine();
        e.read_sites.insert(0x100, 0x5800_0000);

        // r0 = LOAD[mmio] (tainted); CALL; then r0 must be clean.
        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(0), Op::Load(0), reg(5)));
        p.push(marker(0x104));
        p.push((VarNode::NONE, Op::PcodeOp(0), reg(0)));
        let block = lifter_block(p, 0x100, 0x108, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        e.run_block(&block, &mut env);

        assert!(
            e.shadow.reg_tag(0).is_clean(),
            "r0 taint must be killed after an opaque call"
        );
    }
}
