use hashbrown::HashMap;

use crate::input::StreamKey;

// ──────────────────────────────────────────────────────────────────────────────
// AccessContext
// ──────────────────────────────────────────────────────────────────────────────

/// Fine-grained edge endpoint: the PC at which a stream was accessed together
/// with the stream's MMIO address.
///
/// Streams (fuzzable units) remain address-keyed; edges inside
/// `StreamRelationGraph` are recorded at context granularity so that
/// multiple semantic roles of the same address register (e.g. a status
/// register checked at two different PCs) produce distinct edges.
/// The mutator-facing API (`governors`, `dependents`) projects those
/// context edges back onto address-keyed streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AccessContext {
    pub pc:   u64,
    pub addr: StreamKey,
}

impl AccessContext {
    pub fn new(pc: u64, addr: StreamKey) -> Self {
        Self { pc, addr }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// Value of source stream was used to compute the MMIO address of target.
    Address,
    /// Value of source stream gated a branch condition leading to target.
    Control,
    /// Value of source stream governed the loop trip count / copy size for target.
    Length,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    /// Edge inferred structurally from P-code IR (Phase A).
    Structural,
    /// Edge confirmed by dynamic taint analysis (Phase B).
    TaintConfirmed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelKind {
    Identity,
    Half,
    Linear { k: i64, c: i64 },
}

#[derive(Debug, Clone)]
pub struct Relation {
    pub kind: RelKind,
}

impl Relation {
    /// Evaluate g(val_a) → expected count_b.
    pub fn apply(&self, val_a: u64) -> u64 {
        match self.kind {
            RelKind::Identity => val_a,
            RelKind::Half => val_a / 2,
            RelKind::Linear { k, c } => {
                (val_a as i64).saturating_mul(k).saturating_add(c).max(0) as u64
            }
        }
    }

    /// Inverse: given an expected count_b, return a source value val_a such that
    /// `apply(val_a) ≈ count_b`.  Returns `None` when the relation is degenerate
    /// (k == 0 in Linear) or the result would be negative (no valid val_a exists).
    pub fn invert(&self, count_b: u64) -> Option<u64> {
        match self.kind {
            RelKind::Identity => Some(count_b),
            RelKind::Half => Some(count_b.saturating_mul(2)),
            RelKind::Linear { k, c } => {
                if k == 0 {
                    return None;
                }
                // val_a = (count_b - c) / k; use Euclidean division for correct rounding.
                let numerator = (count_b as i64).saturating_sub(c);
                let result = numerator.div_euclid(k);
                if result < 0 { None } else { Some(result as u64) }
            }
        }
    }

    /// Fit a relation from (val_a, count_b) sample pairs.
    pub fn fit(pairs: &[(u64, u64)]) -> Option<Self> {
        if pairs.is_empty() {
            return None;
        }
        if pairs.iter().all(|(a, b)| a == b) {
            return Some(Relation { kind: RelKind::Identity });
        }
        if pairs.iter().all(|(a, b)| *b == a / 2) {
            return Some(Relation { kind: RelKind::Half });
        }
        // Simple integer linear fit: try to find k, c such that b = k*a + c.
        // Use the first two distinct points.
        let distinct: Vec<_> = {
            let mut v: Vec<(u64, u64)> = pairs.to_vec();
            v.dedup_by_key(|x| x.0);
            v
        };
        if distinct.len() >= 2 {
            let (a1, b1) = (distinct[0].0 as i64, distinct[0].1 as i64);
            let (a2, b2) = (distinct[1].0 as i64, distinct[1].1 as i64);
            let da = a2 - a1;
            let db = b2 - b1;
            if da != 0 && db % da == 0 {
                let k = db / da;
                let c = b1 - k * a1;
                // Verify against all samples.
                let fits = pairs.iter().all(|(a, b)| {
                    (*a as i64).saturating_mul(k).saturating_add(c) == *b as i64
                });
                if fits {
                    return Some(Relation { kind: RelKind::Linear { k, c } });
                }
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct StreamEdge {
    /// Access context of the governing stream (source of the dependency).
    pub source:     AccessContext,
    /// Access context of the governed stream (target of the dependency).
    pub target:     AccessContext,
    pub kind:       EdgeKind,
    pub confidence: Confidence,
    /// Populated for confirmed Length edges: maps val_A → expected count_B.
    pub relation:   Option<Relation>,
    /// Discriminating source values for a `Control` edge: concrete values of the
    /// source stream observed (in Phase B) to open the path to the target.  The
    /// mutator injects these directly so it does not have to rediscover the gating
    /// value by random search — the key to unlocking command-dispatch peripherals.
    pub value_set:  Vec<u64>,
}

/// Directed graph of structural and taint-confirmed relationships between MMIO streams.
#[derive(Default)]
pub struct StreamRelationGraph {
    pub edges:    Vec<StreamEdge>,
    outgoing: HashMap<StreamKey, Vec<usize>>,
    incoming: HashMap<StreamKey, Vec<usize>>,
    /// When false, `mutation_weight_factor` always returns 1.0 so the graph is still
    /// extracted and dumped but provides no mutation bias (ablation Arm B).
    assist_enabled: bool,
}

impl StreamRelationGraph {
    pub fn new() -> Self {
        Self { assist_enabled: true, ..Default::default() }
    }

    /// Enable/disable the mutation-assistance bias at runtime (extraction is unaffected).
    pub fn set_assist_enabled(&mut self, enabled: bool) {
        self.assist_enabled = enabled;
    }

    /// Whether mutation assistance (weighting + coupled mutation) is active.
    pub fn assist_enabled(&self) -> bool {
        self.assist_enabled
    }

    /// Add structurally-inferred candidate edges (Phase A output).
    ///
    /// Each source carries its `(PC, address)` context and a structurally-inferred
    /// `EdgeKind` (`Address` or `Control`).  The index maps are keyed by address so the
    /// mutator-facing `governors`/`dependents` projection stays O(1).
    pub fn add_structural_candidates(
        &mut self,
        target: AccessContext,
        sources: &[(AccessContext, EdgeKind)],
    ) {
        for &(source, kind) in sources {
            if source.addr == target.addr || self.has_edge(source, target) {
                continue;
            }
            let idx = self.edges.len();
            self.edges.push(StreamEdge {
                source,
                target,
                kind,
                confidence: Confidence::Structural,
                relation: None,
                value_set: Vec::new(),
            });
            self.outgoing.entry(source.addr).or_default().push(idx);
            self.incoming.entry(target.addr).or_default().push(idx);
        }
    }

    /// Confirm or reclassify an edge from taint analysis (Phase B output).
    ///
    /// **Upgrade-only — Phase B never invents relationships.**  Phase A already
    /// over-approximates dependence (a backward demand slice reaches *every* MMIO
    /// read that could feed B's address or any gating branch on a path to B), so
    /// every genuine dependency is already present as a `Structural` candidate.
    /// Phase B's role is purely to *confirm* (raise confidence) and *reclassify*
    /// (e.g. `Control` → `Length`) those candidates from observed runtime taint.
    ///
    /// Without this restriction the dynamic gating accumulator — which attributes
    /// every source that ever taints a branch condition to every later MMIO read —
    /// would materialise an (almost) complete bipartite graph of spurious edges.
    /// An observation with no structural backing (no candidate with the same
    /// `source.addr → target.addr`) is therefore a taint over-approximation
    /// artifact and is dropped.
    ///
    /// Returns `true` if at least one structural candidate was upgraded.
    pub fn confirm_edge(
        &mut self,
        source: AccessContext,
        target: AccessContext,
        kind: EdgeKind,
        relation: Option<Relation>,
    ) -> bool {
        let Some(indices) = self.incoming.get(&target.addr).cloned() else {
            return false;
        };

        // Prefer an exact context match (same PCs on both endpoints): upgrade only
        // that one edge so distinct semantic roles of an address stay distinct.
        for &idx in &indices {
            let edge = &mut self.edges[idx];
            if edge.source == source && edge.target == target {
                edge.kind = kind;
                edge.confidence = Confidence::TaintConfirmed;
                if relation.is_some() {
                    edge.relation = relation;
                }
                return true;
            }
        }

        // Otherwise fall back to an address-pair match: Phase A and Phase B may
        // attribute an access to different PCs (e.g. the slice anchor vs. the load
        // marker), but the stream-level relationship is the same.  Upgrade every
        // structurally-backed candidate for this address pair.
        let mut upgraded = false;
        for &idx in &indices {
            let edge = &mut self.edges[idx];
            if edge.source.addr == source.addr {
                edge.kind = kind;
                edge.confidence = Confidence::TaintConfirmed;
                if relation.is_some() {
                    edge.relation = relation.clone();
                }
                upgraded = true;
            }
        }
        upgraded
    }

    /// Record a discriminating source value on confirmed `Control` edge(s)
    /// `source → target`.
    ///
    /// `value` is a concrete value of the source stream that Phase B observed to
    /// open the path to the target.  Accumulating these across taint passes builds
    /// the set of known-gating values the mutator can inject directly.  The set is
    /// bounded so a noisy source (e.g. a counter feeding a comparison) cannot grow
    /// it without limit.  Only `Control` edges carry discriminants: `Length` edges
    /// already model the value→count mapping via `relation`, and `Address` edges
    /// consume the value as an address rather than a discriminator.
    pub fn record_discriminant(&mut self, source: StreamKey, target: StreamKey, value: u64) {
        const MAX_DISCRIMINANTS: usize = 16;
        let Some(indices) = self.incoming.get(&target).cloned() else {
            return;
        };
        for idx in indices {
            let edge = &mut self.edges[idx];
            if edge.source.addr == source
                && edge.kind == EdgeKind::Control
                && edge.value_set.len() < MAX_DISCRIMINANTS
                && !edge.value_set.contains(&value)
            {
                edge.value_set.push(value);
            }
        }
    }

    /// True if an edge with this exact context pair already exists.
    pub fn has_edge(&self, source: AccessContext, target: AccessContext) -> bool {
        self.incoming
            .get(&target.addr)
            .map_or(false, |v| v.iter().any(|&i| self.edges[i].source == source))
    }

    /// All edges whose target address is `target` (address-keyed projection for mutator).
    pub fn governors(&self, target: StreamKey) -> impl Iterator<Item = &StreamEdge> {
        let indices = self.incoming.get(&target).map(|v| v.as_slice()).unwrap_or(&[]);
        indices.iter().map(|&i| &self.edges[i])
    }

    /// All edges whose source address is `source` (address-keyed projection for mutator).
    pub fn dependents(&self, source: StreamKey) -> impl Iterator<Item = &StreamEdge> {
        let indices = self.outgoing.get(&source).map(|v| v.as_slice()).unwrap_or(&[]);
        indices.iter().map(|&i| &self.edges[i])
    }

    pub fn confirmed_edges(&self) -> impl Iterator<Item = &StreamEdge> {
        self.edges.iter().filter(|e| e.confidence == Confidence::TaintConfirmed)
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Mutation-assistance hook: how much more often a stream at `addr` should be mutated
    /// because it participates in inter-stream dependencies.  Returns a multiplier ≥ 1.0.
    ///
    /// Value-flow edges (Address/Length) are robust to address-keying and get the
    /// strongest boost; Control edges are coarser (the same status register may be checked
    /// at several sites) and get a smaller boost.  A `TaintConfirmed` edge is weighted above
    /// a merely `Structural` one.
    pub fn mutation_weight_factor(&self, addr: StreamKey) -> f64 {
        if !self.assist_enabled {
            return 1.0;
        }
        let mut factor = 1.0_f64;
        for list in [self.incoming.get(&addr), self.outgoing.get(&addr)].into_iter().flatten() {
            for &idx in list {
                let edge = &self.edges[idx];
                let base = match edge.kind {
                    EdgeKind::Address | EdgeKind::Length => 4.0,
                    EdgeKind::Control => 2.0,
                };
                let confirmed_bonus = match edge.confidence {
                    Confidence::TaintConfirmed => 1.5,
                    Confidence::Structural => 1.0,
                };
                factor = factor.max(base * confirmed_bonus);
            }
        }
        factor
    }

    #[cfg(test)]
    fn confidence_of(&self, source: StreamKey, target: StreamKey) -> Option<Confidence> {
        self.incoming.get(&target).and_then(|v| {
            v.iter().map(|&i| &self.edges[i]).find(|e| e.source.addr == source).map(|e| e.confidence)
        })
    }

    /// Serialize the graph as a JSON array of edges (for offline evaluation / debugging).
    pub fn to_json(&self) -> String {
        let mut s = String::from("[\n");
        for (i, e) in self.edges.iter().enumerate() {
            if i > 0 {
                s.push_str(",\n");
            }
            let relation = match &e.relation {
                Some(r) => format!("{:?}", r.kind),
                None => "none".to_string(),
            };
            let values = e
                .value_set
                .iter()
                .map(|v| format!("\"{v:#x}\""))
                .collect::<Vec<_>>()
                .join(",");
            s.push_str(&format!(
                "  {{\"source_pc\":\"{:#x}\",\"source_addr\":\"{:#x}\",\
                 \"target_pc\":\"{:#x}\",\"target_addr\":\"{:#x}\",\
                 \"kind\":\"{:?}\",\"confidence\":\"{:?}\",\"relation\":\"{}\",\
                 \"values\":[{}]}}",
                e.source.pc, e.source.addr, e.target.pc, e.target.addr,
                e.kind, e.confidence, relation, values
            ));
        }
        s.push_str("\n]\n");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(pc: u64, addr: u64) -> AccessContext {
        AccessContext::new(pc, addr)
    }

    /// apply(invert(x)) must round-trip for all valid relation kinds.
    #[test]
    fn relation_invert_round_trips() {
        // Identity: invert is the identity.
        let r = Relation { kind: RelKind::Identity };
        assert_eq!(r.invert(0), Some(0));
        assert_eq!(r.invert(5), Some(5));
        assert_eq!(r.apply(r.invert(99).unwrap()), 99);

        // Half: invert doubles.
        let r = Relation { kind: RelKind::Half };
        assert_eq!(r.invert(4), Some(8));
        assert_eq!(r.apply(8), 4);
        assert_eq!(r.invert(0), Some(0));

        // Linear{k=2, c=1}: count = 2*val + 1 → val = (count − 1) / 2.
        let r = Relation { kind: RelKind::Linear { k: 2, c: 1 } };
        assert_eq!(r.invert(5), Some(2));   // (5−1)/2 = 2
        assert_eq!(r.apply(2), 5);          // 2*2+1 = 5
        assert_eq!(r.invert(1), Some(0));   // boundary: val = 0

        // Linear with k=0 → degenerate, returns None.
        let r = Relation { kind: RelKind::Linear { k: 0, c: 5 } };
        assert_eq!(r.invert(5), None);

        // Linear with k=1, c=0 → same as identity.
        let r = Relation { kind: RelKind::Linear { k: 1, c: 0 } };
        assert_eq!(r.invert(7), Some(7));
        assert_eq!(r.apply(r.invert(42).unwrap()), 42);

        // Negative c: count = val + (-3) → val = count + 3.
        let r = Relation { kind: RelKind::Linear { k: 1, c: -3 } };
        assert_eq!(r.invert(0), Some(3));
        assert_eq!(r.apply(3), 0);

        // Result would be negative → None.
        let r = Relation { kind: RelKind::Linear { k: 1, c: 10 } };
        assert_eq!(r.invert(5), None); // (5 - 10) / 1 = -5 < 0
    }

    /// Phase B must NOT invent edges: an observation with no structural backing is
    /// dropped, so the confirmed graph can never exceed Phase A's candidate set.
    /// This is the regression guard against the near-complete bipartite blow-up.
    #[test]
    fn confirm_without_structural_backing_is_dropped() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);

        let upgraded = g.confirm_edge(a, b, EdgeKind::Control, None);
        assert!(!upgraded, "unbacked confirmation must report no upgrade");
        assert_eq!(g.edge_count(), 0, "Phase B must not invent an edge from nothing");
    }

    /// A structural candidate is upgraded in place (no new row) and reclassified.
    #[test]
    fn confirm_upgrades_structural_candidate_in_place() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);
        g.add_structural_candidates(b, &[(a, EdgeKind::Control)]);
        assert_eq!(g.confidence_of(0x5800_0000, 0x5800_0004), Some(Confidence::Structural));

        let upgraded = g.confirm_edge(a, b, EdgeKind::Length, None);
        assert!(upgraded);
        assert_eq!(g.edge_count(), 1, "confirmation upgrades in place, never adds a row");
        assert_eq!(g.confidence_of(0x5800_0000, 0x5800_0004), Some(Confidence::TaintConfirmed));
        assert_eq!(g.governors(0x5800_0004).next().unwrap().kind, EdgeKind::Length);
    }

    /// Discriminating values accumulate on a confirmed Control edge, dedup, stay
    /// bounded, and are never attached to a non-Control edge.
    #[test]
    fn record_discriminant_accumulates_bounded_unique_values() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);
        g.add_structural_candidates(b, &[(a, EdgeKind::Control)]);
        g.confirm_edge(a, b, EdgeKind::Control, None);

        // Distinct values accumulate; duplicates are ignored.
        g.record_discriminant(a.addr, b.addr, 3);
        g.record_discriminant(a.addr, b.addr, 3);
        g.record_discriminant(a.addr, b.addr, 7);
        let edge = g.governors(b.addr).next().unwrap();
        assert_eq!(edge.value_set, vec![3, 7], "dedup distinct gating values");

        // The set is bounded (≤16): pushing many distinct values cannot grow it past the cap.
        for v in 100..200 {
            g.record_discriminant(a.addr, b.addr, v);
        }
        let edge = g.governors(b.addr).next().unwrap();
        assert!(edge.value_set.len() <= 16, "discriminant set must stay bounded");
    }

    /// A discriminant is only attached to `Control` edges — never `Length`/`Address`.
    #[test]
    fn record_discriminant_ignores_non_control_edges() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);
        g.add_structural_candidates(b, &[(a, EdgeKind::Control)]);
        g.confirm_edge(a, b, EdgeKind::Length, Some(Relation { kind: RelKind::Identity }));

        g.record_discriminant(a.addr, b.addr, 42);
        let edge = g.governors(b.addr).next().unwrap();
        assert!(edge.value_set.is_empty(), "Length edge must not receive discriminants");
    }

    /// PC mismatch between Phase A and Phase B still confirms via the address pair,
    /// and the confirmed-edge count stays bounded by the structural candidate count.
    #[test]
    fn confirm_matches_by_address_pair_when_pc_differs() {
        let mut g = StreamRelationGraph::new();
        let a_struct = ctx(0x10, 0x5800_0000);
        let b_struct = ctx(0x20, 0x5800_0004);
        g.add_structural_candidates(b_struct, &[(a_struct, EdgeKind::Control)]);

        // Phase B observed the same streams at different PCs.
        let a_obs = ctx(0x99, 0x5800_0000);
        let b_obs = ctx(0xaa, 0x5800_0004);
        let upgraded = g.confirm_edge(a_obs, b_obs, EdgeKind::Control, None);

        assert!(upgraded);
        assert_eq!(g.edge_count(), 1, "no duplicate row for the PC-shifted observation");
        assert_eq!(g.confidence_of(0x5800_0000, 0x5800_0004), Some(Confidence::TaintConfirmed));
    }
}
