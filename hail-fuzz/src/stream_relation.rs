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
                // k==0 means count_B is constant regardless of val_A — independence,
                // not a controllable length relation.  Reject it so these edges stay
                // Control rather than gaining a spurious Length classification.
                if k == 0 {
                    return None;
                }
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
    /// Discriminating source values for a `Control` edge: `(value, hit_count)` pairs
    /// observed in Phase B to open the path to the target.  Sorted by hit_count
    /// descending so the most-confirmed discriminants are injected first.
    /// Capped at `MAX_DISCRIMINANTS` entries; singletons are evicted to make room
    /// for recurring values — high hit_count entries are immune to eviction.
    pub value_set:  Vec<(u64, u32)>,
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
                // Clear fields belonging to the previous kind so stale data
                // (e.g. a relation from a prior Length phase, or discriminants
                // from a prior Control phase) cannot persist on the new kind.
                if kind != EdgeKind::Length {
                    edge.relation = None;
                } else if relation.is_some() {
                    edge.relation = relation;
                }
                if kind != EdgeKind::Control {
                    edge.value_set.clear();
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
                if kind != EdgeKind::Length {
                    edge.relation = None;
                } else if relation.is_some() {
                    edge.relation = relation.clone();
                }
                if kind != EdgeKind::Control {
                    edge.value_set.clear();
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
    /// Record a discriminating source value on confirmed `Control` edge(s)
    /// `source → target`.
    ///
    /// Uses frequency-based eviction: each (value, hit_count) pair is maintained
    /// sorted by recurrence.  When the cap is reached a singleton (hit_count=1) is
    /// evicted to make room for the new value; if all slots are multi-hit (genuine
    /// recurring discriminants) the new value is discarded rather than displacing a
    /// confirmed value.  This prevents one-time random fuzz bytes from permanently
    /// blocking genuine command-code discriminants that appear repeatedly.
    pub fn record_discriminant(&mut self, source: StreamKey, target: StreamKey, value: u64) {
        const MAX_DISCRIMINANTS: usize = 16;
        let Some(indices) = self.incoming.get(&target).cloned() else {
            return;
        };
        for idx in indices {
            let edge = &mut self.edges[idx];
            if edge.source.addr != source || edge.kind != EdgeKind::Control {
                continue;
            }
            // Increment hit count if already present.
            if let Some(entry) = edge.value_set.iter_mut().find(|(v, _)| *v == value) {
                entry.1 += 1;
                continue;
            }
            // New value: insert directly if under cap.
            if edge.value_set.len() < MAX_DISCRIMINANTS {
                edge.value_set.push((value, 1));
                continue;
            }
            // Cap full: evict the first singleton (hit_count==1) to make room.
            // If all entries are multi-hit, the new (unconfirmed) value is discarded.
            if let Some(pos) = edge.value_set.iter().position(|(_, c)| *c == 1) {
                edge.value_set[pos] = (value, 1);
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
            // Sort by hit_count descending: most-confirmed discriminants appear first
            // in the JSON so offline analysis sees the highest-confidence values.
            let values = {
                let mut sorted = e.value_set.clone();
                sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                sorted.iter().map(|(v, _)| format!("\"{v:#x}\"")).collect::<Vec<_>>().join(",")
            };
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
        let r = Relation { kind: RelKind::Identity };
        assert_eq!(r.invert(0), Some(0));
        assert_eq!(r.invert(5), Some(5));
        assert_eq!(r.apply(r.invert(99).unwrap()), 99);

        let r = Relation { kind: RelKind::Half };
        assert_eq!(r.invert(4), Some(8));
        assert_eq!(r.apply(8), 4);
        assert_eq!(r.invert(0), Some(0));

        let r = Relation { kind: RelKind::Linear { k: 2, c: 1 } };
        assert_eq!(r.invert(5), Some(2));
        assert_eq!(r.apply(2), 5);
        assert_eq!(r.invert(1), Some(0));

        let r = Relation { kind: RelKind::Linear { k: 0, c: 5 } };
        assert_eq!(r.invert(5), None);

        let r = Relation { kind: RelKind::Linear { k: 1, c: 0 } };
        assert_eq!(r.invert(7), Some(7));
        assert_eq!(r.apply(r.invert(42).unwrap()), 42);

        let r = Relation { kind: RelKind::Linear { k: 1, c: -3 } };
        assert_eq!(r.invert(0), Some(3));
        assert_eq!(r.apply(3), 0);

        let r = Relation { kind: RelKind::Linear { k: 1, c: 10 } };
        assert_eq!(r.invert(5), None);
    }

    /// Fix 1: Relation::fit must return None when all samples have the same count_B
    /// (k==0 means independence, not a controllable length).
    #[test]
    fn fit_rejects_zero_slope_constant_count() {
        // All count_B values equal 2 regardless of val_A → k=0, degenerate.
        let pairs = vec![(1u64, 2u64), (3, 2), (7, 2), (10, 2)];
        assert!(
            Relation::fit(&pairs).is_none(),
            "constant count_B must not produce a Linear{{k:0}} relation"
        );
        // Single pair: degenerate by definition (cannot distinguish constant from slope).
        assert!(Relation::fit(&[(5, 3)]).is_none());
    }

    /// Fix 1: Non-zero slopes still fit correctly after the k==0 guard.
    #[test]
    fn fit_accepts_nonzero_slope() {
        // count = -1 * val + 4  (the DMA pattern from the real relations.json)
        let pairs = vec![(1u64, 3u64), (2, 2), (3, 1), (0, 4)];
        let rel = Relation::fit(&pairs).expect("should fit Linear{k:-1,c:4}");
        assert!(matches!(rel.kind, RelKind::Linear { k: -1, c: 4 }));
    }

    /// Fix 2: Reclassifying an edge clears stale fields from the previous kind.
    #[test]
    fn reclassification_clears_stale_relation_and_value_set() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);

        // Start as Control; add a discriminant.
        g.add_structural_candidates(b, &[(a, EdgeKind::Control)]);
        g.confirm_edge(a, b, EdgeKind::Control, None);
        g.record_discriminant(a.addr, b.addr, 0x3);
        assert!(!g.governors(b.addr).next().unwrap().value_set.is_empty());

        // Reclassify to Length: value_set must be cleared.
        g.confirm_edge(a, b, EdgeKind::Length, Some(Relation { kind: RelKind::Identity }));
        let edge = g.governors(b.addr).next().unwrap();
        assert_eq!(edge.kind, EdgeKind::Length);
        assert!(edge.value_set.is_empty(), "reclassify→Length must clear value_set");
        assert!(edge.relation.is_some(), "relation should be set after Length confirmation");

        // Reclassify back to Control: relation must be cleared.
        g.confirm_edge(a, b, EdgeKind::Control, None);
        let edge = g.governors(b.addr).next().unwrap();
        assert_eq!(edge.kind, EdgeKind::Control);
        assert!(edge.relation.is_none(), "reclassify→Control must clear relation");
    }

    /// Phase B must NOT invent edges: an observation with no structural backing is
    /// dropped, so the confirmed graph can never exceed Phase A's candidate set.
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

    /// Fix 4: Discriminating values accumulate with hit counts, dedup increments
    /// the existing count, and the set stays bounded at MAX_DISCRIMINANTS.
    #[test]
    fn record_discriminant_accumulates_with_hit_counts() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);
        g.add_structural_candidates(b, &[(a, EdgeKind::Control)]);
        g.confirm_edge(a, b, EdgeKind::Control, None);

        g.record_discriminant(a.addr, b.addr, 3);
        g.record_discriminant(a.addr, b.addr, 3); // duplicate → hit_count=2
        g.record_discriminant(a.addr, b.addr, 7);
        let edge = g.governors(b.addr).next().unwrap();
        let vals: Vec<u64> = edge.value_set.iter().map(|&(v, _)| v).collect();
        assert_eq!(vals, vec![3, 7], "distinct values recorded");
        assert_eq!(
            edge.value_set.iter().find(|&&(v, _)| v == 3).map(|&(_, c)| c),
            Some(2),
            "duplicate increments hit_count"
        );
        assert_eq!(
            edge.value_set.iter().find(|&&(v, _)| v == 7).map(|&(_, c)| c),
            Some(1)
        );

        // Flood: cap must not be exceeded.
        for v in 100u64..200 {
            g.record_discriminant(a.addr, b.addr, v);
        }
        assert!(g.governors(b.addr).next().unwrap().value_set.len() <= 16);
    }

    /// Fix 4: Frequency-based eviction — a singleton is evicted when the cap is
    /// full, but a multi-hit entry survives.
    #[test]
    fn record_discriminant_frequency_eviction_protects_recurring_values() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);
        g.add_structural_candidates(b, &[(a, EdgeKind::Control)]);
        g.confirm_edge(a, b, EdgeKind::Control, None);

        // Fill cap with 16 distinct singletons.
        for v in 0u64..16 {
            g.record_discriminant(a.addr, b.addr, v);
        }
        // Reinforce value 0 → hit_count=2.
        g.record_discriminant(a.addr, b.addr, 0);
        // Push a 17th value: must evict a singleton (hit_count=1), not value 0.
        g.record_discriminant(a.addr, b.addr, 99);

        let edge = g.governors(b.addr).next().unwrap();
        assert_eq!(edge.value_set.len(), 16, "cap must stay at 16 after eviction");
        assert!(
            edge.value_set.iter().any(|&(v, _)| v == 0),
            "multi-hit value must not be evicted"
        );
        assert!(
            edge.value_set.iter().any(|&(v, _)| v == 99),
            "new value must be inserted after evicting a singleton"
        );
        // Value 1 was the first singleton after 0, so it should be evicted.
        assert!(
            !edge.value_set.iter().any(|&(v, _)| v == 1),
            "evicted singleton must be gone"
        );
    }

    /// Fix 4: When all entries are multi-hit, a new singleton is discarded
    /// rather than displacing a confirmed recurring value.
    #[test]
    fn record_discriminant_discards_when_all_entries_are_recurring() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);
        g.add_structural_candidates(b, &[(a, EdgeKind::Control)]);
        g.confirm_edge(a, b, EdgeKind::Control, None);

        // Fill cap, then make all entries multi-hit.
        for v in 0u64..16 {
            g.record_discriminant(a.addr, b.addr, v);
            g.record_discriminant(a.addr, b.addr, v); // hit_count=2 for each
        }
        // Now push a new value: all existing are multi-hit, so discard.
        g.record_discriminant(a.addr, b.addr, 999);
        let edge = g.governors(b.addr).next().unwrap();
        assert!(!edge.value_set.iter().any(|&(v, _)| v == 999), "singleton discarded when all entries recurring");
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

        let a_obs = ctx(0x99, 0x5800_0000);
        let b_obs = ctx(0xaa, 0x5800_0004);
        let upgraded = g.confirm_edge(a_obs, b_obs, EdgeKind::Control, None);

        assert!(upgraded);
        assert_eq!(g.edge_count(), 1, "no duplicate row for the PC-shifted observation");
        assert_eq!(g.confidence_of(0x5800_0000, 0x5800_0004), Some(Confidence::TaintConfirmed));
    }
}
