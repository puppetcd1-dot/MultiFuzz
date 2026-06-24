use hashbrown::HashMap;

use crate::input::StreamKey;

// ──────────────────────────────────────────────────────────────────────────────
// AccessContext
// ──────────────────────────────────────────────────────────────────────────────

/// Edge endpoint: the instruction PC and the stream's MMIO address.
/// Edges are context-granular; the mutator API projects them back to address-keyed streams.
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

impl EdgeKind {
    /// Specificity ranking: Address > Length > Control.  Edges never downgrade.
    pub fn priority(self) -> u8 {
        match self {
            EdgeKind::Address => 3,
            EdgeKind::Length => 2,
            EdgeKind::Control => 1,
        }
    }
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
    pub source:     AccessContext,
    pub target:     AccessContext,
    pub kind:       EdgeKind,
    /// For confirmed Length edges: maps val_A → expected count_B.
    pub relation:   Option<Relation>,
    /// For Control edges: `(value, hit_count)` discriminants, capped at MAX_DISCRIMINANTS.
    pub value_set:  Vec<(u64, u32)>,
}

#[derive(Default)]
pub struct StreamRelationGraph {
    pub edges:    Vec<StreamEdge>,
    outgoing: HashMap<StreamKey, Vec<usize>>,
    incoming: HashMap<StreamKey, Vec<usize>>,
    assist_enabled: bool,
}

impl StreamRelationGraph {
    pub fn new() -> Self {
        Self { assist_enabled: true, ..Default::default() }
    }

    pub fn set_assist_enabled(&mut self, enabled: bool) { self.assist_enabled = enabled; }
    pub fn assist_enabled(&self) -> bool { self.assist_enabled }

    /// Insert a new taint-confirmed edge, or upgrade an existing one.
    ///
    /// - If an edge with the same `(source, target)` exists: upgrade kind if higher
    ///   priority (Address > Length > Control), merge relation (keep existing if new
    ///   is None; update if new is Some), don't downgrade kind.
    /// - If no exact match but same `(source.addr, target.addr)` pair exists: same
    ///   upgrade logic.
    /// - If no edge exists at all: create a new edge and index it.
    pub fn insert_or_confirm_edge(
        &mut self,
        source: AccessContext,
        target: AccessContext,
        kind: EdgeKind,
        relation: Option<Relation>,
    ) {
        // Try exact context match first.
        if let Some(indices) = self.incoming.get(&target.addr).cloned() {
            for &idx in &indices {
                let edge = &self.edges[idx];
                if edge.source == source && edge.target == target {
                    Self::upgrade_edge(&mut self.edges[idx], kind, relation);
                    return;
                }
            }
            // Fall back to address-pair match.
            for &idx in &indices {
                let edge = &self.edges[idx];
                if edge.source.addr == source.addr {
                    Self::upgrade_edge(&mut self.edges[idx], kind, relation);
                    return;
                }
            }
        }
        // No existing edge — create a new one.
        let idx = self.edges.len();
        self.edges.push(StreamEdge {
            source,
            target,
            kind,
            relation,
            value_set: Vec::new(),
        });
        self.outgoing.entry(source.addr).or_default().push(idx);
        self.incoming.entry(target.addr).or_default().push(idx);
    }

    /// Upgrade an existing edge: never downgrade kind, merge relation.
    fn upgrade_edge(edge: &mut StreamEdge, kind: EdgeKind, relation: Option<Relation>) {
        // Never downgrade: Address > Length > Control.
        if kind.priority() >= edge.kind.priority() {
            edge.kind = kind;
            if kind != EdgeKind::Length { edge.relation = None; }
            else if relation.is_some() { edge.relation = relation; }
            if kind != EdgeKind::Control { edge.value_set.clear(); }
        }
    }

    /// Record a discriminant value on Control edges. Frequency-based eviction:
    /// singletons are displaced before multi-hit entries; discards new values when all slots are multi-hit.
    pub fn record_discriminant(&mut self, source: StreamKey, target: StreamKey, value: u64) {
        const MAX_DISCRIMINANTS: usize = 16;
        let Some(indices) = self.incoming.get(&target).cloned() else {
            return;
        };
        for idx in indices {
            let edge = &mut self.edges[idx];
            if edge.source.addr != source || edge.kind != EdgeKind::Control { continue; }
            if let Some(entry) = edge.value_set.iter_mut().find(|(v, _)| *v == value) {
                entry.1 += 1;
                continue;
            }
            if edge.value_set.len() < MAX_DISCRIMINANTS {
                edge.value_set.push((value, 1));
                continue;
            }
            if let Some(pos) = edge.value_set.iter().position(|(_, c)| *c == 1) {
                edge.value_set[pos] = (value, 1);
            }
        }
    }

    /// Propagate fitted relations to all Length edges sharing the same
    /// address pair (sibling edges from earlier passes may lack a relation otherwise).
    pub fn backfill_length_relations(
        &mut self,
        relations: impl Iterator<Item = ((StreamKey, StreamKey), Relation)>,
    ) {
        for ((src_addr, tgt_addr), relation) in relations {
            for edge in &mut self.edges {
                if edge.kind == EdgeKind::Length
                    && edge.source.addr == src_addr
                    && edge.target.addr == tgt_addr
                    && edge.relation.is_none()
                {
                    edge.relation = Some(relation.clone());
                }
            }
        }
    }

    #[cfg(test)]
    pub fn governors(&self, target: StreamKey) -> impl Iterator<Item = &StreamEdge> {
        let indices = self.incoming.get(&target).map(|v| v.as_slice()).unwrap_or(&[]);
        indices.iter().map(|&i| &self.edges[i])
    }

    /// Returns deduplicated `(source_addr, kind)` pairs for all edges targeting
    /// `target_addr`.  When multiple edges share the same `source_addr`, returns
    /// the highest priority kind (Address > Length > Control).
    pub fn get_governors_by_addr(&self, target_addr: StreamKey) -> Vec<(StreamKey, EdgeKind)> {
        let mut best: HashMap<StreamKey, EdgeKind> = HashMap::new();
        let indices = self.incoming.get(&target_addr).map(|v| v.as_slice()).unwrap_or(&[]);
        for &idx in indices {
            let edge = &self.edges[idx];
            let entry = best.entry(edge.source.addr).or_insert(edge.kind);
            if edge.kind.priority() > entry.priority() {
                *entry = edge.kind;
            }
        }
        best.into_iter().collect()
    }

    pub fn dependents(&self, source: StreamKey) -> impl Iterator<Item = &StreamEdge> {
        let indices = self.outgoing.get(&source).map(|v| v.as_slice()).unwrap_or(&[]);
        indices.iter().map(|&i| &self.edges[i])
    }

    /// Returns deduplicated `(target_addr, kind)` pairs for all edges originating
    /// from `source_addr`.  When multiple edges share the same `target_addr`, returns
    /// the highest priority kind (Address > Length > Control).
    pub fn get_dependents_by_addr(&self, source_addr: StreamKey) -> Vec<(StreamKey, EdgeKind)> {
        let mut best: HashMap<StreamKey, EdgeKind> = HashMap::new();
        let indices = self.outgoing.get(&source_addr).map(|v| v.as_slice()).unwrap_or(&[]);
        for &idx in indices {
            let edge = &self.edges[idx];
            let entry = best.entry(edge.target.addr).or_insert(edge.kind);
            if edge.kind.priority() > entry.priority() {
                *entry = edge.kind;
            }
        }
        best.into_iter().collect()
    }

    pub fn confirmed_edges(&self) -> impl Iterator<Item = &StreamEdge> {
        self.edges.iter()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Mutation weight multiplier (>=1.0) for a stream participating in dependencies.
    pub fn mutation_weight_factor(&self, addr: StreamKey) -> f64 {
        if !self.assist_enabled { return 1.0; }
        let mut factor = 1.0_f64;
        for list in [self.incoming.get(&addr), self.outgoing.get(&addr)].into_iter().flatten() {
            for &idx in list {
                let edge = &self.edges[idx];
                let base = match edge.kind { EdgeKind::Address | EdgeKind::Length => 4.0, EdgeKind::Control => 2.0 };
                let bonus = 1.5;
                factor = factor.max(base * bonus);
            }
        }
        factor
    }

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
            let values = {
                let mut sorted = e.value_set.clone();
                sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                sorted.iter().map(|(v, _)| format!("\"{v:#x}\"")).collect::<Vec<_>>().join(",")
            };
            s.push_str(&format!(
                "  {{\"source_pc\":\"{:#x}\",\"source_addr\":\"{:#x}\",\
                 \"target_pc\":\"{:#x}\",\"target_addr\":\"{:#x}\",\
                 \"kind\":\"{:?}\",\"confidence\":\"TaintConfirmed\",\"relation\":\"{}\",\
                 \"values\":[{}]}}",
                e.source.pc, e.source.addr, e.target.pc, e.target.addr,
                e.kind, relation, values
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

    /// Reclassifying an edge to a *more specific* kind clears stale fields
    /// from the previous kind (Control->Length drops the discriminant value_set).
    #[test]
    fn reclassification_clears_stale_relation_and_value_set() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);

        // Start as Control; add a discriminant.
        g.insert_or_confirm_edge(a, b, EdgeKind::Control, None);
        g.record_discriminant(a.addr, b.addr, 0x3);
        assert!(!g.governors(b.addr).next().unwrap().value_set.is_empty());

        // Reclassify to Length (priority Length > Control): value_set must be cleared.
        g.insert_or_confirm_edge(a, b, EdgeKind::Length, Some(Relation { kind: RelKind::Identity }));
        let edge = g.governors(b.addr).next().unwrap();
        assert_eq!(edge.kind, EdgeKind::Length);
        assert!(edge.value_set.is_empty(), "reclassify->Length must clear value_set");
        assert!(edge.relation.is_some(), "relation should be set after Length confirmation");
    }

    /// A confirmed `Address` edge is never downgraded to `Control`.
    #[test]
    fn address_confirmation_is_not_downgraded_by_control() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);

        // Insert as Address first.
        g.insert_or_confirm_edge(a, b, EdgeKind::Address, None);
        assert_eq!(g.governors(b.addr).next().unwrap().kind, EdgeKind::Address);

        // A later Control confirmation for the same edge must NOT clobber Address.
        g.insert_or_confirm_edge(a, b, EdgeKind::Control, None);
        let edge = g.governors(b.addr).next().unwrap();
        assert_eq!(edge.kind, EdgeKind::Address, "Control must not downgrade Address");
    }

    /// A confirmed `Length` (loop) edge is likewise not downgraded by a later
    /// non-looping Control observation -- the fitted relation is preserved.
    #[test]
    fn length_confirmation_is_not_downgraded_by_control() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);

        g.insert_or_confirm_edge(a, b, EdgeKind::Length, Some(Relation { kind: RelKind::Identity }));
        g.insert_or_confirm_edge(a, b, EdgeKind::Control, None);
        let edge = g.governors(b.addr).next().unwrap();
        assert_eq!(edge.kind, EdgeKind::Length, "Control must not downgrade Length");
        assert!(edge.relation.is_some(), "fitted relation must be preserved");
    }

    /// Discriminating values accumulate with hit counts, dedup increments
    /// the existing count, and the set stays bounded at MAX_DISCRIMINANTS.
    #[test]
    fn record_discriminant_accumulates_with_hit_counts() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);
        g.insert_or_confirm_edge(a, b, EdgeKind::Control, None);

        g.record_discriminant(a.addr, b.addr, 3);
        g.record_discriminant(a.addr, b.addr, 3); // duplicate -> hit_count=2
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

    /// Frequency-based eviction: a singleton is evicted when the cap is
    /// full, but a multi-hit entry survives.
    #[test]
    fn record_discriminant_frequency_eviction_protects_recurring_values() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);
        g.insert_or_confirm_edge(a, b, EdgeKind::Control, None);

        // Fill cap with 16 distinct singletons.
        for v in 0u64..16 {
            g.record_discriminant(a.addr, b.addr, v);
        }
        // Reinforce value 0 -> hit_count=2.
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

    /// When all entries are multi-hit, a new singleton is discarded
    /// rather than displacing a confirmed recurring value.
    #[test]
    fn record_discriminant_discards_when_all_entries_are_recurring() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);
        g.insert_or_confirm_edge(a, b, EdgeKind::Control, None);

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

    /// A discriminant is only attached to `Control` edges -- never `Length`/`Address`.
    #[test]
    fn record_discriminant_ignores_non_control_edges() {
        let mut g = StreamRelationGraph::new();
        let a = ctx(0x10, 0x5800_0000);
        let b = ctx(0x20, 0x5800_0004);
        g.insert_or_confirm_edge(a, b, EdgeKind::Length, Some(Relation { kind: RelKind::Identity }));

        g.record_discriminant(a.addr, b.addr, 42);
        let edge = g.governors(b.addr).next().unwrap();
        assert!(edge.value_set.is_empty(), "Length edge must not receive discriminants");
    }

    /// `backfill_length_relations` propagates a fitted relation to all
    /// Length edges sharing the same (src_addr, tgt_addr) pair.
    ///
    /// Scenario: two edges for the same address pair (different access PCs) are both
    /// Length, but only edge-1 has a relation. After calling `backfill_length_relations`,
    /// edge-2 must carry the same relation.
    ///
    /// Since `insert_or_confirm_edge` merges by address pair, we manually push the
    /// second edge to simulate a sibling created through a different code path.
    #[test]
    fn backfill_length_relations_fills_sibling_edges() {
        let src_addr: StreamKey = 0x4001_0000;
        let tgt_addr: StreamKey = 0x4001_0004;

        let a1 = ctx(0x100, src_addr);
        let b1 = ctx(0x200, tgt_addr);
        let a2 = ctx(0x300, src_addr); // same addresses, different PCs
        let b2 = ctx(0x400, tgt_addr);

        let mut g = StreamRelationGraph::new();

        // Edge-1 with a relation.
        let rel = Relation { kind: RelKind::Identity };
        g.insert_or_confirm_edge(a1, b1, EdgeKind::Length, Some(rel.clone()));

        // Manually push edge-2 (same address pair, different PCs, no relation).
        let idx = g.edges.len();
        g.edges.push(StreamEdge {
            source: a2,
            target: b2,
            kind: EdgeKind::Length,
            relation: None,
            value_set: Vec::new(),
        });
        // Index manually so governors() can find it.
        g.outgoing.entry(a2.addr).or_default().push(idx);
        g.incoming.entry(b2.addr).or_default().push(idx);

        // Before backfill: edge-2 has no relation.
        let has_rel_before: Vec<bool> = g.governors(tgt_addr)
            .map(|e| e.relation.is_some())
            .collect();
        assert!(has_rel_before.iter().any(|&r| !r), "one sibling should lack a relation before backfill");

        // Backfill from an iterator yielding the fitted relation for this address pair.
        g.backfill_length_relations(std::iter::once(((src_addr, tgt_addr), rel)));

        // After backfill: both edges must have a relation.
        for edge in g.governors(tgt_addr) {
            assert!(
                edge.relation.is_some(),
                "all Length edges for address pair must have a relation after backfill"
            );
        }
    }

    /// `get_governors_by_addr` returns deduplicated (source_addr, kind) pairs
    /// with highest priority kind when multiple edges share the same source_addr.
    #[test]
    fn get_governors_by_addr_deduplicates_and_picks_highest_priority() {
        let mut g = StreamRelationGraph::new();
        let tgt_addr: StreamKey = 0x5800_0004;

        // Two edges from source_addr_1 to same target: Control and Address.
        let src1_a = ctx(0x10, 0x5800_0000);
        let src1_b = ctx(0x11, 0x5800_0000); // same addr, different PC
        let tgt_a = ctx(0x20, tgt_addr);
        let tgt_b = ctx(0x21, tgt_addr);

        g.insert_or_confirm_edge(src1_a, tgt_a, EdgeKind::Control, None);
        g.insert_or_confirm_edge(src1_b, tgt_b, EdgeKind::Address, None);

        // One edge from source_addr_2 to same target: Control only.
        let src2 = ctx(0x30, 0x5800_0008);
        let tgt_c = ctx(0x22, tgt_addr);
        g.insert_or_confirm_edge(src2, tgt_c, EdgeKind::Control, None);

        let govs = g.get_governors_by_addr(tgt_addr);
        assert_eq!(govs.len(), 2, "should deduplicate to 2 distinct source addresses");

        // Source 0x5800_0000 should have Address (highest priority among Control and Address).
        let src1_entry = govs.iter().find(|(addr, _)| *addr == 0x5800_0000);
        assert!(src1_entry.is_some(), "source_addr_1 must be present");
        assert_eq!(src1_entry.unwrap().1, EdgeKind::Address, "Address beats Control");

        // Source 0x5800_0008 should have Control (only kind present).
        let src2_entry = govs.iter().find(|(addr, _)| *addr == 0x5800_0008);
        assert!(src2_entry.is_some(), "source_addr_2 must be present");
        assert_eq!(src2_entry.unwrap().1, EdgeKind::Control);
    }
}
