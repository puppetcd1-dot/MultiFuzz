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
    /// Source stream may structurally govern target; type not yet classified.
    Unclassified,
    /// Value of source stream was used to compute the MMIO address of target.
    Address,
    /// Value of source stream gated a branch condition leading to target.
    Control,
    /// Value of source stream governed the loop trip count / copy size for target.
    Length,
    /// Value of source stream governed the index/stride into target's buffer.
    Stride,
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
}

/// Directed graph of structural and taint-confirmed relationships between MMIO streams.
#[derive(Default)]
pub struct StreamRelationGraph {
    pub edges:    Vec<StreamEdge>,
    outgoing: HashMap<StreamKey, Vec<usize>>,
    incoming: HashMap<StreamKey, Vec<usize>>,
}

impl StreamRelationGraph {
    pub fn new() -> Self {
        Self::default()
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
            });
            self.outgoing.entry(source.addr).or_default().push(idx);
            self.incoming.entry(target.addr).or_default().push(idx);
        }
    }

    /// Confirm or upgrade an edge from taint analysis (Phase B output).
    pub fn confirm_edge(
        &mut self,
        source: AccessContext,
        target: AccessContext,
        kind: EdgeKind,
        relation: Option<Relation>,
    ) {
        // Upgrade an existing structural candidate if one exists (exact context match on
        // both endpoints — several targets may share an address at different PCs).
        if let Some(indices) = self.incoming.get(&target.addr).cloned() {
            for idx in indices {
                let edge = &mut self.edges[idx];
                if edge.source == source && edge.target == target {
                    edge.kind = kind;
                    edge.confidence = Confidence::TaintConfirmed;
                    if relation.is_some() {
                        edge.relation = relation;
                    }
                    return;
                }
            }
        }
        // No prior structural candidate — insert a new confirmed edge.
        let idx = self.edges.len();
        self.edges.push(StreamEdge {
            source,
            target,
            kind,
            confidence: Confidence::TaintConfirmed,
            relation,
        });
        self.outgoing.entry(source.addr).or_default().push(idx);
        self.incoming.entry(target.addr).or_default().push(idx);
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
    /// Value-flow edges (Address/Length/Stride) are robust to address-keying and get the
    /// strongest boost; Control edges are coarser (the same status register may be checked
    /// at several sites) and get a smaller boost.  A `TaintConfirmed` edge is weighted above
    /// a merely `Structural` one.
    pub fn mutation_weight_factor(&self, addr: StreamKey) -> f64 {
        let mut factor = 1.0_f64;
        for list in [self.incoming.get(&addr), self.outgoing.get(&addr)].into_iter().flatten() {
            for &idx in list {
                let edge = &self.edges[idx];
                let base = match edge.kind {
                    EdgeKind::Address | EdgeKind::Length | EdgeKind::Stride => 4.0,
                    EdgeKind::Control | EdgeKind::Unclassified => 2.0,
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
            s.push_str(&format!(
                "  {{\"source_pc\":\"{:#x}\",\"source_addr\":\"{:#x}\",\
                 \"target_pc\":\"{:#x}\",\"target_addr\":\"{:#x}\",\
                 \"kind\":\"{:?}\",\"confidence\":\"{:?}\",\"relation\":\"{}\"}}",
                e.source.pc, e.source.addr, e.target.pc, e.target.addr,
                e.kind, e.confidence, relation
            ));
        }
        s.push_str("\n]\n");
        s
    }
}
