use hashbrown::HashMap;

use crate::input::StreamKey;

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
    pub source:     StreamKey,
    pub target:     StreamKey,
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
    /// Each entry in `sources` is an MMIO stream key that may govern `target`.
    pub fn add_structural_candidates(&mut self, target: StreamKey, sources: &[StreamKey]) {
        for &source in sources {
            if source == target || self.has_edge(source, target) {
                continue;
            }
            let idx = self.edges.len();
            self.edges.push(StreamEdge {
                source,
                target,
                kind: EdgeKind::Unclassified,
                confidence: Confidence::Structural,
                relation: None,
            });
            self.outgoing.entry(source).or_default().push(idx);
            self.incoming.entry(target).or_default().push(idx);
        }
    }

    /// Confirm or upgrade an edge from taint analysis (Phase B output).
    pub fn confirm_edge(
        &mut self,
        source: StreamKey,
        target: StreamKey,
        kind: EdgeKind,
        relation: Option<Relation>,
    ) {
        // Upgrade an existing structural candidate if one exists.
        if let Some(indices) = self.incoming.get(&target).cloned() {
            for idx in indices {
                let edge = &mut self.edges[idx];
                if edge.source == source {
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
        self.outgoing.entry(source).or_default().push(idx);
        self.incoming.entry(target).or_default().push(idx);
    }

    pub fn has_edge(&self, source: StreamKey, target: StreamKey) -> bool {
        self.incoming
            .get(&target)
            .map_or(false, |v| v.iter().any(|&i| self.edges[i].source == source))
    }

    /// All edges whose target is `target`.
    pub fn governors(&self, target: StreamKey) -> impl Iterator<Item = &StreamEdge> {
        let indices = self.incoming.get(&target).map(|v| v.as_slice()).unwrap_or(&[]);
        indices.iter().map(|&i| &self.edges[i])
    }

    /// All edges whose source is `source`.
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
}
