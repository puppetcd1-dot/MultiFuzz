use hashbrown::{HashMap, HashSet};
use icicle_vm::VmExit;
use rand::Rng;
use rand_distr::{Distribution, WeightedAliasIndex};

use crate::{
    calculate_energy, config,
    input::{MultiStream, StreamKey},
    mutations::{self, Mutation, ALL_MUTATIONS},
    queue::CorpusStore,
    stream_relation::{EdgeKind, Relation, StreamRelationGraph},
    utils::{get_non_empty_streams, get_stream_weights, random_bytes},
    DictionaryRef, Fuzzer, Snapshot, StageData, StageExit,
};

// ──────────────────────────────────────────────────────────────────────────────
// Tuning constants
// ──────────────────────────────────────────────────────────────────────────────

/// Probability that mutating a stream also co-mutates one of its dependency
/// partners (governor/dependent) in the same round.
const COUPLED_MUTATION_PROB: f64 = 0.25;

/// Probability that a selected Length-source stream triggers an OVERFLOW-PROBE
/// instead of a random mutation.  OVERFLOW-PROBE sets the source to a value
/// that maps B's expected size to a known boundary (0, 1, max, 2^n …), directly
/// synthesising boundary-condition inputs that random mutation rarely reaches.
const OVERFLOW_PROBE_PROB: f64 = 0.15;

/// Probability that a stream with corpus-derived splice candidates undergoes a
/// prefix splice (borrowing working bytes from a corpus entry that successfully
/// configured the governed target) instead of a random mutation.
const SPLICE_PROB: f64 = 0.10;

/// Governs how quickly the frontier boost saturates.  A governor of a target
/// with `FRONTIER_SATURATION` corpus inputs receives half the maximum boost;
/// targets with fewer inputs receive a stronger boost.
const FRONTIER_SATURATION: f64 = 5.0;

/// Maximum additive multiplier applied to streams that govern underexplored
/// targets.  At saturation = 0 inputs the weight is multiplied by
/// `1.0 + FRONTIER_BOOST`; the boost decays toward 1.0 as coverage grows.
const FRONTIER_BOOST: f64 = 3.0;

// ──────────────────────────────────────────────────────────────────────────────
// HavocStage
// ──────────────────────────────────────────────────────────────────────────────

/// A fuzzing stage that applies random mutations to the input.
pub(crate) struct HavocStage {
    attempts: u32,
    mutator: HavocMutator,

    streams: Vec<(StreamKey, usize)>,
    stream_distr: WeightedAliasIndex<f64>,
    streams_to_extend: HashMap<StreamKey, usize>,

    /// For each stream, the dependency partners (governors ∪ dependents) that are
    /// also present and non-empty in this input — candidates for coupled mutation.
    coupled: HashMap<StreamKey, Vec<StreamKey>>,
    /// For each Length-source stream, the governed targets with fitted relations.
    /// Used both for length preservation (keeping B sized to g(val_A)) and for
    /// OVERFLOW-PROBE (setting A to g_inv(boundary)).
    length_rel: HashMap<StreamKey, Vec<(StreamKey, Relation)>>,
    /// For each stream A, working byte sequences extracted from corpus entries
    /// where at least one of A's Control/Address governed targets was non-empty.
    /// Splice mutation uses these to inject proven-working values for A so the
    /// execution starts from a configuration that previously unlocked B.
    splice_candidates: HashMap<StreamKey, Vec<Vec<u8>>>,

    log2_max_mutations: u32,
    max_mutations: u32,
    saved: bool,
}

impl StageData for HavocStage {
    fn start(fuzzer: &mut Fuzzer) -> Result<Self, StageExit> {
        fuzzer.copy_current_input();

        let (Some(id), data) = (fuzzer.input_id, &fuzzer.state.input)
        else {
            return Err(StageExit::Unsupported);
        };

        let streams = get_non_empty_streams(data);
        if streams.is_empty() {
            return Err(StageExit::Skip);
        }

        let (coupled, length_rel) = build_coupling(&fuzzer.relation_graph, &streams);

        // Coverage-directed frontier weights: governors of underexplored targets
        // are sampled more often to guide mutation toward unlocking new coverage.
        let reach_counts: HashMap<StreamKey, usize> = fuzzer
            .corpus
            .metadata
            .streams
            .iter()
            .map(|(k, m)| (*k, m.num_inputs))
            .collect();
        let frontier_weights =
            compute_frontier_weights(&fuzzer.relation_graph, &streams, &reach_counts);

        // Prefix-splice candidates: corpus entries where a governed target is
        // non-empty → the governor stream bytes from that entry are candidate
        // splice donors that previously unlocked the target.
        let splice_candidates =
            collect_splice_candidates(&fuzzer.relation_graph, &streams, &fuzzer.corpus);

        let stream_distr = get_stream_weights(fuzzer, id, &streams, &frontier_weights);
        let mutator = HavocMutator::new();
        let attempts = calculate_energy(fuzzer) as u32;
        let log2_max_mutations = log2_max_mutations(fuzzer);
        let max_mutations = max_mutations(fuzzer);

        fuzzer.corpus[id].metadata.havoc_rounds += 1;

        tracing::trace!(
            "[{id}] havoc for {attempts} attempts with {} max mutations \
             ({} splice candidates, {} frontier-boosted streams)",
            2_u64.pow(log2_max_mutations),
            splice_candidates.values().map(|v| v.len()).sum::<usize>(),
            frontier_weights.iter().filter(|&&w| w > 1.5).count(),
        );
        Ok(Self {
            attempts,
            streams,
            stream_distr,
            mutator,
            log2_max_mutations,
            max_mutations,
            streams_to_extend: HashMap::new(),
            coupled,
            length_rel,
            splice_candidates,
            saved: false,
        })
    }

    fn fuzz_one(&mut self, fuzzer: &mut Fuzzer) -> Option<VmExit> {
        self.attempts = self.attempts.checked_sub(1)?;

        Snapshot::restore_initial(fuzzer);
        fuzzer.copy_current_input();
        fuzzer.reset_input_cursor().unwrap();

        self.havoc_v1(fuzzer);

        // Also extend any streams that have caused us to exit because there were too small, these
        // streams will be trimmed back to the correct length as part of `auto_trim_input` if the
        // extension was unnecessary
        let data = &mut fuzzer.state.input;
        for (key, count) in &self.streams_to_extend {
            let bytes = &mut data.streams.entry(*key).or_default().bytes;
            if bytes.len() >= config::MAX_STREAM_LEN {
                continue;
            }

            let local_dict = fuzzer.dict.entry(*key).or_default();
            local_dict.compute_weights();
            let dict = DictionaryRef { local: local_dict, global: &fuzzer.global_dict };
            mutations::extend_input_by(&mut fuzzer.rng, dict, bytes, 4 * count);
        }

        fuzzer.write_input_to_target().unwrap();
        let exit = fuzzer.execute()?;

        // Keep track of the streams that cause us to exit because they are too small.
        if let Some(key) = fuzzer.state.input.last_read {
            *self.streams_to_extend.entry(key).or_default() += 1;
        }

        fuzzer.auto_trim_input().ok()?;

        if fuzzer.debug.havoc && !self.saved {
            let _ = std::fs::write(
                fuzzer.workdir.join(format!("queue/{}.havoc.bin", fuzzer.input_id.unwrap_or(0))),
                fuzzer.state.input.to_bytes(),
            );
            self.saved = true;
        }

        Some(exit)
    }
}

impl HavocStage {
    fn havoc_v1(&mut self, fuzzer: &mut Fuzzer) {
        let mut mutations = fuzzer.rng.gen_range(1..=self.max_mutations);
        while mutations > 0 {
            let (key, _) = self.streams[self.stream_distr.sample(&mut fuzzer.rng)];
            let num_mutations = fuzzer.rng.gen_range(1..=mutations);
            mutations -= num_mutations;

            // OVERFLOW-PROBE: when the selected stream is a confirmed Length source
            // with a fitted relation, occasionally probe at boundary values of B
            // (0, 1, current_len±1, 2^n …) by solving A = g_inv(boundary) and
            // extending B to that size.  This synthesises boundary-condition inputs
            // directly instead of waiting for random mutation to reach them.
            let probe_target: Option<(StreamKey, Relation)> = {
                let has_length =
                    self.length_rel.get(&key).map_or(false, |r| !r.is_empty());
                if has_length && fuzzer.rng.gen_bool(OVERFLOW_PROBE_PROB) {
                    let rels = self.length_rel.get(&key).unwrap();
                    let idx = fuzzer.rng.gen_range(0..rels.len());
                    Some(rels[idx].clone())
                } else {
                    None
                }
            };

            if let Some((target, rel)) = probe_target {
                self.overflow_probe(fuzzer, key, target, &rel);
            } else {
                // PREFIX SPLICE: if this stream has corpus-derived discriminating
                // values (bytes that previously unlocked a governed target), inject
                // them with SPLICE_PROB probability instead of random mutation.
                // This drives the execution toward known-good configurations that
                // unlock specific peripherals, rather than exploring blindly.
                let should_splice =
                    self.splice_candidates.get(&key).map_or(false, |v| !v.is_empty())
                        && fuzzer.rng.gen_bool(SPLICE_PROB);
                if should_splice {
                    self.splice_from_donor(fuzzer, key);
                } else {
                    self.mutate_stream(fuzzer, key, num_mutations);
                }
            }

            // Coupled mutation: with some probability also evolve one of `key`'s
            // dependency partners so related streams move together.
            let partner = self.coupled.get(&key).filter(|r| !r.is_empty()).and_then(|r| {
                fuzzer
                    .rng
                    .gen_bool(COUPLED_MUTATION_PROB)
                    .then(|| r[fuzzer.rng.gen_range(0..r.len())])
            });
            if let Some(key2) = partner {
                self.mutate_stream(fuzzer, key2, 1 + num_mutations / 2);
                self.preserve_length(fuzzer, key, key2);
            }
        }
    }

    /// Apply `num` havoc mutations to a single present, non-empty stream.
    fn mutate_stream(&mut self, fuzzer: &mut Fuzzer, key: StreamKey, num: u32) {
        let data = &mut fuzzer.state.input;
        let Some(stream) = data.streams.get_mut(&key) else { return };
        if stream.bytes.is_empty() {
            return;
        }
        let bytes = &mut stream.bytes;

        let local_dict = fuzzer.dict.entry(key).or_default();
        local_dict.compute_weights();
        let dict = DictionaryRef { local: local_dict, global: &fuzzer.global_dict };

        for _ in 0..num {
            if let Some(mutation) =
                self.mutator.havoc_bytes(&mut fuzzer.rng, dict, bytes, key, &fuzzer.corpus, 0)
            {
                fuzzer.state.mutation_kinds.push((key, mutation).into());
            }
        }
    }

    /// `Length` PRESERVE consequence: when `source` governs `target` through a fitted
    /// relation `g`, ensure `target` holds at least `g(value(source))` bytes so the
    /// loop driven by `source` has data to consume (extension only — never truncates;
    /// `auto_trim_input` trims any surplus afterwards).
    fn preserve_length(&mut self, fuzzer: &mut Fuzzer, source: StreamKey, target: StreamKey) {
        let relation = self
            .length_rel
            .get(&source)
            .and_then(|v| v.iter().find(|(t, _)| *t == target))
            .map(|(_, r)| r.clone());
        let Some(relation) = relation else { return };

        let data = &mut fuzzer.state.input;
        let val_a = data.streams.get(&source).map_or(0, |s| le_value(&s.bytes));
        let want = (relation.apply(val_a) as usize).min(config::MAX_STREAM_LEN);

        let bytes = &mut data.streams.entry(target).or_default().bytes;
        if bytes.len() >= want {
            return;
        }
        let amount = want - bytes.len();
        let local_dict = fuzzer.dict.entry(target).or_default();
        local_dict.compute_weights();
        let dict = DictionaryRef { local: local_dict, global: &fuzzer.global_dict };
        mutations::extend_input_by(&mut fuzzer.rng, dict, bytes, amount);
    }

    /// OVERFLOW-PROBE: set `source` to `g_inv(boundary) ± δ` for a randomly selected
    /// boundary count of `target`, and extend `target` to that boundary size.
    ///
    /// This synthesises boundary-condition inputs directly: the source register is set
    /// to the exact value that places B's expected buffer size at 0, 1, current±1,
    /// or a power-of-two boundary, which is where OOB/overflow errors typically live.
    fn overflow_probe(
        &mut self,
        fuzzer: &mut Fuzzer,
        source: StreamKey,
        target: StreamKey,
        rel: &Relation,
    ) {
        let target_len =
            fuzzer.state.input.streams.get(&target).map_or(0, |s| s.bytes.len());
        let counts = boundary_counts(target_len);
        let count_b = counts[fuzzer.rng.gen_range(0..counts.len())];

        let Some(val_a) = rel.invert(count_b) else { return };

        // Probe exact boundary and its immediate neighbours to catch off-by-one errors.
        let delta: i64 = [-1i64, 0, 0, 1][fuzzer.rng.gen_range(0..4)];
        let probe_val = (val_a as i64).saturating_add(delta).max(0) as u64;

        // Write probe_val into source stream (little-endian, first 4 bytes).
        let src_bytes = &mut fuzzer.state.input.streams.entry(source).or_default().bytes;
        if src_bytes.len() < 4 {
            src_bytes.resize(4, 0);
        }
        let le = probe_val.to_le_bytes();
        src_bytes[..4].copy_from_slice(&le[..4]);

        // Extend target to the boundary size so the execution exercises the edge.
        // auto_trim_input will shrink it back if the extension added no coverage.
        if count_b > 0 && count_b <= config::MAX_STREAM_LEN as u64 {
            let want = count_b as usize;
            let tgt = &mut fuzzer.state.input.streams.entry(target).or_default().bytes;
            if tgt.len() < want {
                tgt.resize(want, 0);
            }
        }
    }

    /// PREFIX SPLICE: replace a prefix of `key`'s bytes with working bytes from a
    /// randomly selected corpus donor — a corpus entry that previously configured
    /// `key` with a value that successfully reached one of its governed targets.
    ///
    /// Hybrid splice: only the leading `splice_len` bytes are replaced; the
    /// remainder is kept from the current input, preserving other state that may
    /// have been accumulated by earlier mutations.
    fn splice_from_donor(&mut self, fuzzer: &mut Fuzzer, key: StreamKey) {
        let Some(donors) = self.splice_candidates.get(&key) else { return };
        if donors.is_empty() {
            return;
        }
        let donor_idx = fuzzer.rng.gen_range(0..donors.len());
        // Clone donor bytes to avoid conflicting borrows of self.
        let donor = donors[donor_idx].clone();
        if donor.is_empty() {
            return;
        }

        let bytes = &mut fuzzer.state.input.streams.entry(key).or_default().bytes;
        // Splice up to the full donor length or a random sub-prefix.
        let max_splice = donor.len();
        let splice_len = fuzzer.rng.gen_range(1..=max_splice);
        if bytes.len() < splice_len {
            bytes.resize(splice_len, 0);
        }
        bytes[..splice_len].copy_from_slice(&donor[..splice_len]);
    }

    #[allow(unused)]
    fn havoc_v2(&mut self, fuzzer: &mut Fuzzer) {
        let Some(input_id) = fuzzer.input_id
        else {
            return;
        };
        let max_find_gap = fuzzer.corpus[input_id].metadata.max_find_gap;

        let data = &mut fuzzer.state.input;
        for &(key, _) in &self.streams {
            if fuzzer.rng.gen_bool(0.5) {
                continue;
            }

            let bytes = &mut data.streams.get_mut(&key).unwrap().bytes;
            let local_dict = fuzzer.dict.entry(key).or_default();
            local_dict.compute_weights();
            let dict = DictionaryRef { local: local_dict, global: &fuzzer.global_dict };

            let mutations = fuzzer.rng.gen_range(1..=self.log2_max_mutations);
            for _ in 0..mutations {
                self.mutator.havoc_bytes(&mut fuzzer.rng, dict, bytes, key, &fuzzer.corpus, 0);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Mutation budget helpers
// ──────────────────────────────────────────────────────────────────────────────

fn log2_max_mutations(fuzzer: &Fuzzer) -> u32 {
    let max_find_gap =
        fuzzer.input_id.map(|id| fuzzer.corpus[id].metadata.max_find_gap).unwrap_or(0);
    match max_find_gap {
        ..=1000 => 3,
        ..=10000 => 4,
        ..=100000 => 5,
        ..=1000000 => 6,
        ..=10000000 => 7,
        _ => 8,
    }
}

fn max_mutations(fuzzer: &Fuzzer) -> u32 {
    let max_find_gap =
        fuzzer.input_id.map(|id| fuzzer.corpus[id].metadata.max_find_gap).unwrap_or(0);
    match max_find_gap {
        ..=100 => 4,
        ..=1000 => 8,
        ..=10000 => 16,
        ..=100000 => 32,
        _ => 64,
    }
}

fn mutation_weight(mutation: &Mutation) -> u32 {
    return match mutation {
        Mutation::BitFlip => 20,
        Mutation::ReplaceByte => 40,
        Mutation::IncDec => 10,
        Mutation::InsertByte => 10,
        Mutation::Insert4 => 10,
        Mutation::RemoveByte => 5,
        Mutation::Remove4 => 5,
        Mutation::InterestingValue => 10,
        Mutation::DictReplace => 20,
        Mutation::DictInsert => 20,
        Mutation::StreamSplice => 5,
        Mutation::InnerSplice => 5,
        Mutation::RandomSplice => 5,
        Mutation::RemoveRegion => 1,
    };
}

// ──────────────────────────────────────────────────────────────────────────────
// HavocMutator
// ──────────────────────────────────────────────────────────────────────────────

pub struct HavocMutator {
    mutation_distr: WeightedAliasIndex<u32>,
}

impl HavocMutator {
    pub(crate) fn new() -> Self {
        let weights = ALL_MUTATIONS.iter().map(mutation_weight).collect();
        Self { mutation_distr: WeightedAliasIndex::new(weights).unwrap() }
    }

    #[allow(unused)]
    fn random_weights<R: Rng>(rng: &mut R) -> Self {
        let weights = ALL_MUTATIONS.iter().map(|_| rng.gen_range(0..100)).collect();
        let Ok(mutation_distr) = WeightedAliasIndex::new(weights)
        else {
            return HavocMutator::new();
        };
        Self { mutation_distr }
    }

    pub(crate) fn havoc_bytes<R>(
        &self,
        rng: &mut R,
        dict: DictionaryRef,
        input: &mut Vec<u8>,
        key: StreamKey,
        corpus: &CorpusStore<MultiStream>,
        offset: usize,
    ) -> Option<Mutation>
    where
        R: Rng,
    {
        if input.is_empty() {
            random_bytes(rng, input);
            return None;
        }

        let mutation = ALL_MUTATIONS[self.mutation_distr.sample(rng)];
        mutations::apply_mutation(mutation, rng, dict, input, key, corpus, offset);

        if input.is_empty() {
            random_bytes(rng, input);
            return None;
        }

        Some(mutation)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Coupling and frontier helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Build the per-stream coupling maps from the relation graph, restricted to streams
/// that are present and non-empty in this input.  Returns `(coupled, length_rel)`:
///   * `coupled[k]`   — partner streams (dependents of `k` ∪ governors of `k`) to
///                      co-mutate when `k` is mutated;
///   * `length_rel[s]`— `(target, relation)` for each confirmed `Length` edge `s → target`
///                      used for length preservation and OVERFLOW-PROBE.
fn build_coupling(
    graph: &StreamRelationGraph,
    streams: &[(StreamKey, usize)],
) -> (HashMap<StreamKey, Vec<StreamKey>>, HashMap<StreamKey, Vec<(StreamKey, Relation)>>) {
    let mut coupled: HashMap<StreamKey, Vec<StreamKey>> = HashMap::new();
    let mut length_rel: HashMap<StreamKey, Vec<(StreamKey, Relation)>> = HashMap::new();
    if !graph.assist_enabled() {
        return (coupled, length_rel);
    }

    let present: HashSet<StreamKey> = streams.iter().map(|(k, _)| *k).collect();
    for &(key, _) in streams {
        for edge in graph.dependents(key) {
            let target = edge.target.addr;
            if target == key || !present.contains(&target) {
                continue;
            }
            coupled.entry(key).or_default().push(target);
            if edge.kind == EdgeKind::Length {
                if let Some(rel) = &edge.relation {
                    length_rel.entry(key).or_default().push((target, rel.clone()));
                }
            }
        }
        for edge in graph.governors(key) {
            let source = edge.source.addr;
            if source == key || !present.contains(&source) {
                continue;
            }
            coupled.entry(key).or_default().push(source);
        }
    }

    for partners in coupled.values_mut() {
        partners.sort_unstable();
        partners.dedup();
    }
    (coupled, length_rel)
}

/// Compute coverage-directed frontier boost weights (parallel to `streams`).
///
/// A stream A that governs a rarely-reached target B receives a weight > 1.0 so
/// the mutation scheduler selects it more often when coverage is thin around B.
/// The boost decays smoothly toward 1.0 as `B.num_inputs` grows past
/// `FRONTIER_SATURATION`, so the bias is strongest at the start of fuzzing and
/// diminishes once B is well-explored.
///
/// Formula: `weight[A] = 1.0 + FRONTIER_BOOST * max_B(FRONTIER_SATURATION / (FRONTIER_SATURATION + reach(B)))`
fn compute_frontier_weights(
    graph: &StreamRelationGraph,
    streams: &[(StreamKey, usize)],
    reach_counts: &HashMap<StreamKey, usize>,
) -> Vec<f64> {
    streams
        .iter()
        .map(|(key, _)| {
            if !graph.assist_enabled() {
                return 1.0;
            }
            let max_frontier = graph
                .dependents(*key)
                .map(|edge| {
                    let reach = reach_counts.get(&edge.target.addr).copied().unwrap_or(0) as f64;
                    FRONTIER_SATURATION / (FRONTIER_SATURATION + reach)
                })
                .fold(0.0_f64, f64::max);
            1.0 + FRONTIER_BOOST * max_frontier
        })
        .collect()
}

/// Collect per-stream splice candidates from the corpus.
///
/// For each Control or Address edge A→B in the graph (restricted to present streams),
/// scan the most-recent corpus entries (up to `MAX_SCAN`) for entries where B is
/// non-empty.  The A-stream bytes from those entries are stored as splice candidates:
/// splicing them into the current input's A-stream seeds the execution with a value
/// that previously unlocked B, giving the fuzzer a structural starting point instead
/// of random bytes.
fn collect_splice_candidates(
    graph: &StreamRelationGraph,
    streams: &[(StreamKey, usize)],
    corpus: &CorpusStore<MultiStream>,
) -> HashMap<StreamKey, Vec<Vec<u8>>> {
    let mut candidates: HashMap<StreamKey, Vec<Vec<u8>>> = HashMap::new();
    if !graph.assist_enabled() {
        return candidates;
    }

    let corpus_count = corpus.inputs();
    if corpus_count == 0 {
        return candidates;
    }

    let present: HashSet<StreamKey> = streams.iter().map(|(k, _)| *k).collect();

    // Scan the most-recent entries first (they tend to cover more of the target).
    const MAX_SCAN: usize = 100;
    const MAX_DONORS_PER_STREAM: usize = 16;
    let scan_start = corpus_count.saturating_sub(MAX_SCAN);

    for &(src_key, _) in streams {
        for edge in graph.dependents(src_key) {
            if !matches!(edge.kind, EdgeKind::Control | EdgeKind::Address) {
                continue;
            }
            let target = edge.target.addr;
            if !present.contains(&target) {
                continue;
            }

            for i in scan_start..corpus_count {
                let entry = &corpus[i];
                // Only entries where target is non-empty (B was reached).
                if entry.data.streams.get(&target).map_or(true, |s| s.bytes.is_empty()) {
                    continue;
                }
                // Take the source stream bytes from this entry as a donor.
                if let Some(src_stream) = entry.data.streams.get(&src_key) {
                    if src_stream.bytes.is_empty() {
                        continue;
                    }
                    let list = candidates.entry(src_key).or_default();
                    if list.len() >= MAX_DONORS_PER_STREAM {
                        break;
                    }
                    let bytes = src_stream.bytes.clone();
                    if !list.iter().any(|b| b == &bytes) {
                        list.push(bytes);
                    }
                }
            }
        }
    }

    candidates
}

/// Boundary byte-counts to probe for a target stream of `current_len` bytes.
///
/// Covers: underflow (0, 1, 2), common powers-of-two, large-value overflows,
/// and the neighbourhood of the current length (±1, ×2) where off-by-one
/// errors are most common.
fn boundary_counts(current_len: usize) -> Vec<u64> {
    let mut counts: Vec<u64> = vec![
        0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 0x7FFF, 0x8000, 0xFFFF, 0x1_0000,
    ];
    if current_len > 0 {
        let l = current_len as u64;
        counts.extend_from_slice(&[l.saturating_sub(1), l, l + 1, l.saturating_mul(2)]);
    }
    counts.sort_unstable();
    counts.dedup();
    counts
}

/// Interpret the leading bytes of a stream as a little-endian unsigned integer.
/// Reads up to 8 bytes.
fn le_value(bytes: &[u8]) -> u64 {
    let n = bytes.len().min(8);
    let mut v = 0u64;
    for (i, &b) in bytes[..n].iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    v
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        input::{MultiStream, StreamData},
        queue::CorpusStore,
        stream_relation::{AccessContext, EdgeKind, RelKind, Relation, StreamRelationGraph},
        State,
    };

    fn ac(pc: u64, addr: StreamKey) -> AccessContext {
        AccessContext::new(pc, addr)
    }

    fn graph_with_edge(
        src: StreamKey,
        tgt: StreamKey,
        kind: EdgeKind,
        confirmed: bool,
    ) -> StreamRelationGraph {
        let mut g = StreamRelationGraph::new();
        g.add_structural_candidates(ac(0x20, tgt), &[(ac(0x10, src), kind)]);
        if confirmed {
            g.confirm_edge(ac(0x10, src), ac(0x20, tgt), kind, None);
        }
        g
    }

    // ── build_coupling ────────────────────────────────────────────────────────

    /// Coupling links a confirmed dependency both ways and surfaces the fitted
    /// relation of a Length edge for length preservation / overflow probe.
    #[test]
    fn coupling_maps_link_present_partners_and_relations() {
        let a = 0x5800_0000u64;
        let b = 0x5800_0004u64;
        let absent = 0x5800_0008u64;

        let mut graph = StreamRelationGraph::new();
        graph.add_structural_candidates(ac(0x20, b), &[(ac(0x10, a), EdgeKind::Control)]);
        graph.confirm_edge(
            ac(0x10, a),
            ac(0x20, b),
            EdgeKind::Length,
            Some(Relation { kind: RelKind::Identity }),
        );
        graph.add_structural_candidates(ac(0x30, absent), &[(ac(0x10, a), EdgeKind::Control)]);

        let streams = vec![(a, 4usize), (b, 4usize)];
        let (coupled, length_rel) = build_coupling(&graph, &streams);

        assert_eq!(
            coupled.get(&a).map(|v| v.as_slice()),
            Some([b].as_slice()),
            "source couples to its present target only"
        );
        assert_eq!(
            coupled.get(&b).map(|v| v.as_slice()),
            Some([a].as_slice()),
            "target couples back to its governor"
        );
        assert!(length_rel.get(&a).is_some(), "Length relation surfaced for source");
    }

    /// With assist disabled, coupling and length_rel must be empty (ablation arm).
    #[test]
    fn coupling_empty_when_assist_disabled() {
        let a = 0x5800_0000u64;
        let b = 0x5800_0004u64;
        let mut graph = StreamRelationGraph::new();
        graph.add_structural_candidates(ac(0x20, b), &[(ac(0x10, a), EdgeKind::Control)]);
        graph.set_assist_enabled(false);

        let (coupled, length_rel) = build_coupling(&graph, &[(a, 4), (b, 4)]);
        assert!(coupled.is_empty() && length_rel.is_empty());
    }

    // ── compute_frontier_weights ──────────────────────────────────────────────

    /// A governor of an unexplored target (0 corpus inputs) must receive the
    /// maximum frontier boost.  A governor of a well-explored target must receive
    /// approximately 1.0 (no significant boost).
    #[test]
    fn frontier_boost_elevates_governor_of_rare_target() {
        let a = 0x5800_0000u64;
        let b = 0x5800_0004u64;
        let streams = vec![(a, 4usize), (b, 4usize)];
        let graph = graph_with_edge(a, b, EdgeKind::Control, false);

        // B has never been reached.
        let reach_zero: HashMap<StreamKey, usize> = HashMap::from_iter([(b, 0)]);
        let weights_zero = compute_frontier_weights(&graph, &streams, &reach_zero);
        let a_idx = streams.iter().position(|(k, _)| *k == a).unwrap();
        let b_idx = streams.iter().position(|(k, _)| *k == b).unwrap();
        // A governs B → strong boost expected.
        assert!(
            weights_zero[a_idx] > 2.0,
            "governor of unexplored target should be boosted; got {:.2}",
            weights_zero[a_idx]
        );
        // B has no out-edges so no boost for B.
        assert!(
            (weights_zero[b_idx] - 1.0).abs() < 1e-9,
            "non-governor should have weight 1.0; got {:.2}",
            weights_zero[b_idx]
        );

        // B is well-explored (100 corpus inputs → strong saturation).
        let reach_full: HashMap<StreamKey, usize> = HashMap::from_iter([(b, 100)]);
        let weights_full = compute_frontier_weights(&graph, &streams, &reach_full);
        assert!(
            weights_full[a_idx] < weights_zero[a_idx],
            "boost must decay as target coverage grows"
        );
    }

    /// When assist is disabled, frontier weights must all be 1.0.
    #[test]
    fn frontier_weights_all_one_when_assist_disabled() {
        let a = 0x5800_0000u64;
        let b = 0x5800_0004u64;
        let mut graph = StreamRelationGraph::new();
        graph.add_structural_candidates(ac(0x20, b), &[(ac(0x10, a), EdgeKind::Control)]);
        graph.set_assist_enabled(false);

        let streams = vec![(a, 4usize), (b, 4usize)];
        let reach: HashMap<StreamKey, usize> = HashMap::from_iter([(b, 0)]);
        let weights = compute_frontier_weights(&graph, &streams, &reach);
        assert!(weights.iter().all(|&w| (w - 1.0).abs() < 1e-9));
    }

    // ── collect_splice_candidates ─────────────────────────────────────────────

    fn make_corpus_with_entry(
        a: StreamKey,
        a_bytes: Vec<u8>,
        b: StreamKey,
        b_bytes: Vec<u8>,
    ) -> CorpusStore<MultiStream> {
        let mut corpus = CorpusStore::<MultiStream>::default();
        let mut state = State::default();
        let streams = hashbrown::HashMap::from_iter([
            (a, StreamData::new(a_bytes)),
            (b, StreamData::new(b_bytes)),
        ]);
        state.input = MultiStream::new(streams);
        corpus.add(&state);
        corpus
    }

    /// When a corpus entry has B non-empty, A's bytes from that entry must be
    /// collected as a splice candidate for stream A.
    #[test]
    fn splice_candidates_collected_from_corpus_entry() {
        let a = 0x5800_0000u64;
        let b = 0x5800_0004u64;
        let a_bytes = vec![0x01u8, 0x02, 0x03, 0x04];

        let mut graph = StreamRelationGraph::new();
        graph.add_structural_candidates(ac(0x20, b), &[(ac(0x10, a), EdgeKind::Control)]);

        let corpus = make_corpus_with_entry(a, a_bytes.clone(), b, vec![0xAA]);
        let streams = vec![(a, 4usize), (b, 1usize)];
        let candidates = collect_splice_candidates(&graph, &streams, &corpus);

        assert!(
            candidates.contains_key(&a),
            "A must have splice candidates when B is reached"
        );
        let list = &candidates[&a];
        assert!(!list.is_empty());
        assert_eq!(list[0], a_bytes, "candidate must match A bytes from donor entry");
    }

    /// When B is empty in all corpus entries, A must have no splice candidates.
    #[test]
    fn no_splice_candidates_when_target_never_reached() {
        let a = 0x5800_0000u64;
        let b = 0x5800_0004u64;
        let mut graph = StreamRelationGraph::new();
        graph.add_structural_candidates(ac(0x20, b), &[(ac(0x10, a), EdgeKind::Control)]);

        // Corpus entry where B is EMPTY.
        let corpus = make_corpus_with_entry(a, vec![0x01, 0x02], b, vec![]);
        let streams = vec![(a, 2usize), (b, 0usize)];
        let candidates = collect_splice_candidates(&graph, &streams, &corpus);

        assert!(
            candidates.get(&a).map_or(true, |v| v.is_empty()),
            "no candidates should be collected when target was never reached"
        );
    }

    /// With assist disabled, splice candidates must always be empty.
    #[test]
    fn splice_candidates_empty_when_assist_disabled() {
        let a = 0x5800_0000u64;
        let b = 0x5800_0004u64;
        let mut graph = StreamRelationGraph::new();
        graph.add_structural_candidates(ac(0x20, b), &[(ac(0x10, a), EdgeKind::Control)]);
        graph.set_assist_enabled(false);

        let corpus = make_corpus_with_entry(a, vec![0x01], b, vec![0xAA]);
        let candidates = collect_splice_candidates(&graph, &[(a, 1), (b, 1)], &corpus);
        assert!(candidates.is_empty());
    }

    // ── boundary_counts ───────────────────────────────────────────────────────

    /// boundary_counts must always include 0, 1, and both neighbours of the
    /// current length, and the list must be sorted and deduplicated.
    #[test]
    fn boundary_counts_includes_standard_values_and_current_len_neighbours() {
        let counts = boundary_counts(10);
        assert!(counts.contains(&0), "must contain 0");
        assert!(counts.contains(&1), "must contain 1");
        assert!(counts.contains(&9), "must contain current_len - 1");
        assert!(counts.contains(&10), "must contain current_len");
        assert!(counts.contains(&11), "must contain current_len + 1");

        // Sorted and deduplicated.
        let mut sorted = counts.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(counts, sorted);
    }

    // ── le_value ──────────────────────────────────────────────────────────────

    #[test]
    fn le_value_reads_little_endian_prefix() {
        assert_eq!(le_value(&[0x04, 0x00, 0x00, 0x00]), 4);
        assert_eq!(le_value(&[0x00, 0x01]), 256);
        assert_eq!(le_value(&[]), 0);
    }
}
