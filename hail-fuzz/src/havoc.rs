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

/// Probability that a Control-source stream with known discriminating values has
/// one injected (overwriting its leading word) instead of a random mutation.
/// Discriminant injection sets the source to a concrete value Phase B observed to
/// open the path to a governed target — the precise way to unlock command-dispatch
/// peripherals that a gating value, not buffer shape, controls.
const DISCRIMINANT_INJECT_PROB: f64 = 0.20;

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
    /// For each Control-source stream, the concrete gating values Phase B observed
    /// to open the path to a governed target.  Injecting one sets the source to a
    /// value known to discriminate the branch, unlocking targets that may not even
    /// be present yet (the new-coverage case).
    discriminants: HashMap<StreamKey, Vec<u64>>,

    log2_max_mutations: u32,
    max_mutations: u32,
    saved: bool,
}

impl StageData for HavocStage {
    fn start(fuzzer: &mut Fuzzer) -> Result<Self, StageExit> {
        // Refresh coverage-directed frontier weights (throttled on coverage growth) before
        // building the stream-selection distribution, so mutation energy is steered toward
        // the MMIO flows gating genuinely uncovered code.
        fuzzer.maybe_recompute_frontier();

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

        // Project the precomputed frontier weights onto this input's streams (parallel to
        // `streams`): a stream that gates an uncovered frontier branch is mutated more often.
        let frontier_boosts: Vec<f64> = streams
            .iter()
            .map(|(key, _)| fuzzer.frontier_weights.get(key).copied().unwrap_or(1.0))
            .collect();

        // Prefix-splice candidates: corpus entries where a governed target is
        // non-empty → the governor stream bytes from that entry are candidate
        // splice donors that previously unlocked the target.
        let splice_candidates =
            collect_splice_candidates(&fuzzer.relation_graph, &streams, &fuzzer.corpus);

        // Gating-value injection candidates: for each present Control source, the
        // concrete values Phase B saw open the path to a governed target.
        let discriminants = collect_discriminants(&fuzzer.relation_graph, &streams);

        let stream_distr = get_stream_weights(fuzzer, id, &streams, &frontier_boosts);
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
            frontier_boosts.iter().filter(|&&w| w > 1.5).count(),
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
            discriminants,
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
                // DISCRIMINANT INJECTION: if this stream is a Control source with
                // known gating values, set it to one of them (with
                // DISCRIMINANT_INJECT_PROB) so the gated path opens — the precise
                // unlock that random search rarely hits on a multi-way comparison.
                let should_inject =
                    self.discriminants.get(&key).map_or(false, |v| !v.is_empty())
                        && fuzzer.rng.gen_bool(DISCRIMINANT_INJECT_PROB);
                // PREFIX SPLICE: otherwise, if this stream has corpus-derived
                // working bytes (that previously unlocked a governed target), inject
                // them with SPLICE_PROB probability instead of random mutation, to
                // drive execution toward known-good configurations rather than
                // exploring blindly.
                let should_splice =
                    self.splice_candidates.get(&key).map_or(false, |v| !v.is_empty())
                        && fuzzer.rng.gen_bool(SPLICE_PROB);
                if should_inject {
                    self.inject_discriminant(fuzzer, key);
                } else if should_splice {
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

    /// DISCRIMINANT INJECTION: overwrite the leading word of `key`'s stream with a
    /// concrete gating value Phase B observed to open the path to a governed target.
    ///
    /// The value is written little-endian over `min(len, 4)` leading bytes — the
    /// width a typical MMIO read consumes — so a narrow read sees the low byte(s)
    /// and a word read sees the whole value.  No bytes beyond the first word are
    /// touched, leaving any subsequent reads from the same stream intact.  The
    /// stream is never created or grown here: an absent/empty source has nothing to
    /// gate with, and growing it is the job of the normal extension path.
    fn inject_discriminant(&mut self, fuzzer: &mut Fuzzer, key: StreamKey) {
        let Some(values) = self.discriminants.get(&key) else { return };
        if values.is_empty() {
            return;
        }
        let value = values[fuzzer.rng.gen_range(0..values.len())];

        let bytes = &mut fuzzer.state.input.streams.entry(key).or_default().bytes;
        if bytes.is_empty() {
            return;
        }
        let width = bytes.len().min(4);
        let le = value.to_le_bytes();
        bytes[..width].copy_from_slice(&le[..width]);
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

/// Collect per-stream discriminating values from the relation graph.
///
/// For each present stream that is the source of a confirmed `Control` edge, gather
/// the gating values Phase B recorded on that edge.  Unlike splice candidates, the
/// governed target need NOT be present: the whole point is to inject a gating value
/// that may unlock a target the current input has not yet reached (new coverage).
fn collect_discriminants(
    graph: &StreamRelationGraph,
    streams: &[(StreamKey, usize)],
) -> HashMap<StreamKey, Vec<u64>> {
    let mut map: HashMap<StreamKey, Vec<u64>> = HashMap::new();
    if !graph.assist_enabled() {
        return map;
    }
    for &(key, _) in streams {
        for edge in graph.dependents(key) {
            if edge.kind != EdgeKind::Control || edge.value_set.is_empty() {
                continue;
            }
            let list = map.entry(key).or_default();
            // Sort by hit_count descending so highest-frequency discriminants
            // appear first; they are more likely to be the true gating values.
            let mut sorted = edge.value_set.clone();
            sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            for (v, _) in sorted {
                if !list.contains(&v) {
                    list.push(v);
                }
            }
        }
    }
    map
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

    // ── collect_discriminants ─────────────────────────────────────────────────

    /// Gating values recorded on a confirmed Control edge are surfaced for the
    /// source stream even when the governed target is absent (the unlock case).
    #[test]
    fn discriminants_collected_for_control_source_even_if_target_absent() {
        let a = 0x5800_0000u64;
        let b = 0x5800_0004u64;

        let mut graph = StreamRelationGraph::new();
        graph.add_structural_candidates(ac(0x20, b), &[(ac(0x10, a), EdgeKind::Control)]);
        graph.confirm_edge(ac(0x10, a), ac(0x20, b), EdgeKind::Control, None);
        graph.record_discriminant(a, b, 3);
        graph.record_discriminant(a, b, 7);

        // Only A is present; B (the target) is NOT in the input.
        let streams = vec![(a, 4usize)];
        let map = collect_discriminants(&graph, &streams);

        assert_eq!(
            map.get(&a).map(|v| v.as_slice()),
            Some([3u64, 7u64].as_slice()),
            "Control source surfaces its gating values regardless of target presence"
        );
    }

    /// Length and Address edges contribute no discriminants, and an assist-disabled
    /// graph yields none at all (ablation arm).
    #[test]
    fn discriminants_excludes_non_control_and_respects_assist() {
        let a = 0x5800_0000u64;
        let b = 0x5800_0004u64;

        let mut graph = StreamRelationGraph::new();
        graph.add_structural_candidates(ac(0x20, b), &[(ac(0x10, a), EdgeKind::Control)]);
        // Reclassify to Length: a discriminant recorded now must be ignored.
        graph.confirm_edge(
            ac(0x10, a),
            ac(0x20, b),
            EdgeKind::Length,
            Some(Relation { kind: RelKind::Identity }),
        );
        graph.record_discriminant(a, b, 5);
        assert!(
            collect_discriminants(&graph, &[(a, 4)]).is_empty(),
            "Length edge contributes no discriminants"
        );

        graph.set_assist_enabled(false);
        assert!(
            collect_discriminants(&graph, &[(a, 4)]).is_empty(),
            "assist disabled yields no discriminants"
        );
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
