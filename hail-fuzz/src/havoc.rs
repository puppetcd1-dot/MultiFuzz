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

/// Probability that mutating a stream also co-mutates one of its dependency
/// partners (governor/dependent) in the same round, so coupled streams evolve
/// together instead of independently.
const COUPLED_MUTATION_PROB: f64 = 0.25;

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
    /// For each source stream, the `Length`-governed targets with a fitted relation,
    /// used to keep the target sized to the trip count implied by the source value.
    length_rel: HashMap<StreamKey, Vec<(StreamKey, Relation)>>,

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
            // Must have at least one non-empty stream.
            return Err(StageExit::Skip);
        }

        let (coupled, length_rel) = build_coupling(&fuzzer.relation_graph, &streams);
        let stream_distr = get_stream_weights(fuzzer, id, &streams);
        let mutator = HavocMutator::new();
        let attempts = calculate_energy(fuzzer) as u32;
        let log2_max_mutations = log2_max_mutations(fuzzer);
        let max_mutations = max_mutations(fuzzer);

        fuzzer.corpus[id].metadata.havoc_rounds += 1;

        tracing::trace!(
            "[{id}] havoc for {attempts} attempts with {} max mutations",
            2_u64.pow(log2_max_mutations)
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
    #[allow(unused)]
    fn havoc_v1(&mut self, fuzzer: &mut Fuzzer) {
        let mut mutations = fuzzer.rng.gen_range(1..=self.max_mutations);
        while mutations > 0 {
            // Select a random stream to mutate.
            let (key, _) = self.streams[self.stream_distr.sample(&mut fuzzer.rng)];

            // Consume some proportion of the total number of mutations on the current stream.
            let num_mutations = fuzzer.rng.gen_range(1..=mutations);
            mutations -= num_mutations;
            self.mutate_stream(fuzzer, key, num_mutations);

            // Coupled mutation: with some probability also evolve one of `key`'s
            // dependency partners so related streams move together rather than
            // independently.  Picking the partner up-front releases the `coupled`
            // borrow before the `&mut self` mutation call.
            let partner = self.coupled.get(&key).filter(|r| !r.is_empty()).and_then(|r| {
                fuzzer.rng.gen_bool(COUPLED_MUTATION_PROB).then(|| r[fuzzer.rng.gen_range(0..r.len())])
            });
            if let Some(key2) = partner {
                self.mutate_stream(fuzzer, key2, 1 + num_mutations / 2);
                // If `key` governs `key2` by a fitted Length relation, keep `key2`
                // sized to the trip count implied by `key`'s (now mutated) value.
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

    #[allow(unused)]
    fn havoc_v2(&mut self, fuzzer: &mut Fuzzer) {
        let Some(input_id) = fuzzer.input_id
        else {
            return;
        };
        let max_find_gap = fuzzer.corpus[input_id].metadata.max_find_gap;

        let data = &mut fuzzer.state.input;
        for &(key, _) in &self.streams {
            // Decided whether this stream should be mutated.
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

/// Determine the maximum number of mutations to try. This number increases the longer it takes to
/// find new inputs.
///
/// @todo: these ranges were selected to be similar the havoc stacking factor used by AFL++, but it
/// is possible that there are better values.
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
    // match mutation {
    //     Mutation::BitFlip => 1,
    //     Mutation::IncDec => 1,

    //     Mutation::ReplaceByte => 4,
    //     Mutation::InsertByte => 4,
    //     Mutation::Insert4 => 4,

    //     Mutation::InterestingValue => 2,
    //     Mutation::DictReplace => 4,
    //     Mutation::DictInsert => 4,

    //     Mutation::StreamSplice => 4,
    //     Mutation::InnerSplice => 4,
    //     Mutation::RandomSplice => 1,

    //     Mutation::RemoveByte => 2,
    //     Mutation::Remove4 => 1,
    //     Mutation::RemoveRegion => 1,
    // }
}

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

/// Build the per-stream coupling maps from the relation graph, restricted to streams
/// that are present and non-empty in this input.  Returns `(coupled, length_rel)`:
///   * `coupled[k]`   — partner streams (dependents of `k` ∪ governors of `k`) to
///                      co-mutate when `k` is mutated;
///   * `length_rel[s]`— `(target, relation)` for each `Length` edge `s → target`
///                      that carries a fitted relation, used for length preservation.
///
/// Returns empty maps when mutation assistance is disabled (ablation arm), so the
/// stage falls back to plain independent mutation.
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

/// Interpret the leading bytes of a stream as a little-endian unsigned integer (the
/// firmware's view of a count/length register).  Reads up to 8 bytes.
fn le_value(bytes: &[u8]) -> u64 {
    let n = bytes.len().min(8);
    let mut v = 0u64;
    for (i, &b) in bytes[..n].iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream_relation::{AccessContext, EdgeKind, RelKind, Relation};

    fn ac(pc: u64, addr: StreamKey) -> AccessContext {
        AccessContext::new(pc, addr)
    }

    /// Coupling links a confirmed dependency both ways (source↔target) but only for
    /// streams that are actually present in the input, and surfaces the fitted
    /// relation of a Length edge for length preservation.
    #[test]
    fn coupling_maps_link_present_partners_and_relations() {
        let a = 0x5800_0000; // governor (source)
        let b = 0x5800_0004; // governed (target)
        let absent = 0x5800_0008;

        let mut graph = StreamRelationGraph::new();
        graph.add_structural_candidates(ac(0x20, b), &[(ac(0x10, a), EdgeKind::Control)]);
        // A Length edge A→B with a fitted relation (confirmed by Phase B).
        graph.confirm_edge(ac(0x10, a), ac(0x20, b), EdgeKind::Length,
            Some(Relation { kind: RelKind::Identity }));
        // An edge to an absent stream must be ignored.
        graph.add_structural_candidates(ac(0x30, absent), &[(ac(0x10, a), EdgeKind::Control)]);

        let streams = vec![(a, 4usize), (b, 4usize)]; // `absent` deliberately omitted
        let (coupled, length_rel) = build_coupling(&graph, &streams);

        assert_eq!(coupled.get(&a).map(|v| v.as_slice()), Some([b].as_slice()),
            "source couples to its present target only (absent stream excluded)");
        assert_eq!(coupled.get(&b).map(|v| v.as_slice()), Some([a].as_slice()),
            "target couples back to its governor");
        assert!(length_rel.get(&a).is_some(), "Length relation surfaced for the source");
    }

    /// Assistance disabled (ablation arm) yields no coupling, falling back to plain mutation.
    #[test]
    fn coupling_empty_when_assist_disabled() {
        let a = 0x5800_0000;
        let b = 0x5800_0004;
        let mut graph = StreamRelationGraph::new();
        graph.add_structural_candidates(ac(0x20, b), &[(ac(0x10, a), EdgeKind::Control)]);
        graph.set_assist_enabled(false);

        let (coupled, length_rel) = build_coupling(&graph, &[(a, 4), (b, 4)]);
        assert!(coupled.is_empty() && length_rel.is_empty());
    }

    #[test]
    fn le_value_reads_little_endian_prefix() {
        assert_eq!(le_value(&[0x04, 0x00, 0x00, 0x00]), 4);
        assert_eq!(le_value(&[0x00, 0x01]), 256);
        assert_eq!(le_value(&[]), 0);
    }
}
