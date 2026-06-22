use std::sync::Arc;

use hashbrown::HashMap;

use crate::stream_relation::AccessContext;

// ──────────────────────────────────────────────────────────────────────────────
// TaintTag
// ──────────────────────────────────────────────────────────────────────────────

/// A taint tag tracks which MMIO streams contributed to a value.
///
/// Representation is an inline `u64` bitmap for ≤62 streams.  Bit 62 is
/// reserved as a sentinel meaning "more streams taint this byte than fit in
/// the inline representation".  Bit 63 is currently unused.
///
/// For firmware with >62 streams in a single taint pass, a `Chunked` variant
/// is used (heap-allocated `Vec<u64>` where chunk[stream_index/64] has bit
/// `stream_index % 64` set).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TaintTag {
    #[default]
    Clean,
    Inline(u64),
    Chunked(Arc<Vec<u64>>),
}

/// Bit 62 of an Inline tag: indicates overflow beyond the inline capacity.
const OVERFLOW_BIT: u64 = 1 << 62;
/// Maximum stream index that fits in the inline representation without overflow.
const INLINE_CAPACITY: usize = 62;

impl TaintTag {
    pub fn is_clean(&self) -> bool {
        matches!(self, TaintTag::Clean)
    }

    /// Set bit `stream_index` in this tag.
    pub fn set_stream(&mut self, stream_index: usize) {
        match self {
            TaintTag::Clean => {
                if stream_index < INLINE_CAPACITY {
                    *self = TaintTag::Inline(1u64 << stream_index);
                } else {
                    let mut chunks = vec![0u64; (stream_index / 64) + 1];
                    chunks[stream_index / 64] |= 1u64 << (stream_index % 64);
                    *self = TaintTag::Chunked(Arc::new(chunks));
                }
            }
            TaintTag::Inline(bits) => {
                if stream_index < INLINE_CAPACITY {
                    *bits |= 1u64 << stream_index;
                } else {
                    // Promote to Chunked.
                    let old_bits = *bits & !OVERFLOW_BIT;
                    let needed = (stream_index / 64) + 1;
                    let mut chunks = vec![0u64; needed.max(1)];
                    // Preserve old inline bits in chunk 0.
                    chunks[0] = old_bits;
                    chunks[stream_index / 64] |= 1u64 << (stream_index % 64);
                    *self = TaintTag::Chunked(Arc::new(chunks));
                }
            }
            TaintTag::Chunked(chunks) => {
                let needed = (stream_index / 64) + 1;
                let v = Arc::make_mut(chunks);
                if v.len() < needed {
                    v.resize(needed, 0);
                }
                v[stream_index / 64] |= 1u64 << (stream_index % 64);
            }
        }
    }

    /// Compute the union (OR) of two tags.
    pub fn union(&self, other: &TaintTag) -> TaintTag {
        match (self, other) {
            (TaintTag::Clean, x) | (x, TaintTag::Clean) => x.clone(),
            (TaintTag::Inline(a), TaintTag::Inline(b)) => TaintTag::Inline(a | b),
            (TaintTag::Chunked(a), TaintTag::Chunked(b)) => {
                let len = a.len().max(b.len());
                let mut v = vec![0u64; len];
                for (i, x) in a.iter().enumerate() { v[i] |= x; }
                for (i, x) in b.iter().enumerate() { v[i] |= x; }
                TaintTag::Chunked(Arc::new(v))
            }
            (TaintTag::Inline(a), TaintTag::Chunked(b))
            | (TaintTag::Chunked(b), TaintTag::Inline(a)) => {
                let mut chunks: Vec<u64> = b.as_ref().clone();
                if chunks.is_empty() { chunks.push(0); }
                // Inline bits go into chunk 0 (inline is ≤62 bits, all in chunk 0).
                chunks[0] |= a & !OVERFLOW_BIT;
                TaintTag::Chunked(Arc::new(chunks))
            }
        }
    }

    /// Iterate over the stream indices set in this tag.
    pub fn iter_streams(&self) -> impl Iterator<Item = usize> + '_ {
        match self {
            TaintTag::Clean => TagIter::Clean,
            TaintTag::Inline(bits) => TagIter::Inline(*bits & !OVERFLOW_BIT),
            TaintTag::Chunked(chunks) => {
                // Seed `bits` with chunk 0; the iterator advances to later chunks as each is
                // exhausted.  (Seeding with 0 here would make the first `next()` skip chunk 0
                // entirely, dropping streams 0–63.)
                let v: &Vec<u64> = chunks.as_ref();
                TagIter::Chunked { chunks: v, chunk: 0, bits: v.first().copied().unwrap_or(0) }
            }
        }
    }
}

enum TagIter<'a> {
    Clean,
    Inline(u64),
    Chunked { chunks: &'a Vec<u64>, chunk: usize, bits: u64 },
}

impl<'a> Iterator for TagIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        match self {
            TagIter::Clean => None,
            TagIter::Inline(bits) => {
                if *bits == 0 { return None; }
                let pos = bits.trailing_zeros() as usize;
                *bits &= *bits - 1;
                Some(pos)
            }
            TagIter::Chunked { chunks, chunk, bits } => {
                loop {
                    if *bits != 0 {
                        let pos = bits.trailing_zeros() as usize;
                        *bits &= *bits - 1;
                        return Some(*chunk * 64 + pos);
                    }
                    *chunk += 1;
                    if *chunk >= chunks.len() { return None; }
                    *bits = chunks[*chunk];
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// StreamIndex
// ──────────────────���───────────────────────────────────────────────────────────

/// Maps `AccessContext`s to compact bit-indices used in TaintTags.
///
/// Keying by `(PC, address)` rather than address alone lets Phase B distinguish
/// taint from the same peripheral register read at different call sites.
pub struct StreamIndex {
    key_to_bit: HashMap<AccessContext, usize>,
    next_bit:   usize,
    /// Once this many contexts are indexed, new ones get the overflow bit instead.
    max_streams: usize,
}

impl StreamIndex {
    pub fn new(max_streams: usize) -> Self {
        Self { key_to_bit: HashMap::new(), next_bit: 0, max_streams }
    }

    pub fn default_capacity() -> Self {
        Self::new(256)
    }

    /// Get or assign the bit index for an access context.
    pub fn get_or_insert(&mut self, ctx: AccessContext) -> Option<usize> {
        if let Some(&idx) = self.key_to_bit.get(&ctx) {
            return Some(idx);
        }
        if self.next_bit >= self.max_streams {
            tracing::warn!(
                "TaintTag overflow: more than {} MMIO access contexts in a single taint pass; \
                 context (pc={:#x}, addr={:#x}) will use overflow sentinel",
                self.max_streams, ctx.pc, ctx.addr
            );
            return None; // Caller should set OVERFLOW_BIT
        }
        let idx = self.next_bit;
        self.next_bit += 1;
        self.key_to_bit.insert(ctx, idx);
        Some(idx)
    }

    /// Resolve a bit index back to its access context (for reporting results).
    pub fn bit_to_ctx(&self, bit: usize) -> Option<AccessContext> {
        self.key_to_bit.iter().find(|(_, &v)| v == bit).map(|(&k, _)| k)
    }

    /// Collect all access contexts whose bits are set in `tag`.
    pub fn contexts_in_tag(&self, tag: &TaintTag) -> Vec<AccessContext> {
        tag.iter_streams().filter_map(|bit| self.bit_to_ctx(bit)).collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ShadowState
// ──────────────────────────────────────────────────────────────────────────────

/// ARM Cortex-M has r0–r15 (16 registers) + xPSR (1) = 17 tracked registers.
const NUM_SHADOW_REGS: usize = 17;

/// Byte-granular shadow memory and shadow registers for dynamic taint tracking.
///
/// `ShadowState` is not attached to the emulator at startup; it is constructed
/// fresh for each taint pass (Phase B) and lives for the duration of one
/// instrumented execution.
pub struct ShadowState {
    /// Sparse map: physical address → taint tag.  Only tainted bytes are stored.
    pub mem:   HashMap<u32, TaintTag>,
    /// Shadow over ARM registers.  Index = pcode VarId of the register.
    pub regs:  Vec<TaintTag>,
    /// Maps StreamKey → bit index for this taint pass.
    pub index: StreamIndex,
}

impl ShadowState {
    pub fn new() -> Self {
        Self {
            mem:   HashMap::new(),
            regs:  vec![TaintTag::Clean; NUM_SHADOW_REGS],
            index: StreamIndex::default_capacity(),
        }
    }

    // ── Source injection ────────────────────────────────────────────────────

    fn tag_for_context(&mut self, ctx: AccessContext) -> TaintTag {
        match self.index.get_or_insert(ctx) {
            Some(bit_idx) => {
                let mut tag = TaintTag::Clean;
                tag.set_stream(bit_idx);
                tag
            }
            None => {
                // Overflow: return a tag with the overflow sentinel set.
                TaintTag::Inline(OVERFLOW_BIT)
            }
        }
    }

    /// Public accessor for the source tag of an access context (used by the Phase B engine
    /// when an MMIO read is observed and its loaded value must be tagged onto a register).
    pub fn source_tag(&mut self, ctx: AccessContext) -> TaintTag {
        self.tag_for_context(ctx)
    }

    // ── Shadow register access ──────────────────────────────────────────────

    pub fn reg_tag(&self, var_id: i16) -> TaintTag {
        if var_id < 0 || var_id as usize >= self.regs.len() {
            return TaintTag::Clean;
        }
        self.regs[var_id as usize].clone()
    }

    pub fn set_reg_tag(&mut self, var_id: i16, tag: TaintTag) {
        if var_id < 0 { return; }
        let idx = var_id as usize;
        if idx >= self.regs.len() {
            self.regs.resize(idx + 1, TaintTag::Clean);
        }
        self.regs[idx] = tag;
    }

    // ── Shadow memory access ────────────────────────────────────────────────

    pub fn mem_tag(&self, addr: u32) -> TaintTag {
        self.mem.get(&addr).cloned().unwrap_or(TaintTag::Clean)
    }

    pub fn mem_tag_range(&self, addr: u32, size: usize) -> TaintTag {
        (0..size as u32).fold(TaintTag::Clean, |acc, i| acc.union(&self.mem_tag(addr + i)))
    }

    pub fn set_mem_tag_range(&mut self, addr: u32, size: usize, tag: TaintTag) {
        for i in 0..size as u32 {
            if tag.is_clean() {
                self.mem.remove(&(addr + i));
            } else {
                self.mem.insert(addr + i, tag.clone());
            }
        }
    }

    // ── Call handling ───────────────────────────────────────────────────────

    /// Conservative CALL handling: kill taint on r0–r3 (accept under-tainting).
    /// Known-function summaries (memcpy/memset/…) would override this once callee
    /// identification (symbol/PLT resolution) is available; until then we kill.
    pub fn kill_call_regs(&mut self) {
        for reg_id in 0..4i16 {
            self.set_reg_tag(reg_id, TaintTag::Clean);
        }
    }
}

impl Default for ShadowState {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(pc: u64) -> AccessContext {
        AccessContext::new(pc, 0x5800_0000 + pc)
    }

    /// Inline → Chunked promotion preserves every previously-set stream bit (no loss at 64th).
    #[test]
    fn tag_promotes_inline_to_chunked_without_loss() {
        let mut tag = TaintTag::Clean;
        // Set bits across the inline boundary (INLINE_CAPACITY = 62) into chunked territory.
        for i in [0usize, 5, 61, 62, 70, 130] {
            tag.set_stream(i);
        }
        let got: std::collections::BTreeSet<usize> = tag.iter_streams().collect();
        let want: std::collections::BTreeSet<usize> = [0, 5, 61, 62, 70, 130].into_iter().collect();
        assert_eq!(got, want, "promotion to Chunked must preserve all bits");
        assert!(matches!(tag, TaintTag::Chunked(_)));
    }

    /// Union across the Inline/Chunked representation boundary is correct.
    #[test]
    fn tag_union_mixed_representations() {
        let mut a = TaintTag::Clean; // stays inline
        a.set_stream(3);
        a.set_stream(10);
        let mut b = TaintTag::Clean; // forced chunked
        b.set_stream(10);
        b.set_stream(100);

        let u = a.union(&b);
        let got: std::collections::BTreeSet<usize> = u.iter_streams().collect();
        assert_eq!(got, [3usize, 10, 100].into_iter().collect());
    }

    /// The StreamIndex assigns the overflow sentinel past capacity rather than silently
    /// dropping a stream: the first two contexts stay precisely attributable, the third
    /// gets only the sentinel bit (no precise attribution) and nothing panics.
    #[test]
    fn index_overflow_sets_sentinel_no_panic() {
        let mut shadow = ShadowState { index: StreamIndex::new(2), ..ShadowState::new() };
        // Two contexts fit the index; the third exceeds capacity.
        let t1 = shadow.source_tag(ctx(0x10));
        let t2 = shadow.source_tag(ctx(0x20));
        let t3 = shadow.source_tag(ctx(0x30)); // overflow

        assert_eq!(shadow.index.contexts_in_tag(&t1), vec![ctx(0x10)]);
        assert_eq!(shadow.index.contexts_in_tag(&t2), vec![ctx(0x20)]);
        // The overflowed context carries only the sentinel bit, so it has no precise
        // attribution — a known-bounded false negative, not a silent error.
        assert!(shadow.index.contexts_in_tag(&t3).is_empty());
    }

    /// An opaque CALL kills r0–r3 taint (plan Fix 4: kill over union).
    #[test]
    fn call_kill_clears_arg_registers() {
        let mut shadow = ShadowState::new();
        for r in 0..4i16 {
            shadow.set_reg_tag(r, {
                let mut t = TaintTag::Clean;
                t.set_stream(r as usize);
                t
            });
        }
        shadow.set_reg_tag(7, {
            let mut t = TaintTag::Clean;
            t.set_stream(7);
            t
        });
        shadow.kill_call_regs();
        for r in 0..4i16 {
            assert!(shadow.reg_tag(r).is_clean(), "r{r} must be killed by the call");
        }
        // Callee-saved register outside r0–r3 is preserved.
        assert!(!shadow.reg_tag(7).is_clean(), "r7 must survive the call");
    }
}
