use hashbrown::HashMap;

use crate::input::StreamKey;

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
    Chunked(Box<Vec<u64>>),
}

/// Bit 62 of an Inline tag: indicates overflow beyond the inline capacity.
const OVERFLOW_BIT: u64 = 1 << 62;
/// Maximum stream index that fits in the inline representation without overflow.
const INLINE_CAPACITY: usize = 62;

impl TaintTag {
    pub fn clean() -> Self {
        Self::Clean
    }

    pub fn is_clean(&self) -> bool {
        matches!(self, TaintTag::Clean)
    }

    /// True if the inline overflow sentinel is set (stream count exceeded inline capacity).
    pub fn has_overflow(&self) -> bool {
        match self {
            TaintTag::Inline(bits) => bits & OVERFLOW_BIT != 0,
            TaintTag::Chunked(_) => false, // Chunked has no overflow limit up to MAX_STREAMS
            TaintTag::Clean => false,
        }
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
                    *self = TaintTag::Chunked(Box::new(chunks));
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
                    *self = TaintTag::Chunked(Box::new(chunks));
                }
            }
            TaintTag::Chunked(chunks) => {
                let needed = (stream_index / 64) + 1;
                if chunks.len() < needed {
                    chunks.resize(needed, 0);
                }
                chunks[stream_index / 64] |= 1u64 << (stream_index % 64);
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
                TaintTag::Chunked(Box::new(v))
            }
            (TaintTag::Inline(a), TaintTag::Chunked(b))
            | (TaintTag::Chunked(b), TaintTag::Inline(a)) => {
                let mut chunks = b.as_ref().clone();
                if chunks.is_empty() { chunks.push(0); }
                // Inline bits go into chunk 0 (inline is ≤62 bits, all in chunk 0).
                chunks[0] |= a & !OVERFLOW_BIT;
                TaintTag::Chunked(Box::new(chunks))
            }
        }
    }

    /// Check whether this tag has any bit set in common with `other`.
    pub fn overlaps(&self, other: &TaintTag) -> bool {
        !self.union(other).is_clean()
    }

    /// Iterate over the stream indices set in this tag.
    pub fn iter_streams(&self) -> impl Iterator<Item = usize> + '_ {
        match self {
            TaintTag::Clean => TagIter::Clean,
            TaintTag::Inline(bits) => TagIter::Inline(*bits & !OVERFLOW_BIT),
            TaintTag::Chunked(chunks) => TagIter::Chunked { chunks, chunk: 0, bits: 0 },
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

/// Maps StreamKeys to compact bit-indices used in TaintTags.
pub struct StreamIndex {
    key_to_bit: HashMap<StreamKey, usize>,
    next_bit:   usize,
    /// Once this many streams are indexed, new streams get the overflow bit instead.
    max_streams: usize,
}

impl StreamIndex {
    pub fn new(max_streams: usize) -> Self {
        Self { key_to_bit: HashMap::new(), next_bit: 0, max_streams }
    }

    pub fn default_capacity() -> Self {
        Self::new(256)
    }

    /// Get or assign the bit index for a stream key.
    pub fn get_or_insert(&mut self, key: StreamKey) -> Option<usize> {
        if let Some(&idx) = self.key_to_bit.get(&key) {
            return Some(idx);
        }
        if self.next_bit >= self.max_streams {
            tracing::warn!(
                "TaintTag overflow: more than {} MMIO streams in a single taint pass; \
                 new stream {:#x} will use overflow sentinel",
                self.max_streams, key
            );
            return None; // Caller should set OVERFLOW_BIT
        }
        let idx = self.next_bit;
        self.next_bit += 1;
        self.key_to_bit.insert(key, idx);
        Some(idx)
    }

    /// Resolve a bit index back to a stream key (for reporting results).
    pub fn bit_to_key(&self, bit: usize) -> Option<StreamKey> {
        self.key_to_bit.iter().find(|(_, &v)| v == bit).map(|(&k, _)| k)
    }

    /// Collect all stream keys whose bits are set in `tag`.
    pub fn streams_in_tag(&self, tag: &TaintTag) -> Vec<StreamKey> {
        tag.iter_streams().filter_map(|bit| self.bit_to_key(bit)).collect()
    }

    pub fn len(&self) -> usize {
        self.next_bit
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

    /// Mark `size` bytes at `phys_addr` as tainted by `stream_key`.
    /// Called at each MMIO read (Phase B source hook in `mmio.rs`).
    pub fn taint_mmio_read(&mut self, stream_key: StreamKey, phys_addr: u32, size: usize) {
        let tag = self.tag_for_stream(stream_key);
        for i in 0..size as u32 {
            self.mem.insert(phys_addr + i, tag.clone());
        }
    }

    fn tag_for_stream(&mut self, key: StreamKey) -> TaintTag {
        match self.index.get_or_insert(key) {
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

    // ── Propagation helpers (called from CodeInjector hooks) ────────────────

    /// Propagate taint through a unary op: `dst_tag = tag(src)`.
    pub fn propagate_unary(&mut self, dst_id: i16, src: pcode::VarNode) {
        let tag = self.reg_tag(src.id);
        self.set_reg_tag(dst_id, tag);
    }

    /// Propagate taint through a binary op: `dst_tag = tag(a) ∪ tag(b)`.
    pub fn propagate_binary(
        &mut self,
        dst_id: i16,
        a: pcode::VarNode,
        b: pcode::Value,
    ) {
        let tag_a = self.reg_tag(a.id);
        let tag_b = match b {
            pcode::Value::Var(vn) => self.reg_tag(vn.id),
            pcode::Value::Const(..) => TaintTag::Clean,
        };
        self.set_reg_tag(dst_id, tag_a.union(&tag_b));
    }

    /// Propagate taint through a LOAD: `dst_tag = ∪ mem_tag[runtime_addr..+size]`.
    pub fn propagate_load(&mut self, dst_id: i16, runtime_addr: u32, size: usize) {
        let tag = self.mem_tag_range(runtime_addr, size);
        self.set_reg_tag(dst_id, tag);
    }

    /// Propagate taint through a STORE: `mem_tag[addr..+size] = tag(val) ∪ tag(addr_reg)`.
    pub fn propagate_store(
        &mut self,
        runtime_addr: u32,
        size: usize,
        val: pcode::VarNode,
        addr_reg: pcode::VarNode,
    ) {
        let tag_val  = self.reg_tag(val.id);
        let tag_addr = self.reg_tag(addr_reg.id);
        let combined = tag_val.union(&tag_addr);
        self.set_mem_tag_range(runtime_addr, size, combined);
    }

    /// Conservative CALL handling: kill taint on r0–r3 (accept under-tainting).
    /// Known-function summaries should override this before it is called.
    pub fn kill_call_regs(&mut self) {
        for reg_id in 0..4i16 {
            self.set_reg_tag(reg_id, TaintTag::Clean);
        }
    }

    /// Known-function summary for memcpy(dst=r0, src=r1, n=r2):
    /// propagate shadow_mem[src..src+n] → shadow_mem[dst..dst+n].
    pub fn summary_memcpy(&mut self, dst: u32, src: u32, n: u32) {
        for i in 0..n {
            let tag = self.mem_tag(src + i);
            if tag.is_clean() {
                self.mem.remove(&(dst + i));
            } else {
                self.mem.insert(dst + i, tag);
            }
        }
        // r0 (return = dst pointer) carries no value taint.
        self.set_reg_tag(0, TaintTag::Clean);
    }

    /// Known-function summary for memset(dst=r0, c=r1, n=r2):
    /// propagate tag(c) → shadow_mem[dst..dst+n].
    pub fn summary_memset(&mut self, dst: u32, n: u32, fill_tag: TaintTag) {
        self.set_mem_tag_range(dst, n as usize, fill_tag);
        self.set_reg_tag(0, TaintTag::Clean);
    }

    // ── Sink checks ─────────────────────────────────────────────────────────

    /// Returns the set of StreamKeys that taint the given register.
    pub fn tainted_streams_in_reg(&self, var_id: i16) -> Vec<StreamKey> {
        self.index.streams_in_tag(&self.reg_tag(var_id))
    }

    /// Returns the set of StreamKeys that taint a memory range.
    pub fn tainted_streams_in_mem(&self, addr: u32, size: usize) -> Vec<StreamKey> {
        self.index.streams_in_tag(&self.mem_tag_range(addr, size))
    }

    /// True if the tag has the overflow sentinel — attribution is uncertain.
    pub fn is_overflowed_reg(&self, var_id: i16) -> bool {
        self.reg_tag(var_id).has_overflow()
    }
}

impl Default for ShadowState {
    fn default() -> Self { Self::new() }
}
