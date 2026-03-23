# Copyright © 2023-2025 Apple Inc.

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .loading import _load_proj_experts, _load_experts



class ExpertCache:
    """Per-layer LCP (Least Critical Priority) cache for expert weights.

    Shared by all 3 projections (gate/up/down) within one MoE layer.
    Eviction priority: P = mu * 0.25^(nu / 128) where mu = activation count,
    nu = steps since last activation. Lower P means evicted first.

    Step tracking: SwitchGLU calls up_proj -> gate_proj -> down_proj sequentially,
    so _proj_count cycles 0->1->2->0. Step increments on the first call (count == 0
    after reset), and frequency/recency are updated once per step.
    """
    __slots__ = ('entries', 'frequency', 'last_active', 'step',
                 '_proj_count', 'capacity', 'hits', 'misses',
                 'all_seen')

    def __init__(self, capacity: int):
        self.entries: dict[int, dict[str, tuple]] = {}
        self.frequency: dict[int, int] = {}
        self.last_active: dict[int, int] = {}
        self.step = 0
        self._proj_count = 0
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.all_seen: set[int] = set()

    def projection_called(self, expert_ids: np.ndarray) -> None:
        """Called once per projection. Increments step every 3rd call."""
        if self._proj_count == 0:
            self.step += 1
            for eid in expert_ids:
                eid = int(eid)
                self.frequency[eid] = self.frequency.get(eid, 0) + 1
                self.last_active[eid] = self.step
                self.all_seen.add(eid)
        self._proj_count = (self._proj_count + 1) % 3

    def lookup(self, expert_id: int, proj_name: str):
        """Return cached (w, s, b) or None."""
        entry = self.entries.get(expert_id)
        if entry is not None:
            return entry.get(proj_name)
        return None

    def put(self, expert_id: int, proj_name: str, w, s, b) -> None:
        if expert_id not in self.entries:
            self.entries[expert_id] = {}
        self.entries[expert_id][proj_name] = (w, s, b)

    def evict_if_needed(self, protected: set[int]) -> None:
        """Evict lowest-priority experts until at or under capacity."""
        while len(self.entries) > self.capacity:
            worst_id = None
            worst_p = float('inf')
            for eid in self.entries:
                if eid in protected:
                    continue
                p = self._priority(eid)
                if p < worst_p:
                    worst_p = p
                    worst_id = eid
            if worst_id is None:
                break
            del self.entries[worst_id]
            del self.frequency[worst_id]
            del self.last_active[worst_id]

    def _priority(self, expert_id: int) -> float:
        mu = self.frequency.get(expert_id, 0)
        nu = self.step - self.last_active.get(expert_id, 0)
        return mu * (0.25 ** (nu / 128))


class LazyQuantizedSwitchLinear(nn.Module):
    """Drop-in replacement for QuantizedSwitchLinear that loads experts on demand.

    On each forward call: loads a fresh lazy ref from the safetensors shard,
    slices only the needed experts, and lets the full tensor be freed after
    evaluation. This avoids the problem where evaluating a slice from a lazy
    tensor permanently materializes the full source tensor in Metal memory.
    """

    def __init__(self, shard_path: str, key_prefix: str, group_size: int,
                 bits: int, mode: str, shard_map: dict[str, str] | None = None):
        super().__init__()
        self._shard_path = shard_path
        self._key_prefix = key_prefix
        self._shard_map = shard_map
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.freeze()

    def _load_expert_subset(self, expert_ids: mx.array):
        """Load only the needed experts from the safetensors shard."""
        shard = mx.load(self._shard_path)
        return _load_proj_experts(shard, self._key_prefix, expert_ids,
                                  shard_map=self._shard_map)

    def __call__(self, x, indices, sorted_indices=False):
        mx.eval(indices)
        indices_np = np.asarray(indices.reshape(-1))
        unique_ids = np.unique(indices_np)

        w, s, b = self._load_expert_subset(mx.array(unique_ids))

        remap = np.empty(int(unique_ids[-1]) + 1, dtype=np.int32)
        remap[unique_ids] = np.arange(len(unique_ids), dtype=np.int32)
        remapped = mx.array(remap[indices_np].reshape(indices.shape))

        return mx.gather_qmm(
            x,
            w,
            s,
            b,
            rhs_indices=remapped,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


class CachedQuantizedSwitchLinear(nn.Module):
    """Expert loader with per-layer LCP caching. Drop-in for QuantizedSwitchLinear.

    Cache hits serve weights from Metal memory. Misses batch-load from the
    safetensors shard, eval once, then insert individually into the cache.
    """

    def __init__(self, shard_path: str, key_prefix: str, group_size: int,
                 bits: int, mode: str, proj_name: str,
                 cache: ExpertCache,
                 shard_map: dict[str, str] | None = None):
        super().__init__()
        self._shard_path = shard_path
        self._key_prefix = key_prefix
        self._shard_map = shard_map
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._proj_name = proj_name
        self._cache = cache
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        mx.eval(indices)
        indices_np = np.asarray(indices.reshape(-1))
        unique_ids = np.unique(indices_np)

        self._cache.projection_called(unique_ids)

        hit_ids = []
        miss_ids = []
        for eid in unique_ids:
            eid = int(eid)
            if self._cache.lookup(eid, self._proj_name) is not None:
                hit_ids.append(eid)
                self._cache.hits += 1
            else:
                miss_ids.append(eid)
                self._cache.misses += 1

        if miss_ids:
            miss_arr = mx.array(miss_ids)
            shard = mx.load(self._shard_path)
            w_batch, s_batch, b_batch = _load_proj_experts(shard, self._key_prefix, miss_arr,
                                                            shard_map=self._shard_map)
            mx.eval(w_batch, s_batch) if b_batch is None else mx.eval(w_batch, s_batch, b_batch)

            for i, eid in enumerate(miss_ids):
                self._cache.put(
                    eid, self._proj_name,
                    w_batch[i], s_batch[i],
                    b_batch[i] if b_batch is not None else None,
                )

        protected = set(int(e) for e in unique_ids)
        self._cache.evict_if_needed(protected)

        all_ids = sorted(int(e) for e in unique_ids)
        ws, ss, bs = [], [], []
        has_bias = None
        for eid in all_ids:
            w, s, b = self._cache.lookup(eid, self._proj_name)
            ws.append(w)
            ss.append(s)
            if has_bias is None:
                has_bias = b is not None
            if has_bias:
                bs.append(b)

        w_cat = mx.stack(ws)
        s_cat = mx.stack(ss)
        b_cat = mx.stack(bs) if has_bias else None

        unique_sorted = np.array(all_ids, dtype=np.int32)
        remap = np.empty(unique_sorted[-1] + 1, dtype=np.int32)
        remap[unique_sorted] = np.arange(len(unique_sorted), dtype=np.int32)
        remapped = mx.array(remap[indices_np].reshape(indices.shape))

        return mx.gather_qmm(
            x,
            w_cat,
            s_cat,
            b_cat,
            rhs_indices=remapped,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


class PredictiveExpertCache:
    """Per-layer cache with GPU-resident weight tensors and lookup table.

    Pre-loads a subset of experts into Metal memory at startup. During the
    forward pass, a lookup table remaps global expert IDs (0-511) to cache
    slots (0 to C-1) entirely on GPU -- no mx.eval needed. Uncached experts
    map to slot 0 (fallback).

    Supports dynamic updates between tokens: captures router indices during
    forward pass, then swaps cold experts for newly-requested ones.
    """
    __slots__ = ('capacity', 'num_experts', 'lookup', 'hit_mask',
                 'weights', 'scales', 'biases',
                 'cached_ids', 'cached_set',
                 'frequency', 'last_active', 'step',
                 '_indices_buffer',
                 '_shard_paths', '_key_prefixes', '_shard_map',
                 '_st_map',
                 'total_requests', 'total_fallbacks',
                 'pinned_set',
                 # Double buffering: Buffer A and Buffer B
                 'weights_A', 'scales_A', 'biases_A', 'cached_ids_A',
                 'weights_B', 'scales_B', 'biases_B', 'cached_ids_B',
                 '_active_buffer',
                 # Prefetch state
                 '_prefetch_buffer', '_pending_prefetch',
                 '_prefetch_in_progress')

    def __init__(self, capacity: int, num_experts: int = 512):
        self.capacity = capacity
        self.num_experts = num_experts
        self.weights: dict[str, mx.array] = {}
        self.scales: dict[str, mx.array] = {}
        self.biases: dict[str, mx.array | None] = {}
        self.lookup: mx.array | None = None
        self.hit_mask: mx.array | None = None
        self.cached_ids: list[int] = []
        self.cached_set: set[int] = set()
        self.frequency: dict[int, int] = {}
        self.last_active: dict[int, int] = {}
        self.step: int = 0
        self._indices_buffer: list[mx.array] = []
        self._shard_paths: dict[str, str] = {}
        self._key_prefixes: dict[str, str] = {}
        self._shard_map: dict[str, str] | None = None
        self._st_map = None
        self.total_requests: int = 0
        self.total_fallbacks: int = 0
        self.pinned_set: set[int] = set()
        # Double buffering: Buffer A and Buffer B
        self.weights_A: dict[str, mx.array] = {}
        self.scales_A: dict[str, mx.array] = {}
        self.biases_A: dict[str, mx.array | None] = {}
        self.cached_ids_A: list[int] = []  # Expert IDs in Buffer A
        self.weights_B: dict[str, mx.array] = {}
        self.scales_B: dict[str, mx.array] = {}
        self.biases_B: dict[str, mx.array | None] = {}
        self.cached_ids_B: list[int] = []  # Expert IDs in Buffer B
        self._active_buffer = 'A'
        # Prefetch state
        self._prefetch_buffer = 'B'  # Target buffer for prefetch
        self._pending_prefetch: list[int] = []
        self._prefetch_in_progress = False

    def build_lookup(self, cached_ids: list[int]) -> None:
        """Build GPU-resident lookup table and hit mask from cached expert IDs.

        Uncached IDs map to slot 0 (fallback) in the lookup table and 0.0 in
        the hit mask. Must be called after populating the weight tensors.
        """
        self.cached_ids = list(cached_ids)
        self.cached_set = set(cached_ids)
        for eid in cached_ids:
            self.frequency.setdefault(eid, 1)
            self.last_active.setdefault(eid, 0)
        self.rebuild_lookup()

    def rebuild_lookup(self) -> None:
        """Rebuild lookup table and hit mask from current cached_ids."""
        lookup_np = np.zeros(self.num_experts, dtype=np.int32)
        hit_np = np.zeros(self.num_experts, dtype=np.float32)
        for slot, eid in enumerate(self.cached_ids):
            lookup_np[eid] = slot
            hit_np[eid] = 1.0
        self.lookup = mx.array(lookup_np)
        self.hit_mask = mx.array(hit_np)

    def remap(self, indices: mx.array) -> mx.array:
        """Map global expert IDs to cache slots. Pure mx.array op, no eval."""
        return self.lookup[indices]

    def get_weights(self, proj_name: str) -> mx.array:
        """Get weights from current active buffer."""
        if self._active_buffer == 'A':
            return self.weights_A.get(proj_name) or self.weights.get(proj_name)
        return self.weights_B.get(proj_name) or self.weights.get(proj_name)

    def get_scales(self, proj_name: str) -> mx.array:
        """Get scales from current active buffer."""
        if self._active_buffer == 'A':
            return self.scales_A.get(proj_name) or self.scales.get(proj_name)
        return self.scales_B.get(proj_name) or self.scales.get(proj_name)

    def get_biases(self, proj_name: str) -> mx.array | None:
        """Get biases from current active buffer."""
        if self._active_buffer == 'A':
            return self.biases_A.get(proj_name) if proj_name in self.biases_A else self.biases.get(proj_name)
        return self.biases_B.get(proj_name) if proj_name in self.biases_B else self.biases.get(proj_name)

    def swap_buffers(self) -> None:
        """Swap active buffer (A <-> B)."""
        self._active_buffer = 'B' if self._active_buffer == 'A' else 'A'

    def predict_next_experts(self) -> list[int]:
        """Predict next layer experts using current layer requests."""
        if not self._indices_buffer:
            return []
        recent = self._indices_buffer[-1]
        flat = np.asarray(recent.reshape(-1))
        return [int(x) for x in np.unique(flat)]

    def prefetch_async(self, expert_ids: list[int]) -> None:
        """Initiate prefetch of experts to prefetch buffer.
        
        This loads experts into the inactive buffer (Buffer B if active is A).
        The data is loaded synchronously but happens during the token gap,
        potentially overlapping with GPU computation.
        """
        if self._prefetch_in_progress:
            return  # Already prefetching
        
        # Filter to experts not in current active buffer
        current_buffer_set = set()
        if self._active_buffer == 'A':
            for proj in ('gate_proj', 'up_proj', 'down_proj'):
                if proj in self.weights_A:
                    current_buffer_set.update(self.cached_ids[:len(self.weights_A)])
        else:
            for proj in ('gate_proj', 'up_proj', 'down_proj'):
                if proj in self.weights_B:
                    current_buffer_set.update(self.cached_ids[:len(self.weights_B)])
        
        missing = [eid for eid in expert_ids if eid not in self.cached_set]
        if not missing or not self._shard_paths:
            return
        
        # Limit prefetch count
        MAX_PREFETCH = 8
        prefetch_ids = missing[:MAX_PREFETCH]
        
        # Determine target buffer
        target_weights = self.weights_A if self._prefetch_buffer == 'A' else self.weights_B
        target_scales = self.scales_A if self._prefetch_buffer == 'A' else self.scales_B
        target_biases = self.biases_A if self._prefetch_buffer == 'A' else self.biases_B
        
        # If target buffer is empty, we need to allocate slots
        if not target_weights:
            # Use slot indices 0 to len(prefetch_ids)-1 for prefetch
            slot_indices = list(range(len(prefetch_ids)))
            new_eids = mx.array(prefetch_ids)
            
            for proj_name in ('gate_proj', 'up_proj', 'down_proj'):
                if proj_name not in self._shard_paths:
                    continue
                    
                shard_path = self._shard_paths[proj_name]
                key_prefix = self._key_prefixes[proj_name]
                shard = mx.load(shard_path) if self._st_map is None else None
                new_w, new_s, new_b = _load_experts(
                    key_prefix, new_eids,
                    shard=shard, shard_map=self._shard_map,
                    st_map=self._st_map
                )
                del shard
                
                if new_b is None:
                    mx.eval(new_w, new_s)
                else:
                    mx.eval(new_w, new_s, new_b)
                
                target_weights[proj_name] = new_w
                target_scales[proj_name] = new_s
                if new_b is not None:
                    target_biases[proj_name] = new_b
            
            # Update cached_ids for prefetch buffer
            if self._prefetch_buffer == 'A':
                self.weights_A = target_weights
                self.scales_A = target_scales
                self.biases_A = target_biases
                # Update cached_ids_A tracking
                self.cached_ids_A = list(prefetch_ids)
            else:
                self.weights_B = target_weights
                self.scales_B = target_scales
                self.biases_B = target_biases
                self.cached_ids_B = list(prefetch_ids)
            
            self._prefetch_in_progress = True
            self._pending_prefetch = prefetch_ids

    def check_prefetch_and_swap(self) -> bool:
        """Check if prefetch completed and swap buffers if ready.
        
        Returns True if buffers were swapped.
        """
        if not self._prefetch_in_progress or not self._pending_prefetch:
            return False
        
        # Check if pending experts are loaded in prefetch buffer
        prefetch_target = self._prefetch_buffer
        weights = self.weights_A if prefetch_target == 'A' else self.weights_B
        
        # Simple check: if prefetch buffer has any data, consider it ready
        if weights:
            self.swap_buffers()
            self._prefetch_in_progress = False
            self._pending_prefetch = []
            return True
        
        return False

    def _lcp_priority(self, eid: int) -> float:
        mu = self.frequency.get(eid, 0)
        nu = self.step - self.last_active.get(eid, 0)
        return mu * (0.25 ** (nu / 128))

    def update(self) -> dict:
        """Process buffered indices and swap cold experts for missed ones.

        Call between tokens. Skips the last buffered entry (in-flight due
        to async_eval double-buffering).

        Returns:
            Dict with ``"swaps"``, ``"fallbacks"``, and ``"requests"`` counts.
        """
        if len(self._indices_buffer) < 2:
            return {"swaps": 0, "fallbacks": 0, "requests": 0}

        to_process = self._indices_buffer[:-1]
        self._indices_buffer = self._indices_buffer[-1:]

        all_requested: set[int] = set()
        for indices in to_process:
            flat = np.asarray(indices.reshape(-1))
            unique = set(int(x) for x in np.unique(flat))
            all_requested |= unique

        self.step += 1
        for eid in all_requested:
            self.frequency[eid] = self.frequency.get(eid, 0) + 1
            self.last_active[eid] = self.step

        misses = all_requested - self.cached_set
        n_requests = len(all_requested)
        n_fallbacks = len(misses)
        self.total_requests += n_requests
        self.total_fallbacks += n_fallbacks

        if not misses or not self._shard_paths:
            return {"swaps": 0, "fallbacks": n_fallbacks, "requests": n_requests}

        evict_candidates = [
            (self._lcp_priority(eid), slot, eid)
            for slot, eid in enumerate(self.cached_ids)
            if eid not in all_requested and eid not in self.pinned_set
        ]
        evict_candidates.sort()

        swaps: list[tuple[int, int, int]] = []
        miss_list = sorted(misses)
        for new_eid in miss_list:
            if not evict_candidates:
                break
            _, slot, old_eid = evict_candidates.pop(0)
            swaps.append((slot, old_eid, new_eid))

        if not swaps:
            return {"swaps": 0, "fallbacks": n_fallbacks, "requests": n_requests}

        # Cap swaps per layer to bound transient memory from shard materialization.
        # Each swap temporarily materializes ~336 MB (full source tensor).
        MAX_SWAPS = 10
        swaps = swaps[:MAX_SWAPS]

        new_eids = mx.array([new_eid for _, _, new_eid in swaps])
        slot_indices = mx.array([slot for slot, _, _ in swaps])
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            shard_path = self._shard_paths[proj_name]
            key_prefix = self._key_prefixes[proj_name]
            shard = mx.load(shard_path) if self._st_map is None else None
            new_w, new_s, new_b = _load_experts(key_prefix, new_eids,
                                                 shard=shard, shard_map=self._shard_map,
                                                 st_map=self._st_map)
            del shard

            if new_b is None:
                mx.eval(new_w, new_s)
            else:
                mx.eval(new_w, new_s, new_b)

            w = self.weights.pop(proj_name)
            w[slot_indices] = new_w
            self.weights[proj_name] = w

            s = self.scales.pop(proj_name)
            s[slot_indices] = new_s
            self.scales[proj_name] = s

            if self.biases[proj_name] is not None and new_b is not None:
                b = self.biases.pop(proj_name)
                b[slot_indices] = new_b
                self.biases[proj_name] = b

        mx.clear_cache()

        for slot, old_eid, new_eid in swaps:
            self.cached_set.discard(old_eid)
            self.cached_set.add(new_eid)
            self.cached_ids[slot] = new_eid
            self.frequency.pop(old_eid, None)
            self.last_active.pop(old_eid, None)

        self.rebuild_lookup()
        mx.eval(self.lookup, self.hit_mask)

        return {"swaps": len(swaps), "fallbacks": n_fallbacks, "requests": n_requests}


class PredictiveCachedSwitchLinear(nn.Module):
    """Zero-eval expert dispatch using pre-loaded weights and GPU lookup table.

    The forward pass stays entirely lazy -- indices are remapped via a
    pre-built lookup table on GPU, and gather_qmm uses pre-loaded weight
    tensors already in Metal memory. No mx.eval until the output token.

    Captures router indices for dynamic cache updates between tokens.
    Only the first projection (up_proj, called first by SwitchGLU) captures
    indices to avoid triple-buffering the same data.
    """

    def __init__(self, group_size: int, bits: int, mode: str,
                 proj_name: str, cache: PredictiveExpertCache):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._proj_name = proj_name
        self._cache = cache
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        if self._proj_name == "up_proj":
            self._cache._indices_buffer.append(indices)
        local_indices = self._cache.remap(indices)
        # Remap breaks sorted order — always pass sorted_indices=False
        # Use double buffer methods
        w = self._cache.get_weights(self._proj_name)
        s = self._cache.get_scales(self._proj_name)
        b = self._cache.get_biases(self._proj_name)
        return mx.gather_qmm(
            x, w, s, b,
            rhs_indices=local_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


class SyncPredictiveCachedSwitchLinear(nn.Module):
    """Same as PredictiveCachedSwitchLinear but WITH mx.eval(indices).

    Isolates the sync-point hypothesis: pre-stacked tensors, GPU lookup table,
    but forces a per-layer pipeline flush via mx.eval. Comparing this against
    PredictiveCachedSwitchLinear measures the cost of sync points alone.
    """

    def __init__(self, group_size: int, bits: int, mode: str,
                 proj_name: str, cache: PredictiveExpertCache):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._proj_name = proj_name
        self._cache = cache
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        if self._proj_name == "up_proj":
            self._cache._indices_buffer.append(indices)
        mx.eval(indices)
        local_indices = self._cache.remap(indices)
        # Remap breaks sorted order — always pass sorted_indices=False
        # Use double buffer methods
        w = self._cache.get_weights(self._proj_name)
        s = self._cache.get_scales(self._proj_name)
        b = self._cache.get_biases(self._proj_name)
        return mx.gather_qmm(
            x, w, s, b,
            rhs_indices=local_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


