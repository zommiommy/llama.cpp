#include "moe-cache.cuh"

// state layout, see ggml_moe_cache_fill
#define MOE_CACHE_CURSOR    0
#define MOE_CACHE_N_VICTIMS 1
#define MOE_CACHE_N_COPIES  2
#define MOE_CACHE_SEQ       3
#define MOE_CACHE_N_PAIRS   4
#define MOE_CACHE_N_SKIP    5
#define MOE_CACHE_HEADER    8

// log layout: record count, capacity, sequence number of the last post to the host, 1 unused, then 3 values per record
#define MOE_CACHE_LOG_HEADER 4
#define MOE_CACHE_LOG_SEQ    2

// xchg layout, see ggml_moe_cache_xchg
#define MOE_CACHE_XCHG_POSTED  0
#define MOE_CACHE_XCHG_DONE    1
#define MOE_CACHE_XCHG_N_PAIRS 2
#define MOE_CACHE_XCHG_IL      3
#define MOE_CACHE_XCHG_HEADER  16

#define MOE_CACHE_PLAN_THREADS 256
#define MOE_CACHE_JOIN_THREADS 256
#define MOE_CACHE_COPY_THREADS 256
#define MOE_CACHE_COPY_CHUNK   (64*1024)

struct moe_cache_xchg_view {
    int32_t * ctl;
    int32_t * expert;
    int32_t * pos;
    float   * x;
    float   * y;
};

static __device__ __host__ moe_cache_xchg_view moe_cache_xchg_at(char * base, const int max_pairs, const int n_embd) {
    moe_cache_xchg_view v;
    v.ctl    = (int32_t *) base;
    v.expert = v.ctl + MOE_CACHE_XCHG_HEADER;
    v.pos    = v.expert + max_pairs;
    v.x      = (float *) (v.pos + max_pairs);
    v.y      = v.x + (int64_t) max_pairs*n_embd;
    return v;
}

static __device__ __forceinline__ void moe_cache_log(int32_t * log, const int log_cap, const int il, const int s, const int x) {
    const int k = atomicAdd(&log[0], 1);
    if (k < log_cap) {
        int32_t * r = log + MOE_CACHE_LOG_HEADER + 3*k;
        r[0] = il;
        r[1] = s;
        r[2] = x;
    }
}

// one block: finds the selected experts in the slot map; skips the missing experts no token selects at a rank below skip_rank;
// leaves up to cpu_max of the other missing experts to the host and posts their pairs and inputs to xchg; takes a victim for
// each other missing expert, updates the slot map, the slot owners and the log, and leaves the copies for moe_cache_copy in the
// state
static __global__ void moe_cache_plan(
        const int32_t * __restrict__ ids, const int n_ids, const int n_used,
        const float * __restrict__ x, const int n_embd,
        int32_t * __restrict__ slot_map, int32_t * __restrict__ state, int32_t * __restrict__ log,
        int32_t * __restrict__ slots, const int il, const int n_expert, const int n_slots,
        char * xchg, const int max_pairs, const int cpu_max, const int skip_rank, const int skip_off) {
    extern __shared__ uint32_t smem[];
    const int nw = (n_expert + 31) / 32;
    uint32_t * used   = smem;            // experts selected by the batch
    uint32_t * missed = smem + nw;       // missing experts already listed
    uint32_t * logged = smem + 2*nw;     // cached experts already logged as hits
    uint32_t * cpu    = smem + 3*nw;     // missing experts left to the host
    uint32_t * skip   = smem + 4*nw;     // missing experts skipped
    int32_t  * miss_e = (int32_t *) (smem + 5*nw);
    int32_t  * miss_s = miss_e + n_ids;  // the best rank of each missing expert, then the victim of each one copied
    int32_t  * pair_i = miss_s + n_ids;  // selection index of each pair left to the host

    __shared__ int n_miss;
    __shared__ int n_rest; // missing experts not skipped, moved to the front of miss_e
    __shared__ int n_cpu;
    __shared__ int n_pairs;
    __shared__ int n_skip;

    int32_t * owner   = state + MOE_CACHE_HEADER;
    int32_t * victims = owner + n_slots;
    int32_t * copies  = victims + n_slots;
    int32_t * skipped = state + skip_off;
    int32_t * row     = slot_map + (int64_t) il*n_expert;

    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
        used[i]   = 0;
        missed[i] = 0;
        logged[i] = 0;
        cpu[i]    = 0;
        skip[i]   = 0;
    }
    if (threadIdx.x == 0) {
        n_miss  = 0;
        n_pairs = 0;
        n_skip  = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int      e   = ids[i];
        const uint32_t bit = 1u << (e % 32);
        atomicOr(&used[e / 32], bit);
        if (row[e] < 0 && !(atomicOr(&missed[e / 32], bit) & bit)) {
            miss_e[atomicAdd(&n_miss, 1)] = e;
        }
    }
    __syncthreads();

    const int nv = state[MOE_CACHE_N_VICTIMS];
    // the pairs the device does not compute take victim slots that hold no other expert of their token, below
    const bool dummies = nv >= 2*n_used;
    const bool tail    = skip_rank > 0 && dummies;

    // the selection of a token is ordered by weight: the best rank of an expert is its first position in any token
    if (tail) {
        for (int j = threadIdx.x; j < n_miss; j += blockDim.x) {
            const int e = miss_e[j];
            int r = n_used;
            for (int i = 0; i < n_ids; ++i) {
                r = ids[i] == e ? min(r, i % n_used) : r;
            }
            miss_s[j] = r;
        }
        __syncthreads();
    }

    // each pick depends on the last, so one thread walks the victim list: a slot holding an expert of the batch is skipped
    if (threadIdx.x == 0) {
        const int log_cap = log[1];
        int n = 0;
        for (int j = 0; j < n_miss; ++j) {
            const int e = miss_e[j];
            if (tail && miss_s[j] >= skip_rank) {
                skip[e / 32] |= 1u << (e % 32);
                moe_cache_log(log, log_cap, il, -2, e);
            } else {
                miss_e[n++] = e;
            }
        }
        n_rest = n;
        n_cpu  = xchg != nullptr && dummies && n_ids <= max_pairs ? min(n, cpu_max) : 0;
        for (int j = 0; j < n_cpu; ++j) {
            cpu[miss_e[j] / 32] |= 1u << (miss_e[j] % 32);
        }
        int cur = state[MOE_CACHE_CURSOR];
        for (int j = n_cpu; j < n; ++j) {
            int tries = 0;
            for (;;) {
                if (tries++ == nv) {
                    __trap(); // every victim holds an expert of the batch
                }
                const int v = victims[cur];
                cur = cur + 1 < nv ? cur + 1 : 0;
                const int key = owner[v];
                if (key >= 0 && key / n_expert == il) {
                    const int e = key % n_expert;
                    if (used[e / 32] & (1u << (e % 32))) {
                        continue;
                    }
                }
                miss_s[j] = v;
                break;
            }
        }
        state[MOE_CACHE_CURSOR]   = cur;
        state[MOE_CACHE_N_COPIES] = n - n_cpu;
    }
    __syncthreads();

    const int log_cap = log[1];
    for (int j = threadIdx.x; j < n_rest; j += blockDim.x) {
        const int e = miss_e[j];
        if (j < n_cpu) {
            moe_cache_log(log, log_cap, il, -1, e);
            continue;
        }
        const int v   = miss_s[j];
        const int old = owner[v];
        if (old >= 0) {
            slot_map[old] = -1;
        }
        owner[v] = il*n_expert + e;
        row[e]   = v;
        copies[2*(j - n_cpu) + 0] = v;
        copies[2*(j - n_cpu) + 1] = e;
        moe_cache_log(log, log_cap, il, v, e);
    }
    __syncthreads();

    moe_cache_xchg_view xc = {};
    if (xchg != nullptr) {
        xc = moe_cache_xchg_at(xchg, max_pairs, n_embd);
    }
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int      e   = ids[i];
        const uint32_t bit = 1u << (e % 32);
        if ((cpu[e / 32] | skip[e / 32]) & bit) {
            continue;
        }
        const int s = row[e];
        slots[i] = s;
        // a cached expert is logged once, so the host can mark it as used
        if (!(missed[e / 32] & bit) && !(atomicOr(&logged[e / 32], bit) & bit)) {
            moe_cache_log(log, log_cap, il, s, -1 - e);
        }
    }
    // the pairs left to the host and the skipped pairs are computed by the device with victim slots until the join overwrites
    // their outputs: mul_mat_id expects the experts of a token to differ, so a token takes victims none of its experts occupy
    const bool any_dummy = n_cpu > 0 || n_rest < n_miss;
    for (int t = threadIdx.x; any_dummy && t < n_ids/n_used; t += blockDim.x) {
        const int32_t * tids = ids + t*n_used;
        int next = 0;
        for (int k = 0; k < n_used; ++k) {
            const int      e      = tids[k];
            const uint32_t bit    = 1u << (e % 32);
            const bool     is_cpu = cpu[e / 32] & bit;
            if (!is_cpu && !(skip[e / 32] & bit)) {
                continue;
            }
            // the victims are distinct, so the pairs of the token take distinct slots
            int s = -1;
            for (bool taken = true; taken;) {
                s     = victims[next++];
                taken = false;
                for (int k2 = 0; k2 < n_used; ++k2) {
                    const int e2 = tids[k2];
                    taken = taken || (!((cpu[e2 / 32] | skip[e2 / 32]) & (1u << (e2 % 32))) && row[e2] == s);
                }
            }
            slots[t*n_used + k] = s;
            if (is_cpu) {
                const int p = atomicAdd(&n_pairs, 1);
                pair_i[p]    = t*n_used + k;
                xc.expert[p] = e;
                xc.pos[p]    = t*n_used + k;
            } else {
                skipped[atomicAdd(&n_skip, 1)] = t*n_used + k;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        state[MOE_CACHE_N_SKIP] = n_skip;
    }
    if (n_pairs == 0) {
        if (threadIdx.x == 0) {
            state[MOE_CACHE_N_PAIRS] = 0;
        }
        return;
    }

    // the inputs of the pairs, then the post: the host may read them once it sees the sequence number
    for (int64_t k = threadIdx.x; k < (int64_t) n_pairs*n_embd; k += blockDim.x) {
        const int p = k / n_embd;
        const int d = k % n_embd;
        xc.x[k] = x[(int64_t) (pair_i[p] / n_used)*n_embd + d];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        // the posts of all layers of the device share the log, so consecutive posts never have the same number
        const int seq = (log[MOE_CACHE_LOG_SEQ] + 1) & 0x7fffffff;
        log[MOE_CACHE_LOG_SEQ]   = seq;
        state[MOE_CACHE_SEQ]     = seq;
        state[MOE_CACHE_N_PAIRS] = n_pairs;
        xc.ctl[MOE_CACHE_XCHG_N_PAIRS] = n_pairs;
        xc.ctl[MOE_CACHE_XCHG_IL]      = il;
        __threadfence_system();
        *(volatile int32_t *) &xc.ctl[MOE_CACHE_XCHG_POSTED] = seq;
    }
}

struct moe_cache_banks {
    const char * host[3];
    char       * bank[3];
    int64_t      size[3]; // bytes of one expert
};

// block (chunk, bank, copy): copies a chunk of one missing expert from host memory, which the device reads directly, to its slot
static __global__ void moe_cache_copy(const int32_t * __restrict__ state, const int n_slots, const moe_cache_banks banks) {
    const int j = blockIdx.z;
    if (j >= state[MOE_CACHE_N_COPIES]) {
        return;
    }
    const int32_t * copies = state + MOE_CACHE_HEADER + 2*n_slots;
    const int v = copies[2*j + 0];
    const int e = copies[2*j + 1];
    const int b = blockIdx.y;

    const int64_t size = banks.size[b];
    const int64_t i0   = (int64_t) blockIdx.x*MOE_CACHE_COPY_CHUNK;
    if (i0 >= size) {
        return;
    }
    const int64_t i1 = min(i0 + MOE_CACHE_COPY_CHUNK, size);

    const char * src = banks.host[b] + (int64_t) e*size;
    char       * dst = banks.bank[b] + (int64_t) v*size;

    if ((((uintptr_t) src | (uintptr_t) dst | (uintptr_t) size) & 15) == 0) {
        for (int64_t i = i0 + 16*threadIdx.x; i < i1; i += 16*blockDim.x) {
            *(uint4 *) (dst + i) = *(const uint4 *) (src + i);
        }
    } else {
        for (int64_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

// one block: zeroes the outputs of the skipped pairs, waits for the host to finish the pairs the plan posted, then writes their
// outputs into the expert outputs
static __global__ void moe_cache_join(
        float * __restrict__ experts, const int n_embd, const int32_t * __restrict__ state, const int skip_off, char * xchg,
        const int max_pairs) {
    extern __shared__ int32_t pair_pos[];
    const int32_t * skipped = state + skip_off;
    const int       n_skip  = state[MOE_CACHE_N_SKIP];
    for (int64_t k = threadIdx.x; k < (int64_t) n_skip*n_embd; k += blockDim.x) {
        experts[(int64_t) skipped[k / n_embd]*n_embd + k % n_embd] = 0.0f;
    }
    const int n_pairs = xchg != nullptr ? state[MOE_CACHE_N_PAIRS] : 0;
    if (n_pairs == 0) {
        return;
    }
    const moe_cache_xchg_view xc  = moe_cache_xchg_at(xchg, max_pairs, n_embd);
    const int32_t             seq = state[MOE_CACHE_SEQ];
    if (threadIdx.x == 0) {
        while (*(volatile const int32_t *) &xc.ctl[MOE_CACHE_XCHG_DONE] != seq) {
            __nanosleep(256);
        }
        __threadfence_system();
    }
    __syncthreads();
    // the positions once from host memory, then the outputs
    for (int p = threadIdx.x; p < n_pairs; p += blockDim.x) {
        pair_pos[p] = ((volatile const int32_t *) xc.pos)[p];
    }
    __syncthreads();
    for (int64_t k = threadIdx.x; k < (int64_t) n_pairs*n_embd; k += blockDim.x) {
        const int p = k / n_embd;
        const int d = k % n_embd;
        experts[(int64_t) pair_pos[p]*n_embd + d] = ((volatile const float *) xc.y)[k];
    }
}

bool ggml_cuda_moe_cache_fill_supported(const ggml_tensor * dst) {
    const int n_banks = ggml_get_op_params_i32(dst, 1);
    if (n_banks < 1 || n_banks > 3) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (dst->src[i]->type != GGML_TYPE_I32) {
            return false;
        }
    }
    // the planner keeps the batch in shared memory
    const int64_t n_ids    = ggml_nelements(dst->src[0]);
    const int64_t n_expert = dst->src[1]->ne[0];
    return n_ids <= 4096 && n_expert <= 65536;
}

static char * moe_cache_xchg_base(const ggml_tensor * dst, const int i) {
    uint64_t base;
    memcpy(&base, (const int32_t *) dst->op_params + i, sizeof(base));
    return (char *) (uintptr_t) base;
}

void ggml_cuda_moe_cache_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids      = dst->src[0];
    ggml_tensor       * slot_map = dst->src[1];
    ggml_tensor       * state    = dst->src[2];
    ggml_tensor       * log      = dst->src[3];
    const ggml_tensor * x        = dst->src[7];

    const int il        = ggml_get_op_params_i32(dst, 0);
    const int n_banks   = ggml_get_op_params_i32(dst, 1);
    const int max_pairs = ggml_get_op_params_i32(dst, 10);
    const int cpu_max   = ggml_get_op_params_i32(dst, 11);
    const int skip_rank = ggml_get_op_params_i32(dst, 12);
    const int skip_off  = ggml_get_op_params_i32(dst, 13);
    char *    xchg      = cpu_max > 0 ? moe_cache_xchg_base(dst, 8) : nullptr;

    const int n_ids    = (int) ggml_nelements(ids);
    const int n_used   = (int) ids->ne[0];
    const int n_expert = (int) slot_map->ne[0];
    const int n_slots  = (int) dst->src[4]->ne[2] - 1;
    const int n_embd   = x != nullptr ? (int) x->ne[0] : 0;

    moe_cache_banks banks = {};
    int64_t max_size = 0;
    for (int b = 0; b < n_banks; ++b) {
        const ggml_tensor * bank = dst->src[4 + b];
        banks.host[b] = moe_cache_xchg_base(dst, 2 + 2*b);
        banks.bank[b] = (char *) bank->data;
        banks.size[b] = bank->nb[2];
        max_size = std::max(max_size, (int64_t) bank->nb[2]);
    }

    cudaStream_t stream = ctx.stream();

    const int    nw   = (n_expert + 31) / 32;
    const size_t smem = 5*nw*sizeof(uint32_t) + 3*n_ids*sizeof(int32_t);
    moe_cache_plan<<<1, MOE_CACHE_PLAN_THREADS, smem, stream>>>(
        (const int32_t *) ids->data, n_ids, n_used, x != nullptr ? (const float *) x->data : nullptr, n_embd,
        (int32_t *) slot_map->data, (int32_t *) state->data, (int32_t *) log->data,
        (int32_t *) dst->data, il, n_expert, n_slots, xchg, max_pairs, cpu_max, skip_rank, skip_off);
    CUDA_CHECK(cudaGetLastError());

    // the grid covers the most copies a batch can need, so its shape stays the same between calls
    const dim3 grid((unsigned) ((max_size + MOE_CACHE_COPY_CHUNK - 1) / MOE_CACHE_COPY_CHUNK), (unsigned) n_banks, (unsigned) n_ids);
    moe_cache_copy<<<grid, MOE_CACHE_COPY_THREADS, 0, stream>>>((const int32_t *) state->data, n_slots, banks);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_moe_cache_join(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * state = dst->src[2];

    // il, xchg base, max pairs, cpu_max, offset of the skipped pairs, see ggml_moe_cache_join
    const int max_pairs = ggml_get_op_params_i32(dst, 3);
    const int cpu_max   = ggml_get_op_params_i32(dst, 4);
    const int skip_off  = ggml_get_op_params_i32(dst, 5);

    const size_t smem = (size_t) dst->ne[1]*dst->ne[2]*sizeof(int32_t); // a position per selected expert at most
    moe_cache_join<<<1, MOE_CACHE_JOIN_THREADS, smem, ctx.stream()>>>(
        (float *) dst->data, (int) dst->ne[0], (const int32_t *) state->data, skip_off,
        cpu_max > 0 ? moe_cache_xchg_base(dst, 1) : nullptr, max_pairs);
    CUDA_CHECK(cudaGetLastError());
}
