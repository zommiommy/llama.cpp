#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

// Prompt-sized batches sort the (token, slot) pairs by expert with a counting sort instead: the kernel above reads every
// pair once per expert, 0.66 ms for 8192 tokens with 10 of 512 experts. The sort keeps the token order within each
// expert, so the maps are the same.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static constexpr int mm_ids_sort_chunk      = 64;  // tokens per chunk of the sort
static constexpr int mm_ids_sort_min_tokens = 512; // smaller batches keep the kernel above
static constexpr int mm_ids_sort_max_experts = 1024;

// counts[ic*n_experts + e]: the pairs of the tokens of chunk ic that use expert e
static __global__ void mm_ids_sort_count(
        const int32_t * __restrict__ ids, int32_t * __restrict__ counts,
        const int n_experts, const int n_tokens, const int n_expert_used, const int si1) {
    extern __shared__ int32_t data_mm_ids_sort_count[];
    int32_t * hist = data_mm_ids_sort_count;
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        hist[e] = 0;
    }
    __syncthreads();

    const int it0 = blockIdx.x*mm_ids_sort_chunk;
    const int n   = (min(it0 + mm_ids_sort_chunk, n_tokens) - it0)*n_expert_used;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int e = ids[(it0 + i/n_expert_used)*si1 + i%n_expert_used];
        if (e >= 0 && e < n_experts) {
            atomicAdd(&hist[e], 1);
        }
    }
    __syncthreads();

    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        counts[blockIdx.x*n_experts + e] = hist[e];
    }
}

// one thread per expert turns the counts into the position of the first pair of each chunk and expert, the experts in
// order: expert_bounds[e] is the first position of expert e, expert_bounds[n_experts] the number of pairs
static __global__ void mm_ids_sort_scan(
        int32_t * __restrict__ offsets, int32_t * __restrict__ expert_bounds, const int n_experts, const int n_chunks) {
    __shared__ int32_t sum[mm_ids_sort_max_experts];
    const int e = threadIdx.x;

    int total = 0;
    if (e < n_experts) {
        for (int ic = 0; ic < n_chunks; ++ic) {
            const int count = offsets[ic*n_experts + e];
            offsets[ic*n_experts + e] = total;
            total += count;
        }
    }
    sum[e] = total;
    __syncthreads();
    for (int offset = 1; offset < (int) blockDim.x; offset *= 2) {
        const int add = e >= offset ? sum[e - offset] : 0;
        __syncthreads();
        sum[e] += add;
        __syncthreads();
    }
    const int first = sum[e] - total;

    if (e < n_experts) {
        for (int ic = 0; ic < n_chunks; ++ic) {
            offsets[ic*n_experts + e] += first;
        }
        expert_bounds[e] = first;
    }
    if (e == (int) blockDim.x - 1) {
        expert_bounds[n_experts] = sum[e];
    }
}

// one warp per chunk walks its pairs in token order and places them; the pairs of an expert within a group of 32 are
// ranked by lane, and the last of them advances the expert
static __global__ void mm_ids_sort_scatter(
        const int32_t * __restrict__ ids, const int32_t * __restrict__ offsets,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y,
        const int si1, const int sis1, const bool write_inverse) {
    extern __shared__ int32_t data_mm_ids_sort_scatter[];
    int32_t * next = data_mm_ids_sort_scatter;
    for (int e = threadIdx.x; e < n_experts; e += WARP_SIZE) {
        next[e] = offsets[blockIdx.x*n_experts + e];
    }
    __syncwarp();

    const int it0 = blockIdx.x*mm_ids_sort_chunk;
    const int n   = (min(it0 + mm_ids_sort_chunk, n_tokens) - it0)*n_expert_used;
    for (int i0 = 0; i0 < n; i0 += WARP_SIZE) {
        const int  i     = i0 + threadIdx.x;
        const int  it    = it0 + i/n_expert_used;
        const int  iex   = i%n_expert_used;
        const int  e     = i < n ? ids[it*si1 + iex] : -1;
        const bool valid = e >= 0 && e < n_experts;

        const unsigned peers = __match_any_sync(0xFFFFFFFF, valid ? e : -1);
        const int      pos   = valid ? next[e] + __popc(peers & ((1u << threadIdx.x) - 1)) : 0;
        __syncwarp();
        if (valid && (peers >> threadIdx.x) == 1) {
            next[e] = pos + 1;
        }
        __syncwarp();

        if (valid) {
            ids_dst[pos] = it*n_expert_used + iex;
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex] = pos;
            } else {
                ids_src1[pos] = it*sis1 + iex % nchannels_y;
            }
        }
    }
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(ggml_cuda_pool & pool,
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (n_tokens >= mm_ids_sort_min_tokens && n_experts <= mm_ids_sort_max_experts &&
            ggml_cuda_info().devices[ggml_cuda_get_device()].cc >= GGML_CUDA_CC_VOLTA) {
        const int n_chunks = (n_tokens + mm_ids_sort_chunk - 1)/mm_ids_sort_chunk;
        ggml_cuda_pool_alloc<int32_t> offsets(pool, (size_t) n_chunks*n_experts);
        const size_t nbytes_shared = n_experts*sizeof(int32_t);

        int n_threads_scan = WARP_SIZE;
        while (n_threads_scan < n_experts) {
            n_threads_scan *= 2;
        }

        mm_ids_sort_count<<<n_chunks, 256, nbytes_shared, stream>>>(ids, offsets.get(), n_experts, n_tokens, n_expert_used, si1);
        mm_ids_sort_scan<<<1, n_threads_scan, 0, stream>>>(offsets.get(), expert_bounds, n_experts, n_chunks);
        mm_ids_sort_scatter<<<n_chunks, WARP_SIZE, nbytes_shared, stream>>>(ids, offsets.get(), ids_src1, ids_dst,
            n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
        return;
    }
#else
    GGML_UNUSED(pool);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
