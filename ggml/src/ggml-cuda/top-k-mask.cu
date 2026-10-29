#include "top-k-mask.cuh"
#include "convert.cuh"

#define CUDA_TOP_K_MASK_BLOCK_SIZE 512

// one block per row: fill the row with -inf, then copy the mask cells of the live picks and the extra cells
template <typename T>
static __global__ void top_k_mask_kernel(
        const T * mask, const int32_t * top_k, const float * score, const int32_t * blocks, const int32_t * extra, T * dst,
        const int64_t n_kv, const int64_t ne1, const int64_t ne2,
        const int64_t m1, const int64_t m2, const int64_t m3,
        const int64_t n_top, const int64_t n_pick, const int64_t n_blk, const int64_t n_extra,
        const int64_t sm1, const int64_t sm2, const int64_t sm3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3,
        const int64_t sk0, const int64_t sk1, const int64_t sk2, const int64_t sk3,
        const int64_t ss0, const int64_t ss1, const int64_t ss2, const int64_t ss3,
        const int64_t sb0, const int64_t sb1,
        const int64_t se0, const int64_t se1, const int64_t se2, const int64_t se3,
        const bool vec) {
    ggml_cuda_pdl_lc();

    const int64_t ir = blockIdx.x;
    const int64_t i1 = ir % ne1;
    const int64_t i2 = (ir / ne1) % ne2;
    const int64_t i3 = ir / (ne1*ne2);

    const T       * mask_row  = mask  + (i1 % m1)*sm1 + (i2 % m2)*sm2 + (i3 % m3)*sm3;
    T             * dst_row   = dst   + i1*sd1 + i2*sd2 + i3*sd3;
    const int32_t * top_k_row = top_k + i1*sk1 + i2*sk2 + i3*sk3;

    ggml_cuda_pdl_sync();

    const T ninf = ggml_cuda_cast<T>(-INFINITY);
    if (vec) {
        constexpr int nv = sizeof(int4)/sizeof(T);
        T fill[nv];
#pragma unroll
        for (int j = 0; j < nv; ++j) {
            fill[j] = ninf;
        }
        const int4 fill4 = *(const int4 *) fill;
        for (int64_t i = threadIdx.x; i < n_kv/nv; i += blockDim.x) {
            ((int4 *) dst_row)[i] = fill4;
        }
    } else {
        for (int64_t i = threadIdx.x; i < n_kv; i += blockDim.x) {
            dst_row[i] = ninf;
        }
    }

    __syncthreads();

    // one thread per pick or extra cell; cells kept twice get the same value
    for (int64_t i = threadIdx.x; i < n_top + n_extra; i += blockDim.x) {
        if (i >= n_top) {
            const int32_t c = extra[(i - n_top)*se0 + i1*se1 + i2*se2 + i3*se3];
            if (c >= 0 && c < n_kv) {
                dst_row[c] = mask_row[c];
            }
            continue;
        }

        const int32_t p = top_k_row[i*sk0];
        if (p < 0 || p >= n_pick) {
            continue;
        }
        if (score && !(score[p*ss0 + i1*ss1 + i2*ss2 + i3*ss3] > -INFINITY)) {
            continue;
        }
        if (!blocks) {
            if (p < n_kv) {
                dst_row[p] = mask_row[p];
            }
            continue;
        }
        for (int64_t k = 0; k < n_blk; ++k) {
            const int32_t c = blocks[k*sb0 + p*sb1];
            if (c >= 0 && c < n_kv) {
                dst_row[c] = mask_row[c];
            }
        }
    }
}

template <typename T>
static void top_k_mask_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mask   = dst->src[0];
    const ggml_tensor * top_k  = dst->src[1];
    const ggml_tensor * score  = dst->src[2];
    const ggml_tensor * blocks = dst->src[3];
    const ggml_tensor * extra  = dst->src[4];

    const int64_t n_kv = mask->ne[0];

    GGML_ASSERT(dst->nb[0] == sizeof(T) && mask->nb[0] == sizeof(T));
    GGML_ASSERT(dst->nb[1] % sizeof(T) == 0 && mask->nb[1] % sizeof(T) == 0);

    // 16-byte stores need every dst row aligned
    const bool vec = (n_kv*sizeof(T)) % sizeof(int4) == 0 &&
        dst->nb[1] % sizeof(int4) == 0 && dst->nb[2] % sizeof(int4) == 0 && dst->nb[3] % sizeof(int4) == 0 &&
        (uintptr_t) dst->data % sizeof(int4) == 0;

    const int64_t n_pick = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;

    const size_t si = sizeof(int32_t);
    const size_t sf = sizeof(float);

    const dim3 block_dims(CUDA_TOP_K_MASK_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims(ggml_nrows(dst), 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(top_k_mask_kernel<T>, launch_params,
            (const T *) mask->data, (const int32_t *) top_k->data,
            score  ? (const float   *) score->data  : nullptr,
            blocks ? (const int32_t *) blocks->data : nullptr,
            extra  ? (const int32_t *) extra->data  : nullptr,
            (T *) dst->data,
            n_kv, dst->ne[1], dst->ne[2],
            mask->ne[1], mask->ne[2], mask->ne[3],
            top_k->ne[0], n_pick, blocks ? blocks->ne[0] : 1, extra ? extra->ne[0] : 0,
            mask->nb[1]/sizeof(T), mask->nb[2]/sizeof(T), mask->nb[3]/sizeof(T),
            dst->nb[1]/sizeof(T),  dst->nb[2]/sizeof(T),  dst->nb[3]/sizeof(T),
            top_k->nb[0]/si, top_k->nb[1]/si, top_k->nb[2]/si, top_k->nb[3]/si,
            score  ? score->nb[0]/sf  : 0, score ? score->nb[1]/sf : 0, score ? score->nb[2]/sf : 0, score ? score->nb[3]/sf : 0,
            blocks ? blocks->nb[0]/si : 0, blocks ? blocks->nb[1]/si : 0,
            extra  ? extra->nb[0]/si  : 0, extra ? extra->nb[1]/si : 0, extra ? extra->nb[2]/si : 0, extra ? extra->nb[3]/si : 0,
            vec);
}

void ggml_cuda_op_top_k_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->type) {
        case GGML_TYPE_F16:
            top_k_mask_cuda<half>(ctx, dst);
            break;
        case GGML_TYPE_F32:
            top_k_mask_cuda<float>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}

#define CUDA_TOP_K_CELLS_BLOCK_SIZE 512

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// exclusive prefix sum of v over the block, with the total of the block
static __device__ __forceinline__ int top_k_cells_block_scan(const int v, int & total) {
    __shared__ int warp_sums[CUDA_TOP_K_CELLS_BLOCK_SIZE/WARP_SIZE];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    int x = v;
#pragma unroll
    for (int o = 1; o < WARP_SIZE; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffff, x, o, WARP_SIZE);
        x += lane >= o ? y : 0;
    }
    if (lane == WARP_SIZE - 1) {
        warp_sums[warp] = x;
    }
    __syncthreads();

    constexpr int n_warps = CUDA_TOP_K_CELLS_BLOCK_SIZE/WARP_SIZE;
    if (warp == 0) {
        int w = lane < n_warps ? warp_sums[lane] : 0;
#pragma unroll
        for (int o = 1; o < WARP_SIZE; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffff, w, o, WARP_SIZE);
            w += lane >= o ? y : 0;
        }
        if (lane < n_warps) {
            warp_sums[lane] = w;
        }
    }
    __syncthreads();

    total = warp_sums[n_warps - 1];
    const int res = (warp > 0 ? warp_sums[warp - 1] : 0) + x - v;
    __syncthreads();
    return res;
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// at most this many picks per row, sorted in shared memory
#define CUDA_TOP_K_CELLS_MAX_TOP 8192

// one block per row: the live picks are sorted, then each thread counts the cells of one pick or extra cell, and a prefix
// sum places them in order
static __global__ void __launch_bounds__(CUDA_TOP_K_CELLS_BLOCK_SIZE) top_k_cells_kernel(
        const int32_t * top_k, const float * score, const int32_t * blocks, const int32_t * extra, int32_t * dst,
        const int64_t n_cells, const int64_t n_sel, const int64_t ne1, const int64_t ne2,
        const int64_t n_top, const int64_t n_sort, const int64_t n_pick, const int64_t n_blk, const int64_t n_extra,
        const int64_t sd1, const int64_t sd2, const int64_t sd3,
        const int64_t sk0, const int64_t sk1, const int64_t sk2, const int64_t sk3,
        const int64_t ss0, const int64_t ss1, const int64_t ss2, const int64_t ss3,
        const int64_t sb0, const int64_t sb1,
        const int64_t se0, const int64_t se1, const int64_t se2, const int64_t se3) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    extern __shared__ int32_t picks[]; // [n_sort], a power of 2 >= n_top

    ggml_cuda_pdl_lc();

    const int64_t ir = blockIdx.x;
    const int64_t i1 = ir % ne1;
    const int64_t i2 = (ir / ne1) % ne2;
    const int64_t i3 = ir / (ne1*ne2);

    int32_t       * dst_row   = dst   + i1*sd1 + i2*sd2 + i3*sd3;
    const int32_t * top_k_row = top_k + i1*sk1 + i2*sk2 + i3*sk3;

    ggml_cuda_pdl_sync();

    // the dead picks and the padding sort last
    constexpr int32_t dead = INT32_MAX;
    for (int64_t j = threadIdx.x; j < n_sort; j += CUDA_TOP_K_CELLS_BLOCK_SIZE) {
        int32_t p = dead;
        if (j < n_top) {
            p = top_k_row[j*sk0];
            if (p < 0 || p >= n_pick || (score && !(score[p*ss0 + i1*ss1 + i2*ss2 + i3*ss3] > -INFINITY))) {
                p = dead;
            }
        }
        picks[j] = p;
    }
    __syncthreads();

    // bitonic sort, ascending
    for (int64_t k = 2; k <= n_sort; k <<= 1) {
        for (int64_t m = k >> 1; m > 0; m >>= 1) {
            for (int64_t i = threadIdx.x; i < n_sort; i += CUDA_TOP_K_CELLS_BLOCK_SIZE) {
                const int64_t l = i ^ m;
                if (l > i) {
                    const int32_t a = picks[i];
                    const int32_t b = picks[l];
                    if ((a > b) == ((i & k) == 0)) {
                        picks[i] = b;
                        picks[l] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    auto valid = [&](const int32_t c) {
        return c >= 0 && c < n_cells;
    };

    int64_t n_out = 0;
    for (int64_t j0 = 0; j0 < n_top + n_extra; j0 += CUDA_TOP_K_CELLS_BLOCK_SIZE) {
        const int64_t j = j0 + threadIdx.x;

        // the pick of this thread, dead or not, or the extra cell
        int32_t p = dead;
        int     n = 0;
        if (j < n_top) {
            p = picks[j];
            if (p == dead) {
                n = 0;
            } else if (!blocks) {
                n = valid(p);
            } else {
                for (int64_t k = 0; k < n_blk; ++k) {
                    n += valid(blocks[k*sb0 + p*sb1]);
                }
            }
        } else if (j < n_top + n_extra) {
            p = extra[(j - n_top)*se0 + i1*se1 + i2*se2 + i3*se3];
            n = valid(p);
        }

        int total;
        int32_t * out = dst_row + n_out + top_k_cells_block_scan(n, total);

        if (n > 0) {
            if (j >= n_top || !blocks) {
                out[0] = p;
            } else {
                for (int64_t k = 0; k < n_blk; ++k) {
                    const int32_t c = blocks[k*sb0 + p*sb1];
                    if (valid(c)) {
                        *out++ = c;
                    }
                }
            }
        }
        n_out += total;
    }

    for (int64_t i = n_out + threadIdx.x; i < n_sel; i += CUDA_TOP_K_CELLS_BLOCK_SIZE) {
        dst_row[i] = -1;
    }
#else
    GGML_UNUSED_VARS(top_k, score, blocks, extra, dst, n_cells, n_sel, ne1, ne2, n_top, n_sort, n_pick, n_blk, n_extra,
        sd1, sd2, sd3, sk0, sk1, sk2, sk3, ss0, ss1, ss2, ss3, sb0, sb1, se0, se1, se2, se3);
    NO_DEVICE_CODE;
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_op_top_k_cells_supported(const ggml_tensor * op) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(op);
    return false;
#else
    return op->type == GGML_TYPE_I32 && op->nb[0] == sizeof(int32_t) && op->src[0]->ne[0] <= CUDA_TOP_K_CELLS_MAX_TOP;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

void ggml_cuda_op_top_k_cells(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * top_k  = dst->src[0];
    const ggml_tensor * score  = dst->src[1];
    const ggml_tensor * blocks = dst->src[2];
    const ggml_tensor * extra  = dst->src[3];

    GGML_ASSERT(dst->type == GGML_TYPE_I32 && dst->nb[0] == sizeof(int32_t));

    const int64_t n_cells = ggml_get_op_params_i32(dst, 0);
    const int64_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_cells;

    const size_t si = sizeof(int32_t);
    const size_t sf = sizeof(float);

    int64_t n_sort = 1;
    while (n_sort < top_k->ne[0]) {
        n_sort *= 2;
    }
    GGML_ASSERT(n_sort <= CUDA_TOP_K_CELLS_MAX_TOP);

    const dim3 block_dims(CUDA_TOP_K_CELLS_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims(ggml_nrows(dst), 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, n_sort*sizeof(int32_t), ctx.stream());

    ggml_cuda_kernel_launch(top_k_cells_kernel, launch_params,
            (const int32_t *) top_k->data,
            score  ? (const float   *) score->data  : nullptr,
            blocks ? (const int32_t *) blocks->data : nullptr,
            extra  ? (const int32_t *) extra->data  : nullptr,
            (int32_t *) dst->data,
            n_cells, dst->ne[0], dst->ne[1], dst->ne[2],
            top_k->ne[0], n_sort, n_pick, blocks ? blocks->ne[0] : 1, extra ? extra->ne[0] : 0,
            dst->nb[1]/si, dst->nb[2]/si, dst->nb[3]/si,
            top_k->nb[0]/si, top_k->nb[1]/si, top_k->nb[2]/si, top_k->nb[3]/si,
            score  ? score->nb[0]/sf  : 0, score ? score->nb[1]/sf : 0, score ? score->nb[2]/sf : 0, score ? score->nb[3]/sf : 0,
            blocks ? blocks->nb[0]/si : 0, blocks ? blocks->nb[1]/si : 0,
            extra  ? extra->nb[0]/si  : 0, extra ? extra->nb[1]/si : 0, extra ? extra->nb[2]/si : 0, extra ? extra->nb[3]/si : 0);
}

// one block per row
template <typename T>
static __global__ void range_mask_kernel(
        const int32_t * ranges, T * dst, const int64_t n_cols, const int64_t ne1, const int64_t ne2,
        const int64_t sr0, const int64_t sr1, const int64_t sr2, const int64_t sr3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3) {
    ggml_cuda_pdl_lc();

    const int64_t ir = blockIdx.x;
    const int64_t i1 = ir % ne1;
    const int64_t i2 = (ir / ne1) % ne2;
    const int64_t i3 = ir / (ne1*ne2);

    ggml_cuda_pdl_sync();

    const int32_t * range = ranges + i1*sr1 + i2*sr2 + i3*sr3;
    const int64_t c0 = min(max((int64_t) range[0], (int64_t) 0), n_cols);
    const int64_t c1 = min(max((int64_t) range[sr0], c0), n_cols);

    T * row = dst + i1*sd1 + i2*sd2 + i3*sd3;
    const T keep = ggml_cuda_cast<T>(0.0f);
    const T drop = ggml_cuda_cast<T>(-INFINITY);
    for (int64_t c = threadIdx.x; c < n_cols; c += blockDim.x) {
        row[c] = c >= c0 && c < c1 ? keep : drop;
    }
}

template <typename T>
static void range_mask_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ranges = dst->src[0];

    const size_t si = sizeof(int32_t);
    const size_t ts = sizeof(T);

    const dim3 block_dims(256, 1, 1);
    const dim3 grid_dims(ggml_nrows(dst), 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(range_mask_kernel<T>, launch_params,
            (const int32_t *) ranges->data, (T *) dst->data, dst->ne[0], dst->ne[1], dst->ne[2],
            ranges->nb[0]/si, ranges->nb[1]/si, ranges->nb[2]/si, ranges->nb[3]/si,
            dst->nb[1]/ts, dst->nb[2]/ts, dst->nb[3]/ts);
}

void ggml_cuda_op_range_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->type) {
        case GGML_TYPE_F16:
            range_mask_cuda<half>(ctx, dst);
            break;
        case GGML_TYPE_F32:
            range_mask_cuda<float>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
