#pragma once

#include "common.cuh"
#include "mma.cuh"
#include "vecdotq.cuh"

#include <type_traits>

// Matrix-vector products with up to 8 columns, the verify batches of speculative decoding, on int8 tensor cores.
// MMVQ decodes each weight once per row but multiplies it into every column with dp4a, so at 8 columns the dp4a work
// outweighs the weight read and the kernel turns compute-bound. Here a warp decodes a 16-row tile of src0 straight
// into mma fragments and multiplies it with all 8 columns in one mma per 32 (or 16) values. The warps of a block take
// turns over the quant blocks of the tile and add up their partial sums at the end.
//
// Thread (g, t) = (lane/4, lane%4) holds the A fragment for rows g and g+8 and values 4t..4t+3 and 16+4t..16+4t+3 of
// a 32-value sub-block, the B fragment for column g, and the results for rows g, g+8 and columns 2t, 2t+1.

#define MMVQ_MMA_NWARPS   4
#define MMVQ_MMA_MIN_ROWS 512

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

using mmvq_mma_tile_a   = ggml_cuda_mma::tile<16, 8, int>; // 16 rows x 32 values
using mmvq_mma_tile_a16 = ggml_cuda_mma::tile<16, 4, int>; // 16 rows x 16 values
using mmvq_mma_tile_b   = ggml_cuda_mma::tile< 8, 8, int>; // 32 values x 8 columns
using mmvq_mma_tile_b16 = ggml_cuda_mma::tile< 8, 4, int>; // 16 values x 8 columns
using mmvq_mma_tile_d   = ggml_cuda_mma::tile<16, 8, int>; // 16 rows x 8 columns

// acc += D * row scale * column scale, rows g and g+8 against columns 2t and 2t+1
static __device__ __forceinline__ void mmvq_mma_scale(
        float * acc, const mmvq_mma_tile_d & D, const float w0, const float w1, const float y0, const float y1) {
    acc[0] += D.x[0] * (w0*y0);
    acc[1] += D.x[1] * (w0*y1);
    acc[2] += D.x[2] * (w1*y0);
    acc[3] += D.x[3] * (w1*y1);
}

// B fragments of column g from one q8_1 block: values 4t..4t+3 and 16+4t..16+4t+3
static __device__ __forceinline__ mmvq_mma_tile_b mmvq_mma_load_b(const block_q8_1 * y) {
    const int t = threadIdx.x % 4;
    mmvq_mma_tile_b B;
    B.x[0] = get_int_b4(y->qs, t);
    B.x[1] = get_int_b4(y->qs, t + 4);
    return B;
}

static __device__ __forceinline__ mmvq_mma_tile_b16 mmvq_mma_load_b16(const block_q8_1 * y, const int half) {
    const int t = threadIdx.x % 4;
    mmvq_mma_tile_b16 B;
    B.x[0] = get_int_b4(y->qs, 4*half + t);
    return B;
}

// negates the bytes of v whose sign bit is set in the low 4 bits of s
static __device__ __forceinline__ int mmvq_mma_apply_signs(const int v, const int s) {
    const int m = __vcmpne4(((s & 0x03) << 7) | ((s & 0x0C) << 21), 0);
    return __vsub4(v ^ m, m);
}

// negates the bytes of v selected by sel (0x08040201 for the low 4 sign bits, 0x80402010 for the high 4) in signs,
// an 8-bit sign mask broadcast to all bytes
static __device__ __forceinline__ int mmvq_mma_apply_signs_sel(const int v, const uint32_t signs, const uint32_t sel) {
    const int m = __vcmpne4(signs & sel, 0);
    return __vsub4(v ^ m, m);
}

// The grid tables of the i-quants, copied to shared memory by each block: a lookup with 32 scattered indices touches up
// to 16 cache lines of a global table, but collides on only a few of the 32 shared memory banks.
static __shared__ uint32_t mmvq_mma_iq3s_grid[512];
static __shared__ uint32_t mmvq_mma_iq3xxs_grid[256];
static __shared__ uint2    mmvq_mma_iq2s_grid[1024];
static __shared__ uint2    mmvq_mma_iq2xs_grid[512];

template <typename T, int n>
static __device__ __forceinline__ void mmvq_mma_stage_table(T * dst, const void * src) {
    for (int i = threadIdx.y*WARP_SIZE + threadIdx.x; i < n; i += MMVQ_MMA_NWARPS*WARP_SIZE) {
        dst[i] = ((const T *) src)[i];
    }
}

// Each specialization multiplies one quant block of rows g and g+8 (x0, x1) with the matching q8_1 blocks: yb holds
// column g (the B operand), ys0 and ys1 columns 2t and 2t+1 (their scales). Batches of min_cols or more columns go
// here instead of to MMVQ, and min_blocks caps the registers for that many blocks per SM; both are tuned on an
// RTX 4090.
template <ggml_type type>
struct mmvq_mma {
    static constexpr int min_cols = 0;
};

template <ggml_type type>
static __device__ __forceinline__ void mmvq_mma_stage() {
    if constexpr (type == GGML_TYPE_IQ3_S) {
        mmvq_mma_stage_table<uint32_t, 512>(mmvq_mma_iq3s_grid, iq3s_grid);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        mmvq_mma_stage_table<uint32_t, 256>(mmvq_mma_iq3xxs_grid, iq3xxs_grid);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        mmvq_mma_stage_table<uint2, 1024>(mmvq_mma_iq2s_grid, iq2s_grid);
    } else if constexpr (type == GGML_TYPE_IQ2_XS) {
        mmvq_mma_stage_table<uint2, 512>(mmvq_mma_iq2xs_grid, iq2xs_grid);
    }
}

template <>
struct mmvq_mma<GGML_TYPE_IQ4_XS> {
    static constexpr int min_cols   = 8;
    static constexpr int min_blocks = 1;
    static __device__ __forceinline__ void block(const char * x0, const char * x1, const block_q8_1 * yb,
            const block_q8_1 * ys0, const block_q8_1 * ys1, float * acc) {
        const block_iq4_xs * b0 = (const block_iq4_xs *) x0;
        const block_iq4_xs * b1 = (const block_iq4_xs *) x1;
        const int t = threadIdx.x % 4;

        const float    d0  = __half2float(b0->d);
        const float    d1  = __half2float(b1->d);
        const uint32_t sh0 = b0->scales_h;
        const uint32_t sh1 = b1->scales_h;
        const uint32_t sl0 = get_int_b4(b0->scales_l, 0);
        const uint32_t sl1 = get_int_b4(b1->scales_l, 0);

#pragma unroll
        for (int sb = 0; sb < QK_K/32; ++sb) {
            const int2 v0 = get_int_from_table_16(get_int_b4(b0->qs, 4*sb + t), kvalues_iq4nl);
            const int2 v1 = get_int_from_table_16(get_int_b4(b1->qs, 4*sb + t), kvalues_iq4nl);
            mmvq_mma_tile_a A;
            A.x[0] = v0.x;
            A.x[1] = v1.x;
            A.x[2] = v0.y;
            A.x[3] = v1.y;
            mmvq_mma_tile_d D;
            ggml_cuda_mma::mma(D, A, mmvq_mma_load_b(yb + sb));

            const int ls0 = ((sl0 >> 4*sb) & 0x0F) | (((sh0 >> 2*sb) & 0x03) << 4);
            const int ls1 = ((sl1 >> 4*sb) & 0x0F) | (((sh1 >> 2*sb) & 0x03) << 4);
            mmvq_mma_scale(acc, D, d0*(ls0 - 32), d1*(ls1 - 32), __low2float(ys0[sb].ds), __low2float(ys1[sb].ds));
        }
    }
};

// 6-bit scale and min of sub-block sb from the 12 packed bytes of a q4_K/q5_K block, as get_scale_min_k4
static __device__ __forceinline__ void mmvq_mma_scale_min_k4(const uint8_t * s, const int sb, int & sc, int & m) {
    if (sb < 4) {
        sc = s[sb]     & 63;
        m  = s[sb + 4] & 63;
    } else {
        sc = (s[sb + 4] & 0x0F) | ((s[sb - 4] >> 6) << 4);
        m  = (s[sb + 4] >>   4) | ((s[sb]     >> 6) << 4);
    }
}

// q4_K and q5_K: values (q4 | qh bit << 4) * d*sc - dmin*m per 32-value sub-block
template <ggml_type type>
static __device__ __forceinline__ void mmvq_mma_block_q45_K(const char * x0, const char * x1, const block_q8_1 * yb,
        const block_q8_1 * ys0, const block_q8_1 * ys1, float * acc) {
    using block_t = std::conditional_t<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K>;
    const block_t * b0 = (const block_t *) x0;
    const block_t * b1 = (const block_t *) x1;
    const int t = threadIdx.x % 4;

    const float2 dm0 = __half22float2(b0->dm);
    const float2 dm1 = __half22float2(b1->dm);
    int s0[3];
    int s1[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        s0[i] = get_int_b4(b0->scales, i);
        s1[i] = get_int_b4(b1->scales, i);
    }

    [[maybe_unused]] int qh0[2] = {0, 0};
    [[maybe_unused]] int qh1[2] = {0, 0};
    if constexpr (type == GGML_TYPE_Q5_K) {
        qh0[0] = get_int_b4(b0->qh, t);
        qh0[1] = get_int_b4(b0->qh, t + 4);
        qh1[0] = get_int_b4(b1->qh, t);
        qh1[1] = get_int_b4(b1->qh, t + 4);
    }

#pragma unroll
    for (int c = 0; c < QK_K/64; ++c) {
        // bytes 32c + 4t.. and 32c + 16 + 4t..: low nibbles are sub-block 2c, high nibbles sub-block 2c + 1
        const int q0[2] = {get_int_b4(b0->qs, 8*c + t), get_int_b4(b0->qs, 8*c + t + 4)};
        const int q1[2] = {get_int_b4(b1->qs, 8*c + t), get_int_b4(b1->qs, 8*c + t + 4)};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int sb = 2*c + h;
            mmvq_mma_tile_a A;
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                int v0 = (q0[k] >> 4*h) & 0x0F0F0F0F;
                int v1 = (q1[k] >> 4*h) & 0x0F0F0F0F;
                if constexpr (type == GGML_TYPE_Q5_K) {
                    v0 |= ((qh0[k] >> sb) & 0x01010101) << 4;
                    v1 |= ((qh1[k] >> sb) & 0x01010101) << 4;
                }
                A.x[2*k + 0] = v0;
                A.x[2*k + 1] = v1;
            }
            mmvq_mma_tile_d D;
            ggml_cuda_mma::mma(D, A, mmvq_mma_load_b(yb + sb));

            int sc0, m0, sc1, m1;
            mmvq_mma_scale_min_k4((const uint8_t *) s0, sb, sc0, m0);
            mmvq_mma_scale_min_k4((const uint8_t *) s1, sb, sc1, m1);
            const float2 y0 = __half22float2(ys0[sb].ds);
            const float2 y1 = __half22float2(ys1[sb].ds);
            mmvq_mma_scale(acc, D, dm0.x*sc0, dm1.x*sc1, y0.x, y1.x);
            // the min term multiplies the sum of each q8_1 block's values
            acc[0] -= dm0.y*m0 * y0.y;
            acc[1] -= dm0.y*m0 * y1.y;
            acc[2] -= dm1.y*m1 * y0.y;
            acc[3] -= dm1.y*m1 * y1.y;
        }
    }
}

template <>
struct mmvq_mma<GGML_TYPE_Q4_K> {
    static constexpr int min_cols   = 6;
    static constexpr int min_blocks = 1;
    static __device__ __forceinline__ void block(const char * x0, const char * x1, const block_q8_1 * yb,
            const block_q8_1 * ys0, const block_q8_1 * ys1, float * acc) {
        mmvq_mma_block_q45_K<GGML_TYPE_Q4_K>(x0, x1, yb, ys0, ys1, acc);
    }
};

template <>
struct mmvq_mma<GGML_TYPE_Q5_K> {
    static constexpr int min_cols   = 6;
    static constexpr int min_blocks = 1;
    static __device__ __forceinline__ void block(const char * x0, const char * x1, const block_q8_1 * yb,
            const block_q8_1 * ys0, const block_q8_1 * ys1, float * acc) {
        mmvq_mma_block_q45_K<GGML_TYPE_Q5_K>(x0, x1, yb, ys0, ys1, acc);
    }
};

// q3_K: 16 scales per block, one per 16 values, so two k16 mmas per 32-value sub-block
template <>
struct mmvq_mma<GGML_TYPE_Q3_K> {
    static constexpr int min_cols   = 4;
    static constexpr int min_blocks = 1;
    static __device__ __forceinline__ void block(const char * x0, const char * x1, const block_q8_1 * yb,
            const block_q8_1 * ys0, const block_q8_1 * ys1, float * acc) {
        const block_q3_K * b[2] = {(const block_q3_K *) x0, (const block_q3_K *) x1};
        const int t = threadIdx.x % 4;

        float d[2];
        int   sc[2][4]; // 16 six-bit scales, unpacked as dequantize_row_q3_K
        int   hm[2][2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            d[r] = __half2float(b[r]->d);
            const uint32_t a0  = get_int_b2(b[r]->scales, 0);
            const uint32_t a1  = get_int_b2(b[r]->scales, 1);
            const uint32_t tmp = get_int_b2(b[r]->scales, 2);
            sc[r][0] = (a0 & 0x0F0F0F0F)        | (((tmp >> 0) & 0x03030303) << 4);
            sc[r][1] = (a1 & 0x0F0F0F0F)        | (((tmp >> 2) & 0x03030303) << 4);
            sc[r][2] = ((a0 >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
            sc[r][3] = ((a1 >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
            hm[r][0] = get_int_b2(b[r]->hmask, t);
            hm[r][1] = get_int_b2(b[r]->hmask, t + 4);
        }

#pragma unroll
        for (int n = 0; n < QK_K/128; ++n) {
            int q[2][2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                q[r][0] = get_int_b2(b[r]->qs, 8*n + t);
                q[r][1] = get_int_b2(b[r]->qs, 8*n + t + 4);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int sb = 4*n + j;
                float part[2][4];
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    mmvq_mma_tile_a16 A;
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        // 2 low bits, minus 4 unless the high bit is set
                        const int v = ((q[r][half] >> 2*j) & 0x03030303) | (((hm[r][half] >> sb) & 0x01010101) << 2);
                        A.x[r] = __vsub4(v, 0x04040404);
                    }
                    mmvq_mma_tile_d D;
                    ggml_cuda_mma::mma(D, A, mmvq_mma_load_b16(yb + sb, half));
                    const int is = 2*sb + half;
                    const float w0 = d[0] * (((const int8_t *) sc[0])[is] - 32);
                    const float w1 = d[1] * (((const int8_t *) sc[1])[is] - 32);
                    part[half][0] = D.x[0]*w0;
                    part[half][1] = D.x[1]*w0;
                    part[half][2] = D.x[2]*w1;
                    part[half][3] = D.x[3]*w1;
                }
                const float y0 = __low2float(ys0[sb].ds);
                const float y1 = __low2float(ys1[sb].ds);
                acc[0] += (part[0][0] + part[1][0]) * y0;
                acc[1] += (part[0][1] + part[1][1]) * y1;
                acc[2] += (part[0][2] + part[1][2]) * y0;
                acc[3] += (part[0][3] + part[1][3]) * y1;
            }
        }
    }
};

// q6_K: 16 int8 scales per block, one per 16 values
template <>
struct mmvq_mma<GGML_TYPE_Q6_K> {
    static constexpr int min_cols   = 7;
    static constexpr int min_blocks = 1;
    static __device__ __forceinline__ void block(const char * x0, const char * x1, const block_q8_1 * yb,
            const block_q8_1 * ys0, const block_q8_1 * ys1, float * acc) {
        const block_q6_K * b[2] = {(const block_q6_K *) x0, (const block_q6_K *) x1};
        const int t = threadIdx.x % 4;

        float d[2];
        int   sc[2][4];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            d[r] = __half2float(b[r]->d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sc[r][i] = get_int_b2(b[r]->scales, i);
            }
        }

#pragma unroll
        for (int n = 0; n < QK_K/128; ++n) {
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int sb = 4*n + p;
                float part[2][4];
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    mmvq_mma_tile_a16 A;
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        // values 32p + 16*half + 4t.. of this 128-value half: ql low or high nibble, 2 bits of qh
                        const int ql = get_int_b2(b[r]->ql, 16*n + 8*(p & 1) + 4*half + t);
                        const int qh = get_int_b2(b[r]->qh,  8*n + 4*half + t);
                        const int v  = ((ql >> 4*(p >> 1)) & 0x0F0F0F0F) | (((qh >> 2*p) & 0x03030303) << 4);
                        A.x[r] = __vsub4(v, 0x20202020);
                    }
                    mmvq_mma_tile_d D;
                    ggml_cuda_mma::mma(D, A, mmvq_mma_load_b16(yb + sb, half));
                    const int is = 8*n + 2*p + half;
                    const float w0 = d[0] * ((const int8_t *) sc[0])[is];
                    const float w1 = d[1] * ((const int8_t *) sc[1])[is];
                    part[half][0] = D.x[0]*w0;
                    part[half][1] = D.x[1]*w0;
                    part[half][2] = D.x[2]*w1;
                    part[half][3] = D.x[3]*w1;
                }
                const float y0 = __low2float(ys0[sb].ds);
                const float y1 = __low2float(ys1[sb].ds);
                acc[0] += (part[0][0] + part[1][0]) * y0;
                acc[1] += (part[0][1] + part[1][1]) * y1;
                acc[2] += (part[0][2] + part[1][2]) * y0;
                acc[3] += (part[0][3] + part[1][3]) * y1;
            }
        }
    }
};

template <>
struct mmvq_mma<GGML_TYPE_IQ3_S> {
    static constexpr int min_cols   = 4;
    static constexpr int min_blocks = 3;
    // values 4p..4p+3 of sub-block sb: a grid entry, its 9th index bit from qh, and 4 sign bits
    static __device__ __forceinline__ int quad(const block_iq3_s * b, const int sb, const int p, const int qh) {
        const int grid = mmvq_mma_iq3s_grid[b->qs[8*sb + p] | (((qh >> p) & 1) << 8)];
        return mmvq_mma_apply_signs(grid, b->signs[4*sb + p/2] >> 4*(p % 2));
    }
    static __device__ __forceinline__ void block(const char * x0, const char * x1, const block_q8_1 * yb,
            const block_q8_1 * ys0, const block_q8_1 * ys1, float * acc) {
        const block_iq3_s * b0 = (const block_iq3_s *) x0;
        const block_iq3_s * b1 = (const block_iq3_s *) x1;
        const int t = threadIdx.x % 4;

        const float d0 = __half2float(b0->d);
        const float d1 = __half2float(b1->d);
        const int qh0[2] = {get_int_b2(b0->qh, 0), get_int_b2(b0->qh, 1)};
        const int qh1[2] = {get_int_b2(b1->qh, 0), get_int_b2(b1->qh, 1)};
        const int sc0 = get_int_b2(b0->scales, 0);
        const int sc1 = get_int_b2(b1->scales, 0);

#pragma unroll
        for (int sb = 0; sb < QK_K/32; ++sb) {
            const int h0 = qh0[sb/4] >> 8*(sb % 4);
            const int h1 = qh1[sb/4] >> 8*(sb % 4);
            mmvq_mma_tile_a A;
            A.x[0] = quad(b0, sb, t,     h0);
            A.x[1] = quad(b1, sb, t,     h1);
            A.x[2] = quad(b0, sb, t + 4, h0);
            A.x[3] = quad(b1, sb, t + 4, h1);
            mmvq_mma_tile_d D;
            ggml_cuda_mma::mma(D, A, mmvq_mma_load_b(yb + sb));
            const float w0 = d0 * (1 + 2*((sc0 >> 4*sb) & 0x0F));
            const float w1 = d1 * (1 + 2*((sc1 >> 4*sb) & 0x0F));
            mmvq_mma_scale(acc, D, w0, w1, __low2float(ys0[sb].ds), __low2float(ys1[sb].ds));
        }
    }
};

template <>
struct mmvq_mma<GGML_TYPE_IQ3_XXS> {
    static constexpr int min_cols   = 4;
    static constexpr int min_blocks = 1;
    // values 4p..4p+3 of a sub-block: a grid entry and 4 of the 8 signs coded by 7 bits of aux
    static __device__ __forceinline__ int quad(const uint8_t * qs, const uint32_t aux, const int p) {
        const int grid = mmvq_mma_iq3xxs_grid[qs[p]];
        return mmvq_mma_apply_signs_sel(grid, unpack_ksigns(aux >> 7*(p/2)), p % 2 ? 0x80402010 : 0x08040201);
    }
    static __device__ __forceinline__ void block(const char * x0, const char * x1, const block_q8_1 * yb,
            const block_q8_1 * ys0, const block_q8_1 * ys1, float * acc) {
        const block_iq3_xxs * b0 = (const block_iq3_xxs *) x0;
        const block_iq3_xxs * b1 = (const block_iq3_xxs *) x1;
        const int t = threadIdx.x % 4;

        const float d0 = __half2float(b0->d);
        const float d1 = __half2float(b1->d);

#pragma unroll
        for (int sb = 0; sb < QK_K/32; ++sb) {
            // the 4 bytes of scale and signs of each sub-block follow the 64 grid indices
            const uint32_t aux0 = get_int_b2(b0->qs, QK_K/16 + sb);
            const uint32_t aux1 = get_int_b2(b1->qs, QK_K/16 + sb);
            mmvq_mma_tile_a A;
            A.x[0] = quad(b0->qs + 8*sb, aux0, t);
            A.x[1] = quad(b1->qs + 8*sb, aux1, t);
            A.x[2] = quad(b0->qs + 8*sb, aux0, t + 4);
            A.x[3] = quad(b1->qs + 8*sb, aux1, t + 4);
            mmvq_mma_tile_d D;
            ggml_cuda_mma::mma(D, A, mmvq_mma_load_b(yb + sb));
            const float w0 = d0 * (0.5f + (aux0 >> 28)) * 0.5f;
            const float w1 = d1 * (0.5f + (aux1 >> 28)) * 0.5f;
            mmvq_mma_scale(acc, D, w0, w1, __low2float(ys0[sb].ds), __low2float(ys1[sb].ds));
        }
    }
};

// iq2_s and iq2_xs: 8-value grid entries and one scale per 16 values, so two k16 mmas per sub-block
template <ggml_type type>
static __device__ __forceinline__ void mmvq_mma_block_iq2(const char * x0, const char * x1, const block_q8_1 * yb,
        const block_q8_1 * ys0, const block_q8_1 * ys1, float * acc) {
    using block_t = std::conditional_t<type == GGML_TYPE_IQ2_S, block_iq2_s, block_iq2_xs>;
    const block_t * b[2] = {(const block_t *) x0, (const block_t *) x1};
    const int t = threadIdx.x % 4;

    const float d[2] = {__half2float(b[0]->d), __half2float(b[1]->d)};

    // values 4q..4q+3 of sub-block sb, q = 0..7
    const auto quad = [&](const int r, const int sb, const int q) -> int {
        const int l = q/2;
        if constexpr (type == GGML_TYPE_IQ2_S) {
            const int idx = b[r]->qs[4*sb + l] | (((b[r]->qh[sb] >> 2*l) & 0x03) << 8);
            const uint2 entry = mmvq_mma_iq2s_grid[idx];
            const int grid = q % 2 ? entry.y : entry.x;
            return mmvq_mma_apply_signs(grid, b[r]->qs[QK_K/8 + 4*sb + l] >> 4*(q % 2));
        } else {
            const int v = b[r]->qs[4*sb + l];
            const uint2 entry = mmvq_mma_iq2xs_grid[v & 511];
            const int grid = q % 2 ? entry.y : entry.x;
            return mmvq_mma_apply_signs_sel(grid, unpack_ksigns(v >> 9), q % 2 ? 0x80402010 : 0x08040201);
        }
    };

#pragma unroll
    for (int sb = 0; sb < QK_K/32; ++sb) {
        const int ls[2] = {b[0]->scales[sb], b[1]->scales[sb]};
        float part[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            mmvq_mma_tile_a16 A;
            A.x[0] = quad(0, sb, 4*half + t);
            A.x[1] = quad(1, sb, 4*half + t);
            mmvq_mma_tile_d D;
            ggml_cuda_mma::mma(D, A, mmvq_mma_load_b16(yb + sb, half));
            const float w0 = d[0] * (0.5f + ((ls[0] >> 4*half) & 0x0F)) * 0.25f;
            const float w1 = d[1] * (0.5f + ((ls[1] >> 4*half) & 0x0F)) * 0.25f;
            part[half][0] = D.x[0]*w0;
            part[half][1] = D.x[1]*w0;
            part[half][2] = D.x[2]*w1;
            part[half][3] = D.x[3]*w1;
        }
        const float y0 = __low2float(ys0[sb].ds);
        const float y1 = __low2float(ys1[sb].ds);
        acc[0] += (part[0][0] + part[1][0]) * y0;
        acc[1] += (part[0][1] + part[1][1]) * y1;
        acc[2] += (part[0][2] + part[1][2]) * y0;
        acc[3] += (part[0][3] + part[1][3]) * y1;
    }
}

template <>
struct mmvq_mma<GGML_TYPE_IQ2_S> {
    static constexpr int min_cols   = 4;
    static constexpr int min_blocks = 1;
    static __device__ __forceinline__ void block(const char * x0, const char * x1, const block_q8_1 * yb,
            const block_q8_1 * ys0, const block_q8_1 * ys1, float * acc) {
        mmvq_mma_block_iq2<GGML_TYPE_IQ2_S>(x0, x1, yb, ys0, ys1, acc);
    }
};

template <>
struct mmvq_mma<GGML_TYPE_IQ2_XS> {
    static constexpr int min_cols   = 4;
    static constexpr int min_blocks = 1;
    static __device__ __forceinline__ void block(const char * x0, const char * x1, const block_q8_1 * yb,
            const block_q8_1 * ys0, const block_q8_1 * ys1, float * acc) {
        mmvq_mma_block_iq2<GGML_TYPE_IQ2_XS>(x0, x1, yb, ys0, ys1, acc);
    }
};

template <ggml_type type>
__launch_bounds__(MMVQ_MMA_NWARPS*WARP_SIZE, mmvq_mma<type>::min_blocks)
static __global__ void mul_mat_vec_q_mma(
        const char * __restrict__ vx, const block_q8_1 * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int64_t stride_row_x, const int stride_col_y, const int stride_col_dst) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int bs  = ggml_cuda_type_traits<type>::bs;
    constexpr int qbs = qk/QK8_1; // q8_1 blocks per quant block

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int g    = lane / 4;
    const int t    = lane % 4;
    const int row0 = 16*blockIdx.x;

    // Each warp stages its 16-row tile of a quant block in shared memory with loads that walk the rows word by word,
    // and the fragments are read from there: reading them from global memory directly touches 8 rows per load.
    // Quant blocks of 2-byte aligned types may start mid-word; their staged words start at the word below and each tile
    // row keeps its global address modulo 4. The last word can reach 2 bytes past the block, which stays inside the
    // tensor's allocation: ggml pads allocations to the buffer alignment. A tile row is 4 mod 8 words long, so
    // fragment reads hit 32 banks.
    constexpr int align      = bs % 4 == 0 ? 4 : 2;
    constexpr int copy_words = (bs + (4 - align) + 3) / 4;
    constexpr int tile_words = copy_words + ((4 - copy_words) % 8 + 8) % 8;
    __shared__ int tile_shared[MMVQ_MMA_NWARPS][16*tile_words];
    int * tile = tile_shared[warp];

    const int64_t row_bytes = stride_row_x * bs;
    // rows and columns past the end read valid data whose results are never written
    const int64_t off0 = min(row0 + g,     nrows_x - 1) * row_bytes;
    const int64_t off1 = min(row0 + g + 8, nrows_x - 1) * row_bytes;
    const block_q8_1 * yb  = vy + min(g,       ncols_dst - 1) * stride_col_y;
    const block_q8_1 * ys0 = vy + min(2*t,     ncols_dst - 1) * stride_col_y;
    const block_q8_1 * ys1 = vy + min(2*t + 1, ncols_dst - 1) * stride_col_y;

    mmvq_mma_stage<type>();
    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nblocks = ncols_x / qk;
    for (int kb = warp; kb < nblocks; kb += MMVQ_MMA_NWARPS) {
#pragma unroll
        for (int i0 = 0; i0 < 16*copy_words; i0 += WARP_SIZE) {
            const int i = i0 + lane;
            if (i0 + WARP_SIZE <= 16*copy_words || i < 16*copy_words) {
                const int r = i / copy_words;
                const int w = i % copy_words;
                const int64_t off = min(row0 + r, nrows_x - 1) * row_bytes + kb*bs;
                tile[r*tile_words + w] = ((const int *) (vx + (off & ~int64_t(3))))[w];
            }
        }
        __syncwarp();
        const char * x0 = (const char *) (tile +  g     *tile_words) + ((off0 + kb*bs) & 3);
        const char * x1 = (const char *) (tile + (g + 8)*tile_words) + ((off1 + kb*bs) & 3);
        mmvq_mma<type>::block(x0, x1, yb + kb*qbs, ys0 + kb*qbs, ys1 + kb*qbs, acc);
        __syncwarp();
    }

    __shared__ float acc_shared[MMVQ_MMA_NWARPS - 1][4][WARP_SIZE];
    if (warp > 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc_shared[warp - 1][i][lane] = acc[i];
        }
    }
    __syncthreads();
    if (warp > 0) {
        return;
    }
#pragma unroll
    for (int w = 0; w < MMVQ_MMA_NWARPS - 1; ++w) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[i] += acc_shared[w][i][lane];
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = row0 + g + 8*(i / 2);
        const int col = 2*t + i % 2;
        if (row < nrows_x && col < ncols_dst) {
            dst[col*stride_col_dst + row] = acc[i];
        }
    }
#else
    GGML_UNUSED_VARS(vx, vy, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

template <ggml_type type>
static void mul_mat_vec_q_mma_launch(
        const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x, const int ncols_dst,
        const int64_t stride_row_x, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    const dim3 block_nums((nrows_x + 15) / 16, 1, 1);
    const dim3 block_dims(WARP_SIZE, MMVQ_MMA_NWARPS, 1);
    mul_mat_vec_q_mma<type><<<block_nums, block_dims, 0, stream>>>(
        (const char *) vx, (const block_q8_1 *) vy, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y,
        stride_col_dst);
}

#define MMVQ_MMA_TYPES(X) \
    X(GGML_TYPE_Q3_K)     \
    X(GGML_TYPE_Q4_K)     \
    X(GGML_TYPE_Q5_K)     \
    X(GGML_TYPE_Q6_K)     \
    X(GGML_TYPE_IQ2_XS)   \
    X(GGML_TYPE_IQ2_S)    \
    X(GGML_TYPE_IQ3_XXS)  \
    X(GGML_TYPE_IQ3_S)    \
    X(GGML_TYPE_IQ4_XS)

// whether a dense src0 with nrows rows times ncols columns goes to the tensor-core kernel
static bool mul_mat_vec_q_mma_should_use(const ggml_type type, const int cc, const int64_t nrows, const int64_t ncols) {
    if (!ampere_mma_available(cc) || ncols > 8 || nrows < MMVQ_MMA_MIN_ROWS) {
        return false;
    }
    switch (type) {
#define MMVQ_MMA_CASE(t) case t: return ncols >= mmvq_mma<t>::min_cols;
        MMVQ_MMA_TYPES(MMVQ_MMA_CASE)
#undef MMVQ_MMA_CASE
        default:
            return false;
    }
}

static void mul_mat_vec_q_mma_switch_type(
        const void * vx, const ggml_type type, const void * vy, float * dst, const int ncols_x, const int nrows_x,
        const int ncols_dst, const int64_t stride_row_x, const int stride_col_y, const int stride_col_dst,
        cudaStream_t stream) {
    switch (type) {
#define MMVQ_MMA_CASE(t)                                                                                              \
        case t:                                                                                                       \
            mul_mat_vec_q_mma_launch<t>(vx, vy, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y,         \
                                        stride_col_dst, stream);                                                      \
            break;
        MMVQ_MMA_TYPES(MMVQ_MMA_CASE)
#undef MMVQ_MMA_CASE
        default:
            GGML_ABORT("unsupported type for the tensor-core MMVQ kernel");
    }
}

#undef MMVQ_MMA_TYPES

#else

static bool mul_mat_vec_q_mma_should_use(const ggml_type type, const int cc, const int64_t nrows, const int64_t ncols) {
    GGML_UNUSED_VARS(type, cc, nrows, ncols);
    return false;
}

static void mul_mat_vec_q_mma_switch_type(
        const void * vx, const ggml_type type, const void * vy, float * dst, const int ncols_x, const int nrows_x,
        const int ncols_dst, const int64_t stride_row_x, const int stride_col_y, const int stride_col_dst,
        cudaStream_t stream) {
    GGML_UNUSED_VARS(vx, type, vy, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                     stream);
    GGML_ABORT("the tensor-core MMVQ kernel is only built for NVIDIA GPUs");
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
