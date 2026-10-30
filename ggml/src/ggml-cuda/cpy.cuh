#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

// n copies of src0's shape into src1's shape, copy k reading at src0->data + k*src_step and writing at
// src1->data + k*dst_step (byte steps, either sign). F32 and F16, no type conversion.
void ggml_cuda_cpy_batched(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1, int n,
                           int64_t src_step, int64_t dst_step);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
