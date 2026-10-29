#include "common.cuh"

void ggml_cuda_op_top_k_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_op_top_k_cells_supported(const ggml_tensor * op);
void ggml_cuda_op_top_k_cells(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_range_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
