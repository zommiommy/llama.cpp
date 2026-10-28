#include "common.cuh"

void ggml_cuda_moe_cache_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_moe_cache_fill_supported(const ggml_tensor * dst);
void ggml_cuda_moe_cache_join(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
