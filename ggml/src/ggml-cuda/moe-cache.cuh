#include "common.cuh"

void ggml_cuda_moe_cache_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_moe_cache_fill_supported(const ggml_tensor * dst);
void ggml_cuda_moe_cache_join(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// copies n ranges between buffers of the current device on the stream, in launches of up to 128 ranges
void ggml_cuda_moe_cache_copy_ranges(cudaStream_t stream, const void * const * src, void * const * dst, const size_t * size, int n);
