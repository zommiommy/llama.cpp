#include "common.cuh"

void ggml_cuda_moe_cache_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// whether device can run the lookup dst: its planner must fit the shared memory of a block
bool ggml_cuda_moe_cache_fill_supported(const ggml_tensor * dst, int device);
void ggml_cuda_moe_cache_join(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// copies the experts a router expects the next lookup to miss on stream GGML_CUDA_MAX_STREAMS - 1, see ggml_moe_cache_prefetch
void ggml_cuda_moe_cache_prefetch(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_moe_cache_prefetch_supported(const ggml_tensor * dst, int device);
// the current stream waits for the copies of the last prefetch, if they may still run
void ggml_cuda_moe_cache_prefetch_join(ggml_backend_cuda_context & ctx);

// copies n ranges between buffers of the current device on the stream, in launches of up to 128 ranges
void ggml_cuda_moe_cache_copy_ranges(cudaStream_t stream, const void * const * src, void * const * dst, const size_t * size, int n);
