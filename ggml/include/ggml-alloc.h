#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct      ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct             ggml_backend * ggml_backend_t;

// Tensor allocator
struct ggml_tallocr {
    ggml_backend_buffer_t buffer;
    void * base;
    size_t alignment;
    size_t offset;
};

GGML_API struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);
GGML_API enum ggml_status    ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor);

// Graph allocator
/*
  Example usage:
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    // optional: create a worst-case graph and reserve the buffers to avoid reallocations
    ggml_gallocr_reserve(galloc, build_graph(max_batch));

    // allocate the graph
    struct ggml_cgraph * graph = build_graph(batch);
    ggml_gallocr_alloc_graph(galloc, graph);

    printf("compute buffer size: %zu bytes\n", ggml_gallocr_get_buffer_size(galloc, 0));

    // evaluate the graph
    ggml_backend_graph_compute(backend, graph);
*/

// special tensor flags for use with the graph allocator:
//   ggml_set_input(): all input tensors are allocated at the beginning of the graph in non-overlapping addresses
//   ggml_set_output(): output tensors are never freed and never overwritten

typedef struct ggml_gallocr * ggml_gallocr_t;

GGML_API ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);
GGML_API ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);
GGML_API void           ggml_gallocr_free(ggml_gallocr_t galloc);

// called when a reservation needs size_new bytes of a buffer of type buft that holds size_cur bytes (0 if it has no buffer yet)
// returns the size the buffer gets, at least size_new: a buffer that has to grow is replaced by one of that size, which can
// leave room for the next reservations, and the call can make room for it; a larger buffer is kept if the returned size is
// size_cur or more, else it is replaced by one of the returned size
typedef size_t (*ggml_gallocr_resize_callback)(ggml_backend_buffer_type_t buft, size_t size_cur, size_t size_new, void * user_data);
GGML_API void           ggml_gallocr_set_resize_callback(ggml_gallocr_t galloc, ggml_gallocr_resize_callback callback, void * user_data);

// pre-allocate buffers from a measure graph - does not allocate or modify the graph
// call with a worst-case graph to avoid buffer reallocations
// not strictly required for single buffer usage: ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
// returns false if the buffer allocation failed
// ggml_gallocr_resrve_n_size writes the buffer sizes per galloc buffer that would be allocated by ggml_gallocr_reserve_n to sizes
GGML_API bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
GGML_API void ggml_gallocr_reserve_n_size(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids,
    size_t * sizes);
GGML_API bool ggml_gallocr_reserve_n(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids);

// automatic reallocation if the topology changes when using a single buffer
// returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
GGML_API bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);

GGML_API size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);

// Utils
// Create a buffer and allocate all the tensors in a ggml_context

// returns the size of the buffer that would be allocated by ggml_backend_alloc_ctx_tensors_from_buft. returns 0 on failure
GGML_API size_t                       ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
// returns NULL on failure or if all tensors in ctx are already allocated or zero-sized
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
