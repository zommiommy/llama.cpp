#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // with free > 0 the cache is elastic: it holds memory only for its active slots, at most size, and adapt() keeps free bytes of each device free
    // cpu_experts > 0: the host computes up to that many missing experts of each layer of a small batch, beside the device
    // tail_skip > 0: a missing expert that no token of a small batch selects at a rank below tail_skip is skipped, its output is zero
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts,
            size_t size, size_t free, int32_t cpu_experts, int32_t tail_skip);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the slots of the experts ids [n_expert_used, n_tokens] of layer il, looked up and filled on the device without the host,
    // nullptr if the device cannot read the host experts: then the slot map is copied with the copy callback
    // with cpu and cpu_experts, the plain SwiGLU experts of x [n_embd, n_tokens] can be left to the host when they are missing, and
    // with tail_skip missing experts at the tail of the routes are skipped: the outputs of the down projection then go through
    // build_join
    ggml_tensor * build_fill(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * x, int32_t il, bool cpu) const;

    // the outputs of the experts [n_embd, n_expert_used, n_tokens] with those the lookup fill left to the host or skipped
    ggml_tensor * build_join(ggml_context * ctx, ggml_tensor * experts, ggml_tensor * fill) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    // for prompt batches of n_tokens: point the copy dst of the host experts w at the ring slot that holds their layer
    // the ring streams all the experts of each layer on a stream of its own, the next layer while the current one computes
    // returns false if the batch is too small for the ring or w is not cached
    bool ring_bind(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t n_tokens);

    // before the graph of a ubatch of n_tokens with n_outputs output rows runs: a prompt ubatch starts filling the ring with
    // its first layers, and the last layer, which computes only the output rows, reads the ring only if they are many
    void ring_prefetch(int64_t n_tokens, int64_t n_outputs);

    // elastic cache: give back the memory other buffers grew into, and with grow take free memory beyond the target
    // a batch of n_tokens too small for the ring frees it
    void adapt(bool grow, int64_t n_tokens);

    // elastic cache: shrink so that size more bytes than the target are free on dev, for a buffer about to grow by size
    void make_room(ggml_backend_dev_t dev, size_t size);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
