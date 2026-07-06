#pragma once

#include "llama.h"
#include "llama-graph.h"

#include <map>
#include <memory>
#include <functional>

struct llama_ubatch;

class llama_batch_allocr;

class llama_io_write_i;
class llama_io_read_i;

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    llama_context_type ctx_type;

    llama_memory_t mem_other;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llama_memory_status_is_fail(llama_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    using layer_share_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    // --- page-granular eviction (partial KV reclaim to host RAM) ---
    // evict up to `max_bytes` of seq_id's committed KV to host RAM in chunks of `chunk_bytes`; returns bytes freed.
    // keeps the sequence's cells/metadata intact (unlike seq_rm) so seq_restore brings it back byte-identically.
    // default no-op: only demand-paged KV caches implement this.
    virtual size_t seq_evict  (llama_seq_id seq_id, size_t max_bytes, size_t chunk_bytes) {
        (void) seq_id; (void) max_bytes; (void) chunk_bytes;
        return 0;
    }
    // restore all pages of seq_id previously evicted by seq_evict; returns false on device OOM.
    virtual bool   seq_restore(llama_seq_id seq_id) {
        (void) seq_id;
        return true;
    }

    // --- read-only prefix page sharing (system-prompt reuse across sequences) ---
    // map the leading `n_tokens` of `src`'s KV pages into `dst`'s per-stream region READ-ONLY (no copy, no extra
    // VRAM), marking dst's prefix cells present. returns tokens actually shared (0 = caller must process the
    // prefix normally). default no-op: only demand-paged, per-stream (--kv-share) KV caches implement this.
    virtual llama_pos seq_share_prefix(llama_seq_id dst, llama_seq_id src, llama_pos n_tokens, llama_pos * out_aliased = nullptr) {
        (void) dst; (void) src; (void) n_tokens;
        if (out_aliased) { *out_aliased = 0; }
        return 0;
    }
    // the cell-count quantum a shared prefix snaps to (a share is always a whole multiple), or 0 if this memory
    // cannot page-share. lets a caller decide whether a cross-sequence borrow would beat its own prefix reuse.
    virtual llama_pos seq_share_align() const { return 0; }

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
