#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t n_active = 0; // slots [0, n_active) hold experts and are in the lists, the rest have no memory

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots, int32_t n_active) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.assign(n_slots, -1);
        next.assign(n_slots, -1);
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
        set_active(n_active);
    }

    // unlink slot s from its list
    void unlink(int32_t s) {
        const bool seg = is_protected[s];
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[seg] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[seg] = prev[s];
        }
        prev[s] = -1;
        next[s] = -1;
        if (seg) {
            is_protected[s] = false;
            n_protected--;
        }
    }

    bool linked(int32_t s) const {
        return prev[s] >= 0 || next[s] >= 0 || head[0] == s || head[1] == s;
    }

    // slot d takes the expert of slot s and its place in the lists; the expert of d is dropped, s is left empty and unlinked
    void relocate(int32_t s, int32_t d) {
        if (key_of[d] >= 0) {
            slot_map[key_of[d] / n_expert][key_of[d] % n_expert] = -1;
        }
        unlink(d);
        const bool seg = is_protected[s];
        prev[d] = prev[s];
        next[d] = next[s];
        if (prev[s] >= 0) {
            next[prev[s]] = d;
        } else {
            head[seg] = d;
        }
        if (next[s] >= 0) {
            prev[next[s]] = d;
        } else {
            tail[seg] = d;
        }
        is_protected[d] = seg; // in the place of s, which leaves the count
        key_of[d] = key_of[s];
        slot_map[key_of[d] / n_expert][key_of[d] % n_expert] = d;
        key_of[s] = -1;
        prev[s] = -1;
        next[s] = -1;
        is_protected[s] = false;
    }

    // grow or shrink the active slots; a new slot is empty and goes first in line for eviction,
    // a removed slot forgets its expert
    void set_active(int32_t n) {
        GGML_ASSERT(n >= 0 && n <= n_slots);
        for (int32_t s = n_active - 1; s >= n; --s) {
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
                key_of[s] = -1;
            }
            if (linked(s)) {
                unlink(s);
            }
        }
        // pushed from the top so the lowest slot is evicted first: misses of consecutive experts get consecutive slots
        for (int32_t s = n - 1; s >= n_active; --s) {
            prev[s] = -1;
            next[s] = head[0];
            if (head[0] >= 0) {
                prev[head[0]] = s;
            } else {
                tail[0] = s;
            }
            head[0] = s;
        }
        n_active = n;
        while (n_protected > n_active/2) {
            move(head[1], false);
        }
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_active/2) {
            move(head[1], false);
        }
    }

    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // the active slots in eviction order: probation first, then protected, each from its least recently used end
    void eviction_order(std::vector<int32_t> & out) const {
        out.clear();
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                out.push_back(s);
            }
        }
    }

    // expert e of layer il was filled into slot s elsewhere, as plan() would have: s joins probation
    void filled(int32_t il, int32_t e, int32_t s) {
        if (key_of[s] >= 0) {
            slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
        }
        key_of[s] = il*n_expert + e;
        slot_map[il][e] = s;
        move(s, false);
    }

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_active) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // the missing experts a lookup leaves to the host: the device posts their pairs and inputs to pinned memory and computes the
    // cached experts meanwhile, a pool of host threads computes the posted ones, and ggml_moe_cache_join writes them back
    //
    // the threads take the work items of a post from one counter, in order: the quantized input of each token, blocks of gate and
    // up rows of each expert, then blocks of down rows; the first thread to need the input of the down rows of an expert computes
    // it once the gate and up blocks of the expert are done, so a post needs no barrier
    struct cpu_share {
        // the dot products of the expert tensors of a layer, and the quantization of their inputs
        struct cpu_layer {
            const ggml_tensor * w[3] = {};      // gate, up, down
            ggml_vec_dot_t      dot[3] = {};
            ggml_from_float_t   quant[3] = {};
            size_t              q_size[3] = {}; // bytes of a quantized input row
            bool                same_q = false; // gate and up read the same quantized input
            int64_t             rows[2] = {};   // rows of a block of gate and up, of down
        };

        ggml_backend_buffer_ptr buf;  // pinned, the device reads and writes it directly
        ggml_moe_cache_xchg xchg = {};
        int64_t n_embd = 0;
        int64_t n_ff   = 0;
        std::vector<cpu_layer> layers;

        int n_threads = 0;
        std::vector<std::thread> threads; // the first waits for the posts and publishes their jobs
        std::atomic<bool>    stop{false};
        std::atomic<int64_t> t_active_us{0}; // the threads poll until a while after the last batch, then sleep

        // the job of the current post, written before it is published
        alignas(64) std::atomic<uint32_t> job{0};
        alignas(64) std::atomic<int32_t>  next{0};   // the next work item
        alignas(64) std::atomic<int32_t>  n_done{0}; // threads but the first done with the job
        alignas(64) int32_t il = 0;
        int32_t n_tokens  = 0;
        int32_t n_experts = 0;
        int32_t n_items   = 0;
        int32_t items_gu  = 0; // blocks of gate and up rows of an expert
        int32_t items_d   = 0; // blocks of down rows of an expert
        std::vector<int32_t> tok_pair; // a pair of each token, which holds its input
        std::vector<int32_t> pair_tok; // the token of each pair
        std::vector<int32_t> exp_id;   // the expert of each expert of the job
        std::vector<int32_t> exp_p0;   // the pairs of expert i are order[exp_p0[i], exp_p0[i + 1])
        std::vector<int32_t> order;
        std::unique_ptr<std::atomic<int32_t>[]> tok_ready; // the quantized input of the token is written
        std::unique_ptr<std::atomic<int32_t>[]> gu_left;   // blocks of gate and up rows of the expert not done yet
        std::unique_ptr<std::atomic<int32_t>[]> act;       // the input of down of the expert: 0 missing, 1 being computed, 2 ready

        std::vector<uint8_t> xq; // per token: the quantized input of gate, then of up
        size_t xq_stride = 0;
        std::vector<float>   gu; // per pair: the outputs of gate, then of up; the gate outputs turn into the input of down
        std::vector<uint8_t> aq; // per pair: the quantized input of down
        size_t aq_stride = 0;

        // used by the first thread only
        std::vector<int32_t> tok_of; // [max_batch] the job token of a batch token, -1 if none
        std::vector<int32_t> exp_of; // [n_expert]  the job expert of an expert, -1 if none
        std::vector<int32_t> cursor;

        ~cpu_share() {
            stop = true;
            for (std::thread & t : threads) {
                t.join();
            }
        }
    };

    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget
        bool elastic = false;  // the banks are demand-paged and only their active slots hold memory

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        // when the device reads the host experts directly, small batches look up and fill the cache on it (ggml_moe_cache_fill):
        // the host replays the log of each batch into the SLRU and hands the device its eviction order before the next one
        bool fill = false;
        ggml_context_ptr ctx_fill;
        ggml_backend_buffer_ptr buf_fill;
        ggml_tensor * dfill = nullptr;  // I32: the slot maps, the states of the groups, then the log
        ggml_tensor * dmap  = nullptr;  // I32 [n_expert, n_layer], the slot maps of the layers of the device
        ggml_tensor * dlog  = nullptr;  // I32 [4 + 3*cap], the hits and fills since the last push
        size_t log_off = 0;             // elements of dfill before the log
        // a push writes everything the host owns, up to the record count of the log, in one copy from this pinned image
        ggml_backend_buffer_ptr buf_img;
        int32_t * img = nullptr;
        int32_t log_seq  = 0;           // the device numbers its posts to the host in the log header, a push keeps the number
        int32_t log_done = 0;           // records of the log already replayed

        std::unique_ptr<cpu_share> cpu; // nullptr when every missing expert is copied to the device
        int64_t cpu_posts = 0;          // layers the host computed experts for, written by its thread
        int64_t cpu_us    = 0;          // and the time it took
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;     // capacity
        int32_t n_min = 0;       // an elastic group keeps enough slots for a small batch
        size_t slot_bytes = 0;   // device memory of one slot over all banks
        moe_cache_slru slru;
        ggml_tensor * dstate = nullptr; // I32, the state of ggml_moe_cache_fill on the device, a view of dfill
        size_t state_off = 0;           // its first element in dfill
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
        bool cpu = false;                   // the host can compute its experts: plain SwiGLU of separate gate, up and down
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    // an elastic cache resizes in steps of this much memory, so it is not resized for every token
    static constexpr size_t resize_step = 128*1024*1024;

    // generation batches check the free memory at most this often
    static constexpr int64_t adapt_interval_us = 200*1000;
    int64_t t_adapt_us = 0;

    int32_t n_expert_used;

    size_t free_target; // device memory the elastic cache keeps free, 0 for a cache of fixed size

    int32_t cpu_experts; // missing experts of a layer of a small batch the host may compute, 0 for none
    int32_t tail_skip;   // a missing expert no token of a small batch selects at a rank below this is skipped, 0 for none

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_fill;  // small batches looked up on the device
    stats stats_copy;  // experts copied from the cache for large batches
    stats stats_ring;  // experts uploaded to the ring
    stats stats_cpu;   // missing experts of small batches the host computed
    stats stats_skip;  // missing experts of small batches skipped at the tail of the routes

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;

    // scratch of the device lookup
    std::vector<int32_t> victims;
    std::vector<int32_t> log_records;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    // prompt batches of at least this many tokens use about every expert of a layer and compute long enough to hide its upload,
    // so the ring streams all of them: at 2048 tokens the upload takes longer than the compute and the ring's memory costs more
    // cache than the overlap saves (884 vs 898 t/s on an 8K prompt), at 8192 it reads 8K and 32K prompts 5% and 12% faster
    static constexpr int64_t ring_min = 4096;

    // the experts of a prompt batch, two layers at a time, filled on a stream of their own: the experts of the next layer
    // are uploaded while the current layer computes
    struct ring {
        ggml_backend_t copy = nullptr; // another backend of the device, for its stream
        ggml_backend_event_t filled[2] = { nullptr, nullptr };
        ggml_backend_event_t released  = nullptr;
        ggml_backend_buffer_ptr buf;   // two slots, allocated for prompt batches only
        size_t slot_bytes = 0;
        int32_t layer[2] = { -1, -1 }; // the layer each slot holds or is being filled with
        int32_t cur      = -1;         // the slot of the layer bound last
        int32_t bound    = -1;         // the layer bound last in this batch
        bool    failed   = false;      // no memory for the ring until the next small batch
    };
    std::vector<ring> rings; // by device

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts,
            size_t size, size_t free, int32_t cpu_experts, int32_t tail_skip) :
            n_expert_used(model.hparams.n_expert_used_max()), free_target(free), cpu_experts(cpu_experts), tail_skip(tail_skip),
            layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
                if (free_target > 0) {
                    // demand paging lets the banks reserve the whole budget and hold memory for the active slots only
                    ggml_backend_buffer_type_t buft_lazy = ggml_backend_dev_buffer_type_lazy(ggml_backend_get_device(d.backend));
                    if (buft_lazy != nullptr) {
                        d.buft    = buft_lazy;
                        d.elastic = true;
                    } else {
                        LLAMA_LOG_WARN("%s: %s has no demand paging, its MoE cache keeps a fixed size\n", __func__,
                            ggml_backend_dev_name(ggml_backend_get_device(d.backend)));
                    }
                }
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        rings.resize(devices.size());
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            {
                const llama_layer & ml = model.layers[il];
                layers[il].cpu = ml.ffn_gate_exps != nullptr && ml.ffn_up_exps != nullptr && ml.ffn_down_exps != nullptr &&
                    ml.ffn_gate_up_exps == nullptr &&
                    !ml.ffn_gate_exps_b && !ml.ffn_up_exps_b && !ml.ffn_down_exps_b &&
                    !ml.ffn_gate_exps_s && !ml.ffn_up_exps_s && !ml.ffn_down_exps_s &&
                    model.hparams.swiglu_clamp_exp[il] <= 1e-6f &&
                    std::none_of(experts.begin(), experts.end(), [&](ggml_tensor * t) { return model.hdmd.rot.count(t) > 0; });
            }
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            for (const ggml_tensor * t : g.ref) {
                g.slot_bytes += t->nb[2];
            }
            // an elastic group starts with the slots of the largest small batch and grows into free memory
            g.n_min = (int32_t) std::min<int64_t>(g.n_slots, std::min<int64_t>(max_batch*n_expert_used, n_expert));
            g.slru.init(model.layers.size(), n_expert, g.n_slots, d.elastic ? g.n_min : g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                // demand-paged banks hold no memory yet, and their pages are zeroed when committed
                if (!d.elastic) {
                    ggml_backend_buffer_clear(d.buf.get(), 0);
                }
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            for (group & g : groups) {
                if (g.n_slots > 0 && devices[g.id].elastic && !commit_slots(g, 0, g.slru.n_active)) {
                    throw std::runtime_error("failed to commit the MoE cache slots of a small batch");
                }
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }

            for (size_t id = 0; id < devices.size(); ++id) {
                init_fill(id);
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts%s%s\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0,
                d.elastic ? ", elastic: grows into free memory beyond the reserve and gives it back" : "",
                d.fill ? ", small batches look it up on the device" : "");
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)%s\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]),
                        d.elastic ? ", at most" : "");
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // the device memory of slots [s0, s1) of every bank, plus the slot after them: CUDA MMQ can read past the last expert
    bool commit_slots(const group & g, int32_t s0, int32_t s1) const {
        for (const ggml_tensor * bank : g.banks) {
            const size_t off = (const char *) bank->data - (const char *) ggml_backend_buffer_get_base(bank->buffer);
            if (!ggml_backend_buffer_ensure_range(bank->buffer, off + s0*bank->nb[2], (size_t) (s1 + 1 - s0)*bank->nb[2])) {
                return false;
            }
        }
        return true;
    }

    // give back the device memory of every bank above slot n, which stays for MMQ to read past slot n - 1: from
    // there up to the end of the bank, so the pages straddling the boundaries of earlier shrinks go too
    void release_slots(const group & g, int32_t n) const {
        for (const ggml_tensor * bank : g.banks) {
            const size_t off = (const char *) bank->data - (const char *) ggml_backend_buffer_get_base(bank->buffer);
            ggml_backend_buffer_release_range(bank->buffer, off + (n + 1)*bank->nb[2], (size_t) (g.n_slots - n)*bank->nb[2]);
        }
    }

    // resize the active slots of group g by about delta bytes, within [n_min, n_slots]
    void resize(group & g, int64_t delta) {
        const int32_t n_old = g.slru.n_active;
        int64_t n_new = n_old + delta/(int64_t) g.slot_bytes;
        n_new = std::max<int64_t>(g.n_min, std::min<int64_t>(g.n_slots, n_new));
        if (n_new > n_old) {
            // a failed commit keeps the slots committed so far
            constexpr int32_t chunk = 64;
            int32_t n = n_old;
            while (n < n_new) {
                const int32_t n_next = (int32_t) std::min<int64_t>(n + chunk, n_new);
                if (!commit_slots(g, n, n_next)) {
                    break;
                }
                n = n_next;
            }
            n_new = n;
            g.slru.set_active((int32_t) n_new);
        } else if (n_new < n_old) {
            keep_recent(g, (int32_t) n_new);
            g.slru.set_active((int32_t) n_new);
            release_slots(g, (int32_t) n_new);
        }
        if (n_new != n_old) {
            LLAMA_LOG_DEBUG("%s: %d -> %d of %d slots (%.0f MiB)\n", __func__, n_old, (int32_t) n_new, g.n_slots, n_new*g.slot_bytes/1024.0/1024.0);
        }
    }

    // before the slots from n_new up are released: the experts that would be evicted next are dropped instead of
    // the ones the released slots hold, which move down into the slots of the dropped experts
    void keep_recent(group & g, int32_t n_new) {
        moe_cache_slru & r = g.slru;
        const int32_t n_drop = r.n_active - n_new;
        std::vector<int32_t> next_out; // in eviction order: probation first, from its least recently used end
        std::vector<bool> out(r.n_active, false);
        for (int seg = 0; seg < 2 && (int32_t) next_out.size() < n_drop; ++seg) {
            for (int32_t s = r.head[seg]; s >= 0 && (int32_t) next_out.size() < n_drop; s = r.next[s]) {
                next_out.push_back(s);
                out[s] = true;
            }
        }
        ggml_backend_t backend = devices[g.id].backend;
        size_t i = 0;
        for (int32_t s = n_new; s < r.n_active; ++s) {
            if (out[s] || r.key_of[s] < 0) {
                continue;
            }
            while (next_out[i] >= n_new) {
                i++;
            }
            const int32_t d = next_out[i++];
            for (ggml_tensor * bank : g.banks) {
                ggml_reset(ctx_views.get());
                ggml_tensor * src = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], 1, bank->nb[1], bank->nb[2], s*bank->nb[2]);
                ggml_tensor * dst = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], 1, bank->nb[1], bank->nb[2], d*bank->nb[2]);
                ggml_backend_view_init(src);
                ggml_backend_view_init(dst);
                ggml_backend_tensor_copy_async(backend, backend, src, dst);
            }
            r.relocate(s, d);
        }
    }

    // grow or shrink the elastic groups so that each device keeps about free_target bytes free
    void adapt(bool grow, int64_t n_tokens) {
        for (size_t id = 0; id < devices.size(); ++id) {
            apply_log(id);
            if (devices[id].cpu) {
                devices[id].cpu->t_active_us = ggml_time_us();
            }
        }
        bool freed = false;
        for (size_t id = 0; id < rings.size(); ++id) {
            rings[id].bound = -1;
            if (n_tokens < ring_min) {
                freed = freed || rings[id].buf != nullptr;
                ring_release(id);
            }
        }
        // querying the free memory takes about a millisecond: generation checks it every few batches, as the KV grows slowly
        const int64_t t_now = ggml_time_us();
        const bool check = freed || n_tokens > max_batch || t_now - t_adapt_us >= adapt_interval_us;
        if (check) {
            t_adapt_us = t_now;
        }
        for (size_t id = 0; id < devices.size() && check; ++id) {
            const device & d = devices[id];
            if (!d.elastic || d.host_bytes == 0) {
                continue;
            }
            // shrink below the target and grow above it by a step, so a growing KV cache releases memory every few ubatches
            const int64_t surplus = free_surplus(id);
            int64_t delta = 0;
            if (surplus < 0) {
                delta = surplus - (int64_t) resize_step;
            } else if (grow && surplus > 2*(int64_t) resize_step) {
                delta = surplus - (int64_t) resize_step;
            }
            if (delta != 0) {
                resize_device(id, delta);
            }
        }
        for (size_t id = 0; id < devices.size(); ++id) {
            push_state(id);
        }
    }

    // shrink the elastic groups of dev so that size more bytes than the target are free
    void make_room(ggml_backend_dev_t dev, size_t size) {
        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (!d.elastic || d.host_bytes == 0 || ggml_backend_get_device(d.backend) != dev) {
                continue;
            }
            const int64_t surplus = free_surplus(id) - (int64_t) size;
            if (surplus < 0) {
                apply_log(id);
                resize_device(id, surplus);
                push_state(id);
            }
        }
    }

    // free memory of device id beyond the target
    int64_t free_surplus(size_t id) const {
        size_t free;
        size_t total;
        ggml_backend_dev_memory(ggml_backend_get_device(devices[id].backend), &free, &total);
        return (int64_t) free - (int64_t) free_target;
    }

    // the groups of a device share a change of its cache by the size of their experts, as they share the budget
    void resize_device(size_t id, int64_t delta) {
        for (group & g : groups) {
            if (g.id == (int32_t) id && g.n_slots > 0) {
                resize(g, (int64_t) ((double) delta*g.host_bytes/devices[id].host_bytes));
            }
        }
    }

    // device memory held by the slots
    size_t committed(size_t id) const {
        size_t res = 0;
        for (const group & g : groups) {
            if (g.id == (int32_t) id && g.n_slots > 0) {
                res += (g.slru.n_active + 1)*g.slot_bytes;
            }
        }
        return res;
    }

    // small batches can look up and fill the cache on the device if it reads every host expert of its layers directly
    void init_fill(size_t id) {
        device & d = devices[id];
        if (!d.ctx) {
            return;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
        ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(dev);
        if (buft_host == nullptr) {
            return;
        }
        int64_t n_expert = 0;
        size_t  n_groups = 0;
        for (const group & g : groups) {
            if (g.id != (int32_t) id || g.n_slots == 0) {
                continue;
            }
            n_expert = std::max<int64_t>(n_expert, g.ref[0]->ne[2]);
            n_groups++;
            for (int32_t il : g.layers) {
                for (const ggml_tensor * w : layers[il].experts) {
                    if (w->buffer == nullptr || ggml_backend_buffer_get_type(w->buffer) != buft_host || layers[il].experts.size() > 3) {
                        return;
                    }
                }
            }
        }
        if (n_groups == 0) {
            return;
        }

        // each record is a distinct expert of a layer of a batch, twice as many as a batch can need
        const int64_t n_ids   = max_batch*n_expert_used;
        const int64_t log_cap = 2*(int64_t) layers.size()*n_ids;

        // the slot maps, the state of each group and the log in one tensor, so that the host writes its part in one copy
        const size_t pad = 64;
        size_t n_fill = GGML_PAD((size_t) n_expert*layers.size(), pad);
        for (group & g : groups) {
            if (g.id == (int32_t) id && g.n_slots > 0) {
                g.state_off = n_fill;
                n_fill += GGML_PAD((size_t) GGML_MOE_CACHE_STATE_SIZE(g.n_slots, n_ids), pad);
            }
        }
        d.log_off = n_fill;
        n_fill += 4 + 3*log_cap;

        ggml_init_params params = {
            /*.mem_size   =*/ (3 + n_groups)*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        d.ctx_fill.reset(ggml_init(params));
        ggml_context * ctx = d.ctx_fill.get();
        d.dfill = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n_fill);
        d.dmap  = ggml_view_2d(ctx, d.dfill, n_expert, (int64_t) layers.size(), n_expert*sizeof(int32_t), 0);
        d.dlog  = ggml_view_1d(ctx, d.dfill, 4 + 3*log_cap, d.log_off*sizeof(int32_t));
        for (group & g : groups) {
            if (g.id == (int32_t) id && g.n_slots > 0) {
                g.dstate = ggml_view_1d(ctx, d.dfill, GGML_MOE_CACHE_STATE_SIZE(g.n_slots, n_ids), g.state_off*sizeof(int32_t));
            }
        }
        d.buf_fill.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev)));
        if (d.buf_fill) {
            d.buf_img.reset(ggml_backend_buft_alloc_buffer(buft_host, (d.log_off + 4)*sizeof(int32_t)));
        }
        if (!d.buf_fill || !d.buf_img) {
            LLAMA_LOG_WARN("%s: no memory for the device lookup of the MoE cache of %s\n", __func__, ggml_backend_dev_name(dev));
            for (group & g : groups) {
                if (g.id == (int32_t) id) {
                    g.dstate = nullptr;
                }
            }
            d.dfill = nullptr;
            d.dmap  = nullptr;
            d.dlog  = nullptr;
            d.buf_img.reset();
            d.buf_fill.reset();
            d.ctx_fill.reset();
            return;
        }
        // as weights, the lookups run on the device
        ggml_backend_buffer_set_usage(d.buf_fill.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(d.buf_img.get(), 0);
        d.img = (int32_t *) ggml_backend_buffer_get_base(d.buf_img.get());

        d.fill = true;
        push_state(id);
        init_cpu(id);
    }

    // up to cpu_experts missing experts of each layer of a small batch are computed by the host, with all cores but one
    void init_cpu(size_t id) {
        const int cpu_max = cpu_experts;
        if (cpu_max <= 0) {
            return;
        }
        device & d = devices[id];

        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        auto * get_traits = cpu_dev == nullptr ? nullptr : (decltype(ggml_get_type_traits_cpu) *)
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu_dev), "ggml_get_type_traits_cpu");
        if (get_traits == nullptr) {
            LLAMA_LOG_WARN("%s: the CPU backend has no dot products for the host experts, they stay on the device\n", __func__);
            return;
        }

        auto c = std::make_unique<cpu_share>();
        c->layers.resize(layers.size());
        int64_t n_expert = 0;
        for (const group & g : groups) {
            if (g.id != (int32_t) id || g.n_slots == 0) {
                continue;
            }
            for (int32_t il : g.layers) {
                const layer & l = layers[il];
                cpu_share::cpu_layer & cl = c->layers[il];
                bool ok = l.cpu;
                for (int i = 0; i < 3 && ok; ++i) {
                    const ggml_tensor * w = l.experts[i];
                    const ggml_type_traits_cpu * tw = get_traits(w->type);
                    const ggml_type tq = tw->vec_dot_type;
                    cl.w[i]      = w;
                    cl.dot[i]    = tw->vec_dot;
                    cl.quant[i]  = get_traits(tq)->from_float;
                    cl.q_size[i] = ggml_row_size(tq, w->ne[0]);
                    ok = cl.dot[i] != nullptr && cl.quant[i] != nullptr && w->ne[0] % ggml_blck_size(tq) == 0 && w->nb[0] == ggml_type_size(w->type);
                }
                const int64_t n_embd = l.experts[0]->ne[0];
                const int64_t n_ff   = l.experts[0]->ne[1];
                ok = ok && l.experts[1]->ne[0] == n_embd && l.experts[1]->ne[1] == n_ff && l.experts[2]->ne[0] == n_ff && l.experts[2]->ne[1] == n_embd &&
                    (c->n_embd == 0 || (c->n_embd == n_embd && c->n_ff == n_ff));
                if (!ok) {
                    LLAMA_LOG_WARN("%s: the host cannot compute the experts of layer %d, they stay on the device\n", __func__, il);
                    return;
                }
                c->n_embd = n_embd;
                c->n_ff   = n_ff;
                n_expert  = std::max(n_expert, l.experts[0]->ne[2]);
                cl.same_q = cl.quant[0] == cl.quant[1] && cl.q_size[0] == cl.q_size[1];
                // blocks of about 32 KiB of weights and of whole cache lines of outputs
                const size_t gu_row = cl.w[0]->nb[1] + cl.w[1]->nb[1];
                cl.rows[0] = std::min<int64_t>(n_ff,   std::max<int64_t>(16, (int64_t) (32*1024/gu_row)          / 16*16));
                cl.rows[1] = std::min<int64_t>(n_embd, std::max<int64_t>(16, (int64_t) (32*1024/cl.w[2]->nb[1]) / 16*16));
                c->xq_stride = std::max(c->xq_stride, GGML_PAD(cl.q_size[0] + (cl.same_q ? 0 : cl.q_size[1]), 64));
                c->aq_stride = std::max(c->aq_stride, GGML_PAD(cl.q_size[2], 64));
            }
        }
        if (c->n_embd == 0) {
            return;
        }

        ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
        const int max_pairs = (int) (max_batch*n_expert_used);
        c->buf.reset(ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(dev), GGML_MOE_CACHE_XCHG_SIZE(max_pairs, c->n_embd)));
        if (!c->buf) {
            LLAMA_LOG_WARN("%s: no pinned memory for the host experts, they stay on the device\n", __func__);
            return;
        }
        ggml_backend_buffer_clear(c->buf.get(), 0);
        c->xchg = { ggml_backend_buffer_get_base(c->buf.get()), max_pairs, cpu_max };

        c->tok_pair.resize(max_batch);
        c->pair_tok.resize(max_pairs);
        c->exp_id.resize(max_pairs);
        c->exp_p0.resize(max_pairs + 1);
        c->order.resize(max_pairs);
        c->tok_ready.reset(new std::atomic<int32_t>[max_batch]);
        c->gu_left.reset(new std::atomic<int32_t>[max_pairs]);
        c->act.reset(new std::atomic<int32_t>[max_pairs]);
        c->xq.resize(c->xq_stride*max_batch);
        c->gu.resize(2*c->n_ff*max_pairs);
        c->aq.resize(c->aq_stride*max_pairs);
        c->tok_of.assign(max_batch, -1);
        c->exp_of.assign(n_expert, -1);
        c->cursor.resize(max_pairs);

        // the device waits for the host at every layer, so the host computes with every core but the one that drives the device
        c->n_threads = std::max(1, (int) std::thread::hardware_concurrency()/2 - 1);

        d.cpu = std::move(c);
        cpu_share & cs = *d.cpu;
        cs.threads.reserve(cs.n_threads);
        cs.threads.emplace_back([this, id] { cpu_lead(id); });
        for (int i = 1; i < cs.n_threads; ++i) {
            cs.threads.emplace_back([&cs] { cpu_follow(cs); });
        }
        LLAMA_LOG_INFO("%s: %s leaves up to %d missing experts of a layer to the host, computed by %d threads\n", __func__,
            ggml_backend_dev_name(dev), cpu_max, cs.n_threads);
    }

    static void cpu_pause() {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
    }

    // the threads spin while batches come and sleep between requests
    static void cpu_idle(const cpu_share & c, int32_t & spins) {
        if (++spins < 1024) {
            cpu_pause();
            return;
        }
        spins = 0;
        if (ggml_time_us() - c.t_active_us.load(std::memory_order_relaxed) > 200*1000) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    // waits for the device to post the experts it leaves to the host, publishes their job, works on it and marks it done
    void cpu_lead(size_t id) {
        device & d = devices[id];
        cpu_share & c = *d.cpu;
        int32_t * ctl   = (int32_t *) c.xchg.base; // posted, done, pair count, layer
        int32_t   last  = __atomic_load_n(&ctl[1], __ATOMIC_ACQUIRE);
        int32_t   spins = 0;
        while (!c.stop.load(std::memory_order_relaxed)) {
            const int32_t seq = __atomic_load_n(&ctl[0], __ATOMIC_ACQUIRE);
            if (seq == last) {
                cpu_idle(c, spins);
                continue;
            }
            last = seq;
            const int64_t t0 = ggml_time_us();
            cpu_publish(c, __atomic_load_n(&ctl[3], __ATOMIC_RELAXED), __atomic_load_n(&ctl[2], __ATOMIC_RELAXED));
            cpu_work(c);
            while (c.n_done.load(std::memory_order_acquire) != c.n_threads - 1) {
                cpu_pause();
            }
            __atomic_store_n(&ctl[1], seq, __ATOMIC_RELEASE);
            const int64_t t1 = ggml_time_us();
            c.t_active_us.store(t1, std::memory_order_relaxed);
            d.cpu_posts++;
            d.cpu_us += t1 - t0;
        }
    }

    // groups the posted pairs of layer il by token and by expert, and publishes their job
    void cpu_publish(cpu_share & c, int32_t il, int32_t n_pairs) {
        const int32_t * expert = (const int32_t *) c.xchg.base + 16;
        const int32_t * pos    = expert + c.xchg.max_pairs;

        int32_t n_tokens  = 0;
        int32_t n_experts = 0;
        for (int32_t p = 0; p < n_pairs; ++p) {
            const int32_t t = pos[p] / (int32_t) n_expert_used;
            if (c.tok_of[t] < 0) {
                c.tok_of[t] = n_tokens;
                c.tok_pair[n_tokens++] = p;
            }
            c.pair_tok[p] = c.tok_of[t];
            const int32_t e = expert[p];
            if (c.exp_of[e] < 0) {
                c.exp_of[e] = n_experts;
                c.exp_id[n_experts] = e;
                c.exp_p0[++n_experts] = 0;
            }
            c.exp_p0[c.exp_of[e] + 1]++;
        }
        c.exp_p0[0] = 0;
        for (int32_t i = 0; i < n_experts; ++i) {
            c.exp_p0[i + 1] += c.exp_p0[i];
            c.cursor[i] = c.exp_p0[i];
        }
        for (int32_t p = 0; p < n_pairs; ++p) {
            c.order[c.cursor[c.exp_of[expert[p]]]++] = p;
        }
        for (int32_t i = 0; i < n_tokens; ++i) {
            c.tok_of[pos[c.tok_pair[i]] / (int32_t) n_expert_used] = -1;
            c.tok_ready[i].store(0, std::memory_order_relaxed);
        }

        const cpu_share::cpu_layer & l = c.layers[il];
        c.il        = il;
        c.n_tokens  = n_tokens;
        c.n_experts = n_experts;
        c.items_gu  = (int32_t) ((c.n_ff   + l.rows[0] - 1)/l.rows[0]);
        c.items_d   = (int32_t) ((c.n_embd + l.rows[1] - 1)/l.rows[1]);
        c.n_items   = n_tokens + n_experts*(c.items_gu + c.items_d);
        for (int32_t i = 0; i < n_experts; ++i) {
            c.exp_of[c.exp_id[i]] = -1;
            c.gu_left[i].store(c.items_gu, std::memory_order_relaxed);
            c.act[i].store(0, std::memory_order_relaxed);
        }
        c.next.store(0, std::memory_order_relaxed);
        c.n_done.store(0, std::memory_order_relaxed);
        c.job.fetch_add(1, std::memory_order_release);
    }

    static void cpu_follow(cpu_share & c) {
        uint32_t seen  = 0; // a thread that starts late still takes part in the job already published
        int32_t  spins = 0;
        for (;;) {
            const uint32_t job = c.job.load(std::memory_order_acquire);
            if (job == seen) {
                if (c.stop.load(std::memory_order_relaxed)) {
                    return;
                }
                cpu_idle(c, spins);
                continue;
            }
            seen = job;
            cpu_work(c);
            c.n_done.fetch_add(1, std::memory_order_release);
        }
    }

    // takes the work items of the published job until none is left
    static void cpu_work(cpu_share & c) {
        const cpu_share::cpu_layer & l = c.layers[c.il];
        const int64_t n_embd = c.n_embd;
        const int64_t n_ff   = c.n_ff;
        const float * x = (const float *) ((const int32_t *) c.xchg.base + 16 + 2*(size_t) c.xchg.max_pairs);
        float       * y = (float *) x + (size_t) c.xchg.max_pairs*n_embd;
        const int32_t n_q  = c.n_tokens;
        const int32_t n_gu = c.n_experts*c.items_gu;
        for (;;) {
            const int32_t i = c.next.fetch_add(1, std::memory_order_relaxed);
            if (i >= c.n_items) {
                return;
            }
            if (i < n_q) {
                const float * xt = x + (size_t) c.tok_pair[i]*n_embd;
                uint8_t     * q  = c.xq.data() + (size_t) i*c.xq_stride;
                l.quant[0](xt, q, n_embd);
                if (!l.same_q) {
                    l.quant[1](xt, q + l.q_size[0], n_embd);
                }
                c.tok_ready[i].store(1, std::memory_order_release);
                continue;
            }
            if (i < n_q + n_gu) {
                const int32_t ie = (i - n_q) / c.items_gu;
                const int64_t r0 = (int64_t) ((i - n_q) % c.items_gu)*l.rows[0];
                const int64_t r1 = std::min(r0 + l.rows[0], n_ff);
                const int32_t k0 = c.exp_p0[ie];
                const int32_t k1 = c.exp_p0[ie + 1];
                for (int32_t k = k0; k < k1; ++k) {
                    while (c.tok_ready[c.pair_tok[c.order[k]]].load(std::memory_order_acquire) == 0) {
                        cpu_pause();
                    }
                }
                const char * wg = (const char *) l.w[0]->data + (size_t) c.exp_id[ie]*l.w[0]->nb[2];
                const char * wu = (const char *) l.w[1]->data + (size_t) c.exp_id[ie]*l.w[1]->nb[2];
                const size_t qu = l.same_q ? 0 : l.q_size[0];
                for (int64_t r = r0; r < r1; ++r) {
                    for (int32_t k = k0; k < k1; ++k) {
                        const int32_t   p = c.order[k];
                        const uint8_t * q = c.xq.data() + (size_t) c.pair_tok[p]*c.xq_stride;
                        float         * o = c.gu.data() + (size_t) p*2*n_ff;
                        l.dot[0]((int) n_embd, o + r,        0, wg + r*l.w[0]->nb[1], 0, q,      0, 1);
                        l.dot[1]((int) n_embd, o + n_ff + r, 0, wu + r*l.w[1]->nb[1], 0, q + qu, 0, 1);
                    }
                }
                c.gu_left[ie].fetch_sub(1, std::memory_order_release);
                continue;
            }
            const int32_t j  = i - n_q - n_gu;
            const int32_t ie = j / c.items_d;
            const int64_t r0 = (int64_t) (j % c.items_d)*l.rows[1];
            const int64_t r1 = std::min(r0 + l.rows[1], n_embd);
            const int32_t k0 = c.exp_p0[ie];
            const int32_t k1 = c.exp_p0[ie + 1];
            if (c.act[ie].load(std::memory_order_acquire) != 2) {
                while (c.gu_left[ie].load(std::memory_order_acquire) != 0) {
                    cpu_pause();
                }
                int32_t idle = 0;
                if (c.act[ie].compare_exchange_strong(idle, 1, std::memory_order_acq_rel)) {
                    // SwiGLU into the gate outputs, quantized as the input of down
                    for (int32_t k = k0; k < k1; ++k) {
                        const int32_t p = c.order[k];
                        float       * g = c.gu.data() + (size_t) p*2*n_ff;
                        const float * u = g + n_ff;
                        for (int64_t f = 0; f < n_ff; ++f) {
                            g[f] = g[f]/(1.0f + expf(-g[f]))*u[f];
                        }
                        l.quant[2](g, c.aq.data() + (size_t) p*c.aq_stride, n_ff);
                    }
                    c.act[ie].store(2, std::memory_order_release);
                } else {
                    while (c.act[ie].load(std::memory_order_acquire) != 2) {
                        cpu_pause();
                    }
                }
            }
            const char * wd = (const char *) l.w[2]->data + (size_t) c.exp_id[ie]*l.w[2]->nb[2];
            for (int64_t r = r0; r < r1; ++r) {
                for (int32_t k = k0; k < k1; ++k) {
                    const int32_t p = c.order[k];
                    l.dot[2]((int) n_ff, y + (size_t) p*n_embd + r, 0, wd + r*l.w[2]->nb[1], 0, c.aq.data() + (size_t) p*c.aq_stride, 0, 1);
                }
            }
        }
    }

    // replay the hits and fills the device logged into the SLRU, once the batches that wrote them are done
    void apply_log(size_t id) {
        device & d = devices[id];
        if (!d.fill) {
            return;
        }
        ggml_backend_synchronize(d.backend);

        // the header and the records of a few batches in one read, the rest only after long batches
        const size_t n_first = std::min<size_t>(ggml_nelements(d.dlog), 4 + 3*4096);
        log_records.resize(n_first);
        ggml_backend_tensor_get(d.dlog, log_records.data(), 0, n_first*sizeof(int32_t));
        const int32_t n = log_records[0];
        // a lost fill would leave a slot whose expert the SLRU does not know
        GGML_ASSERT(n <= log_records[1] && "MoE cache log overflow");
        d.log_seq = log_records[2];
        if (4 + 3*(size_t) n > n_first) {
            log_records.resize(4 + 3*(size_t) n);
            ggml_backend_tensor_get(d.dlog, log_records.data() + n_first, n_first*sizeof(int32_t), (log_records.size() - n_first)*sizeof(int32_t));
        }
        // the log is emptied by the next push, a replay before it skips the records already replayed
        for (int32_t k = d.log_done; k < n; ++k) {
            const int32_t il = log_records[4 + 3*k + 0];
            const int32_t s  = log_records[4 + 3*k + 1];
            const int32_t x  = log_records[4 + 3*k + 2];
            group & g = groups[layers[il].ig];
            if (s == -2) {
                stats_skip.misses++;
                stats_skip.bytes += g.slot_bytes;
            } else if (s < 0) {
                stats_cpu.misses++;
                stats_cpu.bytes += g.slot_bytes;
            } else if (x >= 0) {
                g.slru.filled(il, x, s);
                stats_fill.misses++;
                stats_fill.bytes += g.slot_bytes;
            } else {
                if (g.slru.key_of[s] == il*g.slru.n_expert + (-1 - x)) {
                    g.slru.touch(s);
                }
                stats_fill.hits++;
            }
        }
        d.log_done = n;
    }

    // hand the device what the SLRU holds after the replay: the slot maps, and the slot owners and eviction order of each group,
    // and an empty log; one copy from pinned memory, ordered before the next batch
    void push_state(size_t id) {
        device & d = devices[id];
        if (!d.fill) {
            return;
        }
        // the last push may still read the image
        ggml_backend_synchronize(d.backend);

        const int64_t n_expert_d = d.dmap->ne[0];
        std::fill(d.img, d.img + ggml_nelements(d.dmap), -1);
        for (const group & g : groups) {
            if (g.id != (int32_t) id || g.n_slots == 0) {
                continue;
            }
            for (int32_t il : g.layers) {
                std::copy_n(g.slru.slot_map[il], g.slru.n_expert, d.img + il*n_expert_d);
            }
            const moe_cache_slru & r = g.slru;
            r.eviction_order(victims);
            int32_t * state = d.img + g.state_off;
            std::fill(state, state + 8, 0);
            state[1] = (int32_t) victims.size();
            for (int32_t s = 0; s < g.n_slots; ++s) {
                const int32_t key = r.key_of[s];
                state[8 + s] = key < 0 ? -1 : (key / r.n_expert)*(int32_t) n_expert_d + key % r.n_expert;
            }
            std::copy(victims.begin(), victims.end(), state + 8 + g.n_slots);
        }
        int32_t * log = d.img + d.log_off;
        log[0] = 0;
        log[1] = (int32_t) ((ggml_nelements(d.dlog) - 4)/3);
        log[2] = d.log_seq;
        log[3] = 0;
        ggml_backend_tensor_set_async(d.backend, d.dfill, d.img, 0, (d.log_off + 4)*sizeof(int32_t));
        d.log_done = 0;
    }

    // the slots of the experts ids [n_expert_used, n_tokens] of layer il, looked up and filled on the device; with cpu, missing
    // experts can be left to the host, which computes them from the input x [n_embd, n_tokens]; with the tail skip or the host,
    // build_join is required
    ggml_tensor * build_fill(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * x, int32_t il, bool cpu) const {
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        const device & d = devices[g.id];
        if (!d.fill || g.dstate == nullptr) {
            return nullptr;
        }
        ggml_tensor * banks[3];
        const void  * host[3];
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            banks[ip] = g.banks[ip];
            host[ip]  = l.experts[ip]->data;
        }
        const bool share = cpu && l.cpu && d.cpu && ggml_is_contiguous(x);
        const int32_t skip_rank = std::min<int32_t>(tail_skip, (int32_t) ids->ne[0]);
        return ggml_moe_cache_fill(ctx, ids, share ? x : nullptr, d.dmap, g.dstate, d.dlog, banks, host, (int) l.experts.size(), il,
            share ? &d.cpu->xchg : nullptr, skip_rank);
    }

    // the outputs experts [n_embd, n_expert_used, n_tokens] of layer il with those the lookup fill did not leave to the device:
    // computed by the host or skipped
    ggml_tensor * build_join(ggml_context * ctx, ggml_tensor * experts, ggml_tensor * fill) const {
        // op parameters 11 and 12 of the lookup: the missing experts it may leave to the host, the rank of the skipped tail
        const int32_t * params = fill != nullptr && fill->op == GGML_OP_MOE_CACHE_FILL ? (const int32_t *) fill->op_params : nullptr;
        if (params == nullptr || (params[11] == 0 && params[12] == 0)) {
            return experts;
        }
        return ggml_moe_cache_join(ctx, experts, fill);
    }

    // device memory of an expert tensor in a ring slot, with one more expert: CUDA MMQ can read past the last one
    static size_t ring_tensor_bytes(const ggml_tensor * t) {
        return GGML_PAD(t->nb[2]*(t->ne[2] + 1), 256);
    }

    // a ring slot holds the largest layer of its device
    size_t ring_slot_bytes(size_t id) const {
        size_t res = 0;
        for (const group & g : groups) {
            if (g.id != (int32_t) id || g.n_slots == 0) {
                continue;
            }
            size_t bytes = 0;
            for (const ggml_tensor * t : g.ref) {
                bytes += ring_tensor_bytes(t);
            }
            res = std::max(res, bytes);
        }
        return res;
    }

    // the ring of device id holds memory only while prompt batches run, taken from the cache
    bool ring_init(size_t id) {
        ring & r = rings[id];
        if (r.buf) {
            return true;
        }
        if (r.failed) {
            return false;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(devices[id].backend);
        if (r.copy == nullptr) {
            r.copy      = ggml_backend_dev_init(dev, nullptr);
            r.filled[0] = ggml_backend_event_new(dev);
            r.filled[1] = ggml_backend_event_new(dev);
            r.released  = ggml_backend_event_new(dev);
            if (r.copy == nullptr || r.filled[0] == nullptr || r.filled[1] == nullptr || r.released == nullptr) {
                LLAMA_LOG_WARN("%s: %s has no second stream with events, prompt batches copy their experts in turn\n", __func__, ggml_backend_dev_name(dev));
                ring_free(r);
                r.failed = true;
                return false;
            }
        }
        r.slot_bytes = ring_slot_bytes(id);
        make_room(dev, 2*r.slot_bytes);
        r.buf.reset(ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), 2*r.slot_bytes));
        if (!r.buf) {
            LLAMA_LOG_WARN("%s: no memory for the %.2f MiB ring of %s, prompt batches copy their experts in turn\n", __func__,
                2*r.slot_bytes/1024.0/1024.0, ggml_backend_dev_name(dev));
            r.failed = true;
            return false;
        }
        // as compute data the slots are left alone by tensor initialization, and zeroed for the expert past the last
        ggml_backend_buffer_set_usage(r.buf.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_buffer_clear(r.buf.get(), 0);
        r.layer[0] = r.layer[1] = -1;
        r.cur = r.bound = -1;
        return true;
    }

    // freeing the slots waits for the graphs that read them
    void ring_release(size_t id) {
        ring & r = rings[id];
        if (r.buf) {
            ggml_backend_synchronize(devices[id].backend);
            ggml_backend_synchronize(r.copy);
            r.buf.reset();
        }
        r.layer[0] = r.layer[1] = -1;
        r.cur = r.bound = -1;
        r.failed = false;
    }

    static void ring_free(ring & r) {
        r.buf.reset();
        for (ggml_backend_event_t * ev : { &r.filled[0], &r.filled[1], &r.released }) {
            if (*ev != nullptr) {
                ggml_backend_event_free(*ev);
                *ev = nullptr;
            }
        }
        if (r.copy != nullptr) {
            ggml_backend_free(r.copy);
            r.copy = nullptr;
        }
    }

    // the next layer after il cached on device id, -1 if there is none
    int32_t ring_next_layer(int32_t il, size_t id) const {
        for (int32_t j = il + 1; j < (int32_t) layers.size(); ++j) {
            if (layers[j].ig >= 0 && groups[layers[j].ig].id == (int32_t) id) {
                return j;
            }
        }
        return -1;
    }

    // fill slot s with the experts of layer il on the ring stream: the cached ones from the banks, the others from host memory
    // the uploads from host memory go first: a copy between device buffers runs on the SMs, so it waits for the kernels of the
    // compute stream, and on one stream the uploads queued behind it would wait too
    void ring_fill(ring & r, int32_t s, int32_t il) {
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        const int32_t * slots = g.slru.slot_map[il];
        char * base = (char *) ggml_backend_buffer_get_base(r.buf.get()) + s*r.slot_bytes;
        for (const bool from_cache : { false, true }) {
            size_t off = 0;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const int64_t n_expert = w->ne[2];
                const size_t  nb2      = w->nb[2];
                for (int64_t e = 0; e < n_expert; ) {
                    const bool cached = slots != nullptr && slots[e] >= 0;
                    int64_t n = 1;
                    while (e + n < n_expert && (cached ? slots[e + n] == slots[e] + n : (slots == nullptr || slots[e + n] < 0))) {
                        n++;
                    }
                    if (cached == from_cache) {
                        ggml_reset(ctx_views.get());
                        ggml_tensor * dst = ggml_new_tensor_3d(ctx_views.get(), w->type, w->ne[0], w->ne[1], n);
                        ggml_backend_tensor_alloc(r.buf.get(), dst, base + off + e*nb2);
                        if (cached) {
                            ggml_tensor * bank = g.banks[ip];
                            ggml_tensor * src  = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
                            ggml_backend_view_init(src);
                            ggml_backend_tensor_copy_async(r.copy, r.copy, src, dst);
                            stats_copy.hits  += n;
                            stats_copy.bytes += n*nb2;
                        } else {
                            ggml_backend_tensor_set_async(r.copy, dst, (const char *) w->data + e*nb2, 0, n*nb2);
                            stats_ring.misses += n;
                            stats_ring.bytes  += n*nb2;
                        }
                    }
                    e += n;
                }
                off += ring_tensor_bytes(w);
            }
        }
        ggml_backend_event_record(r.filled[s], r.copy);
        r.layer[s] = il;
    }

    bool ring_bind(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t n_tokens) {
        if (n_tokens < ring_min) {
            return false;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return false;
        }
        const int32_t il = it->second.il;
        const layer & l  = layers[il];
        const size_t  id = groups[l.ig].id;
        if (backend != devices[id].backend || !ring_init(id)) {
            return false;
        }
        ring & r = rings[id];

        // the first expert tensor of the layer: the compute stream waits for its slot, filled now unless it was prefetched,
        // then the next layer is prefetched into the other slot once the ops that read that slot are done
        if (r.bound != il) {
            int32_t s = r.layer[0] == il ? 0 : r.layer[1] == il ? 1 : -1;
            if (s < 0) {
                s = r.cur < 0 ? 0 : 1 - r.cur;
                ggml_backend_event_record(r.released, backend);
                ggml_backend_event_wait(r.copy, r.released);
                ring_fill(r, s, il);
            }
            ggml_backend_event_wait(backend, r.filled[s]);
            r.cur   = s;
            r.bound = il;

            const int32_t next = ring_next_layer(il, id);
            if (next >= 0 && r.layer[1 - s] != next) {
                ggml_backend_event_record(r.released, backend);
                ggml_backend_event_wait(r.copy, r.released);
                ring_fill(r, 1 - s, next);
            }
        }

        // the ops read w from the ring: the copy of w the scheduler allocated stays unused
        size_t off = 0;
        for (int32_t ip = 0; ip < it->second.ip; ++ip) {
            off += ring_tensor_bytes(l.experts[ip]);
        }
        dst->buffer = r.buf.get();
        dst->data   = (char *) ggml_backend_buffer_get_base(r.buf.get()) + r.cur*r.slot_bytes + off;
        return true;
    }

    ~impl() {
        // the host threads read the layers and the state, stop them first
        for (device & d : devices) {
            d.cpu.reset();
        }
        log_stats();
        for (ring & r : rings) {
            ring_free(r);
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].slru.n_active) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        size_t n_hit = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        log("small batches on the device", stats_fill);
        if (stats_cpu.misses > 0) {
            int64_t posts = 0;
            int64_t us    = 0;
            for (const device & d : devices) {
                posts += d.cpu_posts;
                us    += d.cpu_us;
            }
            LLAMA_LOG_INFO("llama_moe_cache: small batches: %zu missing experts computed by the host, %.2f MiB read, %.1f us and %.2f experts per layer, %.1f GB/s\n",
                stats_cpu.misses, stats_cpu.bytes/1024.0/1024.0, posts > 0 ? (double) us/posts : 0.0, posts > 0 ? (double) stats_cpu.misses/posts : 0.0,
                us > 0 ? stats_cpu.bytes/1e3/us : 0.0);
        }
        if (stats_skip.misses > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: small batches: %zu missing experts skipped at the tail of the routes, %.2f MiB not read\n",
                stats_skip.misses, stats_skip.bytes/1024.0/1024.0);
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (stats_ring.misses > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: prompt batches: %zu experts uploaded to the ring, %.2f MiB\n", stats_ring.misses, stats_ring.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts,
        size_t size, size_t free, int32_t cpu_experts, int32_t tail_skip) :
    pimpl(new impl(model, backends, bufts, size, free, cpu_experts, tail_skip)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

ggml_tensor * llama_moe_cache::build_fill(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * x, int32_t il, bool cpu) const {
    return pimpl->build_fill(ctx, ids, x, il, cpu);
}

ggml_tensor * llama_moe_cache::build_join(ggml_context * ctx, ggml_tensor * experts, ggml_tensor * fill) const {
    return pimpl->build_join(ctx, experts, fill);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

bool llama_moe_cache::ring_bind(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t n_tokens) {
    return pimpl->ring_bind(backend, w, dst, n_tokens);
}

void llama_moe_cache::adapt(bool grow, int64_t n_tokens) {
    pimpl->adapt(grow, n_tokens);
}

void llama_moe_cache::make_room(ggml_backend_dev_t dev, size_t size) {
    pimpl->make_room(dev, size);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (size_t id = 0; id < pimpl->devices.size(); ++id) {
        const auto & d = pimpl->devices[id];
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.elastic ? pimpl->committed(id) : d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
