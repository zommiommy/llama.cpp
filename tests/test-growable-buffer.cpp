// V1: unit test for the CUDA growable (demand-paged) buffer type.
//
// Exercises ggml_backend_cuda_growable_buffer_type via the generic
// ggml_backend_dev_buffer_type_lazy() hook. Two parts:
//   A) direct 1 GiB reservation           -> VA reserved, zero physical committed.
//   B) real allocator (alloc_ctx_tensors) -> commit-on-write, interior release, re-commit;
//      readback byte-identical throughout.
// Assertions use the deterministic per-buffer stats counters (ggml_backend_cuda_buffer_stats),
// never nvidia-smi. Skips gracefully with no GPU; verifies the eager fallback with no VMM.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static int g_failures = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            g_failures++;                                                             \
        }                                                                             \
    } while (0)

int main() {
    ggml_backend_load_all(); // ensure all backends (incl. CUDA) are registered before enumeration

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev = d;
            break;
        }
    }

    if (dev == nullptr) {
        printf("test-growable-buffer: no GPU device found; skipping\n");
        return 0;
    }
    printf("test-growable-buffer: device = %s\n", ggml_backend_dev_name(dev));

    ggml_backend_buffer_type_t lazy = ggml_backend_dev_buffer_type_lazy(dev);

    if (lazy == nullptr) {
        // no VMM: the lazy buffer type is unavailable and callers fall back to eager allocation.
        // verify the stats accessor reports a normal buffer as fully committed with no lazy ops.
        printf("test-growable-buffer: lazy buffer type unavailable (no VMM) -> testing eager fallback\n");
        ggml_backend_buffer_type_t normal = ggml_backend_dev_buffer_type(dev);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(normal, 4ull * 1024 * 1024);
        CHECK(buf != nullptr);
        if (buf) {
            size_t reserved = 0, committed = 0, commit_ops = 123, release_ops = 123;
            ggml_backend_cuda_buffer_stats(buf, &reserved, &committed, &commit_ops, &release_ops);
            CHECK(commit_ops == 0);
            CHECK(release_ops == 0);
            ggml_backend_buffer_free(buf);
        }
        printf(g_failures ? "test-growable-buffer: FAILED (%d)\n" : "test-growable-buffer: OK (fallback)\n", g_failures);
        return g_failures ? 1 : 0;
    }

    // ---- Part A: a large reservation commits nothing up front ----
    {
        const size_t GiB = 1ull << 30;
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(lazy, GiB);
        CHECK(buf != nullptr);
        if (buf) {
            size_t reserved = 0, committed = 1, commit_ops = 1, release_ops = 1;
            ggml_backend_cuda_buffer_stats(buf, &reserved, &committed, &commit_ops, &release_ops);
            printf("  [A] reserve 1 GiB: reserved=%zu MiB committed=%zu commit_ops=%zu\n",
                   reserved / 1024 / 1024, committed, commit_ops);
            CHECK(reserved >= GiB);   // rounded up to granularity
            CHECK(committed == 0);    // nothing physically backed yet
            CHECK(commit_ops == 0);
            ggml_backend_buffer_free(buf);
        }
    }

    // ---- Part B: real allocator path (as used by the KV cache) ----
    const size_t span_bytes = 32ull * 1024 * 1024; // 32 MiB -> spans many granules (>= 3 for any sane granularity)
    const int64_t n         = (int64_t) (span_bytes / sizeof(float));

    struct ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, /*.no_alloc =*/ true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    // allocate the tensor from the lazy buft via the real allocator (reserves VA, no physical commit)
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, lazy);
    CHECK(buf != nullptr);
    if (buf == nullptr) {
        ggml_free(ctx);
        printf("test-growable-buffer: FAILED (alloc_ctx)\n");
        return 1;
    }

    {
        size_t committed = 1, commit_ops = 1;
        ggml_backend_cuda_buffer_stats(buf, nullptr, &committed, &commit_ops, nullptr);
        printf("  [B] after alloc_ctx: committed=%zu commit_ops=%zu (expect 0)\n", committed, commit_ops);
        CHECK(committed == 0);   // allocation reserves VA only; init_tensor is a no-op
        CHECK(commit_ops == 0);
    }

    std::vector<float> src(n), dst(n);
    for (int64_t i = 0; i < n; i++) {
        src[i] = (float) ((uint64_t) i * 2654435761ull % 100003ull) * 0.5f;
    }

    // write spanning many granules -> commits grow, readback identical
    ggml_backend_tensor_set(t, src.data(), 0, span_bytes);
    size_t committed_w = 0, commit_ops_w = 0;
    ggml_backend_cuda_buffer_stats(buf, nullptr, &committed_w, &commit_ops_w, nullptr);
    printf("  [B] after write:   committed=%zu MiB commit_ops=%zu\n", committed_w / 1024 / 1024, commit_ops_w);
    CHECK(committed_w >= span_bytes);
    CHECK(commit_ops_w >= 3);

    memset(dst.data(), 0, span_bytes);
    ggml_backend_tensor_get(t, dst.data(), 0, span_bytes);
    CHECK(memcmp(src.data(), dst.data(), span_bytes) == 0);

    // release an interior span -> committed shrinks, release_ops > 0
    ggml_backend_buffer_release_range(buf, 2ull * 1024 * 1024, 28ull * 1024 * 1024);
    size_t committed_r = 0, release_ops_r = 0;
    ggml_backend_cuda_buffer_stats(buf, nullptr, &committed_r, nullptr, &release_ops_r);
    printf("  [B] after release: committed=%zu MiB release_ops=%zu\n", committed_r / 1024 / 1024, release_ops_r);
    CHECK(committed_r < committed_w);
    CHECK(release_ops_r > 0);

    // re-write the released span -> re-commits, readback still identical
    ggml_backend_tensor_set(t, src.data(), 0, span_bytes);
    memset(dst.data(), 0, span_bytes);
    ggml_backend_tensor_get(t, dst.data(), 0, span_bytes);
    CHECK(memcmp(src.data(), dst.data(), span_bytes) == 0);

    const size_t gran = ggml_backend_dev_lazy_granule(dev);
    printf("  [C] lazy granule:  %zu MiB\n", gran / 1024 / 1024);
    CHECK(gran > 0);
    CHECK(6 * gran <= span_bytes);
    if (gran > 0 && 6 * gran <= span_bytes) {

        // ---- Part C: evict/restore round-trip + fault-in ----
        ggml_backend_tensor_set(t, src.data(), 0, span_bytes);
        size_t c0 = 0;
        ggml_backend_cuda_buffer_stats(buf, nullptr, &c0, nullptr, nullptr);

        size_t freed = ggml_backend_buffer_evict_range(buf, 2 * gran, 4 * gran, gran, SIZE_MAX);
        size_t c1 = 0, hb1 = 0, eo1 = 0, ro1 = 0;
        ggml_backend_cuda_buffer_stats(buf, nullptr, &c1, nullptr, nullptr);
        ggml_backend_cuda_buffer_evict_stats(buf, &hb1, &eo1, &ro1);
        printf("  [C] after evict:   freed=%zu MiB committed=%zu MiB host=%zu MiB evict_ops=%zu restore_ops=%zu\n",
               freed / 1024 / 1024, c1 / 1024 / 1024, hb1 / 1024 / 1024, eo1, ro1);
        CHECK(freed >= 4 * gran);
        CHECK(c1 < c0);
        CHECK(c0 - c1 == freed);
        CHECK(hb1 == freed);
        CHECK(eo1 >= 1);

        bool ok = ggml_backend_buffer_restore_range(buf, 2 * gran, 4 * gran);
        size_t c2 = 0, hb2 = 1, ro2 = 0;
        ggml_backend_cuda_buffer_stats(buf, nullptr, &c2, nullptr, nullptr);
        ggml_backend_cuda_buffer_evict_stats(buf, &hb2, nullptr, &ro2);
        printf("  [C] after restore: committed=%zu MiB host=%zu MiB restore_ops=%zu\n",
               c2 / 1024 / 1024, hb2 / 1024 / 1024, ro2);
        CHECK(ok);
        CHECK(c2 == c0);
        CHECK(hb2 == 0);
        CHECK(ro2 >= 1);

        memset(dst.data(), 0, span_bytes);
        ggml_backend_tensor_get(t, dst.data(), 0, span_bytes);
        CHECK(memcmp(src.data(), dst.data(), span_bytes) == 0);

        freed = ggml_backend_buffer_evict_range(buf, 2 * gran, 4 * gran, gran, SIZE_MAX);
        CHECK(freed >= 4 * gran);
        memset(dst.data(), 0, span_bytes);
        ggml_backend_tensor_get(t, dst.data(), 2 * gran, 4 * gran);
        CHECK(memcmp((const char *) src.data() + 2 * gran, dst.data(), 4 * gran) == 0);
        size_t hb_fault = 1;
        ggml_backend_cuda_buffer_evict_stats(buf, &hb_fault, nullptr, nullptr);
        CHECK(hb_fault == 0);

        // ---- Part D: release/discard stale-shadow guard ----
        ggml_backend_tensor_set(t, src.data(), 0, span_bytes);
        freed = ggml_backend_buffer_evict_range(buf, 2 * gran, 4 * gran, gran, SIZE_MAX);
        size_t c_after_evict = 0;
        ggml_backend_cuda_buffer_stats(buf, nullptr, &c_after_evict, nullptr, nullptr);
        size_t hb_before_release = 0;
        ggml_backend_cuda_buffer_evict_stats(buf, &hb_before_release, nullptr, nullptr);
        CHECK(freed >= 4 * gran);
        CHECK(hb_before_release > 0);

        ggml_backend_buffer_release_range(buf, 2 * gran, 4 * gran);
        size_t c_after_release = 0, hb_after_release = 1;
        ggml_backend_cuda_buffer_stats(buf, nullptr, &c_after_release, nullptr, nullptr);
        ggml_backend_cuda_buffer_evict_stats(buf, &hb_after_release, nullptr, nullptr);
        printf("  [D] after release: committed=%zu MiB host=%zu MiB\n",
               c_after_release / 1024 / 1024, hb_after_release / 1024 / 1024);
        CHECK(hb_after_release == 0);
        CHECK(c_after_release == c_after_evict);

        memset(dst.data(), 0xAB, span_bytes);
        ggml_backend_tensor_get(t, dst.data(), 2 * gran, 4 * gran);
        const unsigned char * fresh = (const unsigned char *) dst.data();
        bool all_zero = true;
        for (size_t i = 0; i < 4 * gran; i++) {
            if (fresh[i] != 0) {
                all_zero = false;
                break;
            }
        }
        CHECK(all_zero);
        CHECK(memcmp((const char *) src.data() + 2 * gran, dst.data(), 4 * gran) != 0);

        // ---- Part E: chunk grouping over the full span ----
        ggml_backend_tensor_set(t, src.data(), 0, span_bytes);
        size_t eoA = 0;
        ggml_backend_cuda_buffer_evict_stats(buf, nullptr, &eoA, nullptr);
        freed = ggml_backend_buffer_evict_range(buf, 0, span_bytes, 4 * gran, SIZE_MAX);
        size_t eoB = 0;
        ggml_backend_cuda_buffer_evict_stats(buf, nullptr, &eoB, nullptr);
        const size_t full_granules = span_bytes / gran;
        const size_t expected_chunks = (full_granules + 3) / 4;
        const size_t actual_chunks = eoB - eoA;
        printf("  [E] chunk evict:   freed=%zu MiB chunks=%zu expected=%zu\n",
               freed / 1024 / 1024, actual_chunks, expected_chunks);
        CHECK(actual_chunks + 1 >= expected_chunks);
        CHECK(actual_chunks <= expected_chunks + 1);
        ok = ggml_backend_buffer_restore_range(buf, 0, span_bytes);
        CHECK(ok);
        memset(dst.data(), 0, span_bytes);
        ggml_backend_tensor_get(t, dst.data(), 0, span_bytes);
        CHECK(memcmp(src.data(), dst.data(), span_bytes) == 0);

        // ---- Part F: read-only page sharing (system-prompt prefix reuse) ----
        CHECK(10 * gran <= span_bytes);
        if (10 * gran <= span_bytes) {
            ggml_backend_buffer_release_range(buf, 8 * gran, 2 * gran);
            ggml_backend_tensor_set(t, src.data(), 4 * gran, 2 * gran);

            size_t c0_share = 0;
            ggml_backend_cuda_buffer_stats(buf, nullptr, &c0_share, nullptr, nullptr);

            const size_t shared = ggml_backend_buffer_share_range(buf, 8 * gran, 4 * gran, 2 * gran);
            printf("  [F] shared=%zu MiB\n", shared / 1024 / 1024);
            CHECK(shared == 2 * gran);

            size_t c1_share = 0;
            ggml_backend_cuda_buffer_stats(buf, nullptr, &c1_share, nullptr, nullptr);
            CHECK(c1_share == c0_share);

            memset(dst.data(), 0, 2 * gran);
            ggml_backend_tensor_get(t, dst.data(), 8 * gran, 2 * gran);
            CHECK(memcmp(src.data(), dst.data(), 2 * gran) == 0);

            std::vector<float> alt(2 * gran / sizeof(float));
            for (size_t i = 0; i < alt.size(); i++) {
                alt[i] = (float) (i * 7 + 3);
            }
            ggml_backend_tensor_set(t, alt.data(), 4 * gran, 2 * gran);
            memset(dst.data(), 0, 2 * gran);
            ggml_backend_tensor_get(t, dst.data(), 8 * gran, 2 * gran);
            CHECK(memcmp(alt.data(), dst.data(), 2 * gran) == 0);
        }

        // ---- Part G: cross-stream prefix sharing (KV stream prefix reuse) ----
        CHECK(4 * gran <= span_bytes);
        if (4 * gran <= span_bytes) {
            const size_t stride  = 3 * gran;
            const size_t prefix  = gran;
            const size_t s0_base = 0;
            const size_t s1_base = stride;

            std::vector<float> owner(prefix / sizeof(float));
            std::vector<float> borrower(prefix / sizeof(float));
            for (size_t i = 0; i < owner.size(); i++) {
                owner[i] = (float) (1000 + i * 11);
                borrower[i] = (float) (2000 + i * 13);
            }

            ggml_backend_tensor_set(t, owner.data(), s0_base, prefix);
            ggml_backend_tensor_set(t, borrower.data(), s1_base, prefix);
            ggml_backend_buffer_release_range(buf, s1_base, prefix);
            ggml_backend_tensor_set(t, owner.data(), s0_base, prefix);

            size_t c0_cross_share = 0;
            ggml_backend_cuda_buffer_stats(buf, nullptr, &c0_cross_share, nullptr, nullptr);

            const size_t shared = ggml_backend_buffer_share_range(buf, s1_base, s0_base, prefix);
            printf("  [G] cross-stream shared=%zu MiB stride=%zu MiB prefix=%zu MiB\n",
                   shared / 1024 / 1024, stride / 1024 / 1024, prefix / 1024 / 1024);
            CHECK(shared == prefix);

            size_t c1_cross_share = 0;
            ggml_backend_cuda_buffer_stats(buf, nullptr, &c1_cross_share, nullptr, nullptr);
            CHECK(c1_cross_share == c0_cross_share);

            memset(dst.data(), 0, prefix);
            ggml_backend_tensor_get(t, dst.data(), s1_base, prefix);
            CHECK(memcmp(owner.data(), dst.data(), prefix) == 0);

            std::vector<float> updated(prefix / sizeof(float));
            for (size_t i = 0; i < updated.size(); i++) {
                updated[i] = (float) (3000 + i * 17);
            }
            ggml_backend_tensor_set(t, updated.data(), s0_base, prefix);
            memset(dst.data(), 0, prefix);
            ggml_backend_tensor_get(t, dst.data(), s1_base, prefix);
            CHECK(memcmp(updated.data(), dst.data(), prefix) == 0);

            CHECK(ggml_backend_buffer_share_range(buf, s1_base + 1, s0_base, prefix) == 0);
            CHECK(ggml_backend_buffer_share_range(buf, s1_base, s0_base, prefix - 1) == 0);

            size_t c2_cross_share = 0;
            ggml_backend_cuda_buffer_stats(buf, nullptr, &c2_cross_share, nullptr, nullptr);
            printf("  [G] committed stable=%zu MiB\n", c2_cross_share / 1024 / 1024);
            CHECK(c2_cross_share == c0_cross_share);
        }

        // ---- Part H: in-buffer copy into private pages (KV stream fork) ----
        CHECK(8 * gran <= span_bytes);
        if (8 * gran <= span_bytes) {
            const size_t copied_len       = 2 * gran;
            const size_t src_base         = 0;
            const size_t copy_dst_off     = 6 * gran;
            const size_t sub_dst_gran_off = 2 * gran;
            const size_t sub_src_off      = src_base + 7;
            const size_t sub_dst_off      = sub_dst_gran_off + 100;
            const size_t sub_len          = gran / 2 + 123;
            const size_t borrowed_dst_off = 3 * gran; // still borrowed from Part G

            CHECK(copy_dst_off + copied_len <= span_bytes);
            CHECK(sub_src_off + sub_len <= src_base + copied_len);
            CHECK(sub_dst_off + sub_len <= span_bytes);
            CHECK(sub_dst_off / gran == (sub_dst_off + sub_len - 1) / gran);

            // Seed the source bytes, then make both destination regions genuinely free/private.
            // release_range only frees whole granules contained in the release window, so the
            // unaligned sub-copy destination is freed by releasing its enclosing granule.
            ggml_backend_tensor_set(t, src.data(), src_base, copied_len);
            ggml_backend_buffer_release_range(buf, copy_dst_off, copied_len);
            ggml_backend_buffer_release_range(buf, sub_dst_gran_off, gran);

            size_t c0_copy = 0;
            ggml_backend_cuda_buffer_stats(buf, nullptr, &c0_copy, nullptr, nullptr);

            CHECK(ggml_backend_buffer_copy_range(buf, borrowed_dst_off, src_base, gran) == 0);

            const size_t copied = ggml_backend_buffer_copy_range(buf, copy_dst_off, src_base, copied_len);
            CHECK(copied == copied_len);

            memset(dst.data(), 0, copied_len);
            ggml_backend_tensor_get(t, dst.data(), copy_dst_off, copied_len);
            CHECK(memcmp(src.data(), dst.data(), copied_len) == 0);

            const size_t copied_sub = ggml_backend_buffer_copy_range(buf, sub_dst_off, sub_src_off, sub_len);
            CHECK(copied_sub == sub_len);

            memset(dst.data(), 0, sub_len);
            ggml_backend_tensor_get(t, dst.data(), sub_dst_off, sub_len);
            CHECK(memcmp((const char *) src.data() + sub_src_off, dst.data(), sub_len) == 0);

            size_t c1_copy = 0;
            ggml_backend_cuda_buffer_stats(buf, nullptr, &c1_copy, nullptr, nullptr);
            CHECK(c1_copy >= c0_copy);
            const size_t committed_delta          = c1_copy >= c0_copy ? c1_copy - c0_copy : 0;
            const size_t expected_committed_delta = 3 * gran;
            printf("  [H] copy_range copied=%zu MiB sub=%zu bytes committed_delta=%zu MiB\n",
                   copied / 1024 / 1024, copied_sub, committed_delta / 1024 / 1024);

            const size_t logical_h = ggml_backend_buffer_get_size(buf);
            CHECK(ggml_backend_buffer_copy_range(buf, 0, 0, 0) == 0);
            CHECK(logical_h > 0);
            CHECK(ggml_backend_buffer_copy_range(buf, logical_h - 1, 0, 2) == 0);
            CHECK(committed_delta > 0);
            CHECK(committed_delta % gran == 0);
            CHECK(committed_delta == expected_committed_delta);
            CHECK(committed_delta != copied + copied_sub);
        }

    }
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);

    printf(g_failures ? "test-growable-buffer: FAILED (%d)\n" : "test-growable-buffer: OK\n", g_failures);
    return g_failures ? 1 : 0;
}
