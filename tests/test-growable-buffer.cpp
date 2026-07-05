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

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);

    printf(g_failures ? "test-growable-buffer: FAILED (%d)\n" : "test-growable-buffer: OK\n", g_failures);
    return g_failures ? 1 : 0;
}
