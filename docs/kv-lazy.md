# Lazy (demand-paged) KV cache — `--kv-lazy`

`--kv-lazy` makes the CUDA KV cache (and, for hybrid SSM+attention models, the
recurrent state) reserve only **virtual address space** at `--ctx-size` and
commit physical VRAM **2 MiB pages on demand** as sequences actually grow.
With `-fa on` (the standard config) peak VRAM tracks real token usage (± the 2 MiB
allocation granule) instead of being reserved up front. Kernels, ggml graphs, KV layout, FlashAttention and the
`--split-mode layer` pipeline are byte-identical to the eager path — only the
physical backing is lazy.

It targets the multi-agent serving case (`llama-server --parallel N` with large
per-slot contexts) where most slots never fill their context: idle/short slots
cost ~0 VRAM instead of the full `--ctx-size` reservation.

## Mechanism (implementation map)

- **ggml**: a CUDA *growable* buffer type (`ggml_backend_cuda_growable_buffer_type`)
  built on the CUDA VMM driver API (`cuMemAddressReserve` / `cuMemCreate` /
  `cuMemMap` / `cuMemSetAccess`). Generic backend hooks
  `ggml_backend_buffer_ensure_range` / `release_range` and
  `ggml_backend_dev_buffer_type_lazy` dispatch to it; they are no-ops for
  non-growable buffers. Newly committed pages are zeroed, preserving the KV
  zero-init invariant. Telemetry: `ggml_backend_cuda_buffer_stats`.
- **attention KV** (`llama_kv_cache`): selects the growable buft per layer when
  `kv_lazy` is set; commits the exact graph window `[0, n_kv)` per active stream
  in `ensure_kv_window()` **before** `apply_ubatch` mutates cell metadata, so a
  commit OOM fails the ubatch before mutating any cell metadata. Frees the unused tail
  on `seq_rm` / `clear`.
- **recurrent state** (`llama_memory_recurrent`): same growable buft; commits the
  active cell range `[head, head+n)` across rollback groups in
  `ensure_state_window()`, wrapped by `find_slot_ensure()` which snapshots +
  restores metadata on OOM. Frees empty cells on `seq_rm` / `clear`.
- **fallback**: if a device lacks CUDA VMM (or `GGML_CUDA_NO_VMM`), the lazy buft
  is `NULL` and allocation transparently falls back to the eager path — identical
  behavior to today, no error, one-line log.

## Building (CUDA)

Canonical (matches the fork's normal release build):

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

On NixOS the CUDA toolchain must come from a nix shell (the RTX 4090 + 4070S are
both Ada = `sm_89`). The exact recipe used to develop + validate this feature:

```sh
# a complete merged CUDA 12.8 toolkit (nvcc + cudart + cccl + cublas + stubs):
CUDA=$(nix build --impure --no-link --print-out-paths --expr '
  let p = import <nixpkgs> { config.allowUnfree = true; config.cudaSupport = true; };
      c = p.cudaPackages_12_8;
      outs = p.lib.concatMap (x: map (o: x.${o}) (x.outputs or ["out"]))
             [ c.cuda_nvcc c.cuda_cudart c.cuda_cccl c.libcublas c.cuda_nvtx c.cuda_profiler_api c.cuda_nvrtc ];
  in p.symlinkJoin { name = "cuda-full-12.8"; paths = outs; }')

nix-shell -p cmake ninja cudaPackages_12_8.cuda_nvcc cudaPackages_12_8.cuda_cudart \
             cudaPackages_12_8.libcublas cudaPackages_12_8.cuda_cccl --run "
  cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=89 -DCUDAToolkit_ROOT=$CUDA \
        -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=c++
  cmake --build build -j
"
```

Run the resulting binaries with `LD_LIBRARY_PATH=$CUDA/lib:/run/opengl-driver/lib`.

## Heterogeneous multi-GPU balance (RTX 4090 24 GB + RTX 4070S 12 GB)

Because lazy KV commits **per device**, `--tensor-split` must balance
**weights + peak-committed KV + peak-committed SSM state** per device, not just
weights:

```
VRAM_dev  >=  W_dev
            + Σ_{attn layers on dev} ( n_kv_peak x (row_K + row_V) )
            + Σ_{recr layers on dev} ( n_active_seq x per_seq_RS )
```

Layer->device assignment uses the normalized `--tensor-split` in **detected
device order**. Run `llama-server --list-devices` first and set
`CUDA_VISIBLE_DEVICES` so index 0 is the 4090; then `--tensor-split 24,12`
entries map `[4090, 4070S]`.

Worked example — Qwen3.6-27B IQ4_XS (16 attn layers @ head256/GQA4 q8_0 KV
~34 KiB/token; 48 gated-DeltaNet layers ~150 MiB/seq), 16 slots x 32K tokens:

| Component            | total   | 4090 (`--tensor-split 24,…`) | 4070S (`…,12`) |
|----------------------|---------|------------------------------|----------------|
| weights (IQ4_XS)     | 14.3 GiB| ~9.6                         | ~4.7           |
| attn KV (peak)       | ~17 GiB | ~11.4                        | ~5.6           |
| SSM state (16 seq)   | ~2.4 GiB| ~1.6                         | ~0.8           |
| **total**            | ~33.7   | **~22.8 / 24**               | **~10.9 / 12** |

Tune from `24,12` using the V4 watermarks toward whichever card saturates first
(here the 4090). If a 16 x 32K workload OOMs, lower per-agent ctx or slot count.

## Runbook (dual-GPU icaro)

```sh
CUDA_VISIBLE_DEVICES=<4090-uuid>,<4070S-uuid> \
build/bin/llama-server \
  -m Qwen3.6-27B-IQ4_XS.gguf \
  -ngl 99 --tensor-split 24,12 \
  -fa on -ctk q8_0 -ctv q8_0 \
  --kv-lazy --parallel 16 -c 524288 \
  --host 127.0.0.1 --port 8080
```

(`--tensor-split` is the starting point; pin the tuned value after V4/V5.)

## Scope and limitations (v1)

- **Applies to** standard attention (`llama_kv_cache`), hybrid
  (`llama_memory_hybrid`), and recurrent (`llama_memory_recurrent`) caches — i.e.
  the large, unbounded-context caches lazy growth is designed for.
- **Strong lazy-V assumes `-fa on`**: with FlashAttention the V cache is contiguous
  per-token rows, so it commits exactly `[0, n_kv)` like K. Without FlashAttention V
  is transposed (a token spans one column across all rows), so `ensure_kv_window()`
  commits the full per-stream V region — K still grows lazily, V is effectively eager.
  The standard config (`-fa on`) gets full lazy behavior for both.
- **Window-bounded caches stay eager**: sliding-window (iSWA), DeepSeek DSA/DSV4,
  and hybrid+SWA caches keep the eager allocation and log a one-line warning when
  `--kv-lazy` is requested (their KV is already bounded by the window, so lazy
  growth buys little). This is intentional, not a silent no-op.
- **Grow-OOM handling**: a failed on-demand commit is rejected *before* it mutates
  cache state (the KV window is committed pre-mutation; recurrent metadata is
  snapshot-restored), so the failing ubatch leaves the cache consistent. The decode
  then returns an error; the server resets the in-flight slots, continues serving, and
  logs a clear diagnostic. *Isolated single-slot eviction* (error only the largest slot, keep the
  rest running) is **deferred** — doing it safely requires rolling back the
  server-side per-slot prompt state that `pre_decode()` advances before decode,
  which could not be verified without the target model + GPUs.
- **No copy-on-write page sharing between sequences, no swap-to-host.** Decommit
  is granule-aligned: a freed range reclaims only whole 2 MiB granules wholly
  inside it (edge granules shared with neighbors stay resident). For the recurrent
  `s_l` state (3.15 MiB rows) this reclaims ~1 granule/layer/seq; the tiny `r_l`
  rows effectively don't reclaim (acceptable — `s_l` is ~96% of the footprint).

## Phase status

- **Phase 1 (base lazy KV, non-MTP serving)** — implemented.
- **Phase 2 (lazy KV in the MTP + speculative path)** — implemented; **functional
  support verified locally**. Server draft/MTP contexts inherit `kv_lazy` via
  `common_context_params_to_llama`; `llama-speculative{,-simple}` now expose
  `--kv-lazy` too (added `LLAMA_EXAMPLE_SPECULATIVE`/`_LOOKUP` to the arg). On
  `Qwen3.6-35B-A3B-MTP` (`--spec-type draft-mtp --kv-lazy`, single 4070 SUPER) the
  main context (`(lazy) KV` + `(lazy) RS`) **and** the MTP draft context
  (`creating MTP draft context …` → `kv_lazy = true`, `(lazy) KV = 16 MiB`) both use
  the growable buft, and MTP decoding worked — 31/48 drafts accepted (65%).
  (Qwen3.6's MTP head is dense-attention-only: KV, no RS.) The large-context /
  acceptance-parity MTP gate is still deferred (see Verification).
- **Phase 3 (performant MTP + `--parallel N`)** — the multi-seq recurrent
  batching this phase targeted is **already present on upstream master**:
  `llm_graph_context::build_rs()` gathers the active sequences' states with
  `ggml_get_rows` into a contiguous `[state_size, n_seqs]` tensor and the gated
  DeltaNet / SSM scan runs batched over `n_seqs` (see `src/models/qwen35.cpp`
  `build_layer_attn_linear`). The open TODO from PR #22673 was resolved after the
  b7631 checkpoint this plan was written against, so no additional gather/scatter
  is needed. The optional D2H/H2D pre-norm embedding-transfer optimization
  (cross-context tensor sharing) is left as a future, profile-gated refinement.

## Verification

Unit test (device-local, deterministic — uses `ggml_backend_cuda_buffer_stats`,
not `nvidia-smi`):

```sh
build/bin/test-growable-buffer   # (built only with -DGGML_CUDA=ON)
```

**V1 — PASSED on RTX 4070 SUPER (VMM: yes, 2 MiB granularity):** a 1 GiB
reservation reports `committed == 0, commit_ops == 0`; the real allocator path
(`ggml_backend_alloc_ctx_tensors_from_buft`) commits 0 at alloc; a 32 MiB write
commits 16 granules; an interior `release_range` frees 14 granules
(`release_ops == 14`); readback is byte-identical before and after release/
re-commit.

### Measured locally — Qwen3.5-35B-A3B UD-Q4_K_XL, single RTX 4070 SUPER (12 GB)

Partial offload (the 22 GB model can't fully fit 12 GB):
`-ngl 999 --n-cpu-moe 32 -fa on -ctk q8_0 -ctv q4_0`; arch `qwen35moe` (40 layers,
attention + gated-DeltaNet recurrent, 256 experts / 8 used). Both the `(lazy) KV`
and `(lazy) RS` buffers engage (load log). These are this-setup numbers only — not
a dual-GPU or full-GPU-offload result.

- **V2 correctness — PASS.** `llama-cli -st --temp 0 --seed 42`, 107-tok prompt +
  64 greedy tokens: output byte-identical with and without `--kv-lazy`.
- **V3 throughput — PASS (no penalty).** `llama-batched-bench -npp 2048 -ntg 512
  -npl 1 -c 32768`, median of 3: prefill 679->675 t/s (-0.6%), decode 17.7->18.8
  t/s (+6.3%) — both inside this CPU-bound run's variance, far under the 20% gate.
  (Experts on CPU make decode CPU-bound; the pure-GPU-path overhead is best
  measured on the dual-GPU rig where the model fits entirely.)
- **V4 watermark — PASS, scales with ctx.** `-npp 512 -ntg 128 -npl 1`, peak GPU
  MiB (nvidia-smi), eager vs lazy:

  | ctx    | eager | lazy | saved |
  |--------|-------|------|-------|
  | 32768  |  9126 | 8985 |  ~141 |
  | 131072 | 10033 | 9030 | ~1003 |
  | 262144 | 11418 | 9371 | ~2047 |

  Eager reserves the whole-ctx KV up front; lazy commits only touched pages, so a
  short request on a 262K context frees ~2 GB (at 262K eager peaks ~860 MiB under
  the card's 12282 MiB; lazy leaves ~2.9 GB headroom). The saving tracks context
  size (`kv_unified=false`, n_seq_max=1 here); a separate npl=8 run at 32K showed a
  smaller gap dominated by the larger batch compute buffer, not KV.

### Still deferred to the dual-GPU rig (RTX 4090 + 4070S)

`--parallel N` end-to-end serving (V5), the dual-GPU tensor-split runbook (V6), and
the **large-context MTP gate** (`-c 262144` + acceptance parity vs non-lazy MTP) need
the full model resident across both cards:

```sh
# V5 multi-agent: runbook command above + N concurrent /v1/chat/completions streams
# V6 dual-GPU:   CUDA_VISIBLE_DEVICES=<4090>,<4070S> ... --tensor-split 24,12 --kv-lazy
# MTP large-ctx: llama-server -m <MTP-gguf> --spec-type draft-mtp --spec-draft-n-max 3 --kv-lazy -c 262144
#   (functional MTP+kv-lazy already verified locally at small ctx: 31/48 drafts accepted)
```
