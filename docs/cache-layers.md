# Per-layer KV cache control — `--cache-layer`

`--cache-layer` configures each attention layer's KV cache independently:
quantization type for K and V, an optional attention window (ablation mask), and
the Hadamard (QuaRot) rotation. Preset-compatible (`LLAMA_ARG_CACHE_LAYER`), so it
works in `models.ini` / llama-server router presets.

```
--cache-layer "IL=TYPEK[/TYPEV][:wN][:rot|:norot],..."

  IL        model layer index, or "*" = default for all attention layers
            (explicit IL wins over "*"; "*" wins over -ctk/-ctv)
  TYPEK/V   any KV cache type (f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl);
            one type sets both K and V
  :wN       restrict this layer's attention to the last N tokens (KQ-mask only:
            measurement/ablation tool, no memory savings in v1)
  :rot      force the QuaRot rotation on/off for this layer; default is AUTO =
  :norot    rotate iff this layer's cache type is quantized
```

Example — mixed precision from measured sensitivity, tail layers windowed:

```
--cache-layer "*=q4_0,3=q8_0,7=q8_0,11=q8_0,15=q8_0,19=q8_0,23=q8_0,27=q8_0,55=q4_0:w1024,59=q4_0:w1024"
```

Supported on the plain and hybrid (non-iSWA) `llama_kv_cache`; other cache classes
warn and ignore. `--kv-lazy` composes (per-layer row sizes are accounted in the
granule math). State save/load stores per-tensor types as usual.

## Measuring sensitivity

- `benches/layer-importance-sweep.sh` — per-layer KLD vs an f16 baseline
  (`--kl-divergence`), quant and window ablations (`SWEEP=quant|window|both`).
- `benches/kv-grid-sweep.sh` — K/V-separated grid: each layer × {K only, V only} ×
  {q8_0, q5_1, q4_0}.
- `benches/kv-pareto.py` — greedy accuracy-vs-size ladder from the grid CSV;
  emits `--cache-layer` specs for validation runs.

Use a corpus rendered with the model's own chat template on real workload data;
plain-prose PPL rankings do not transfer (measured: layer importance INVERTED
between wikitext PPL and real agent-transcript KLD on TC-Qwen3.6-27B).

Rotation caveat that motivated per-layer `:rot`: rotating f16 layers adds pure
rounding noise (~0.74 mean KLD measured across mixed configs at 32K); rotating
quantized layers helps (−0.02 KLD on a single-layer probe). AUTO does the right
thing per layer.

## Design: incremental cache degradation (planned)

Goal: start a session at f16 KV and re-quantize the cache downward as the context
grows, walking the measured Pareto curve instead of paying worst-case precision
up front.

Mechanism (`llama_kv_cache_requantize(il, k|v, new_type)`, between decodes):

1. Allocate a replacement tensor (same shape, new type) on the layer's buft.
   Under `--kv-lazy` the new tensor commits physical pages only for the used
   range `[0, used_max_p1)`.
2. Run a conversion pass: `ggml_cpy(old_range → new_range)` per stream
   (dequant→requant on device). Chunk the copy to bound the transient VRAM
   (old + new coexist only per chunk under VMM; whole-layer spike ≤ one layer's
   K or V otherwise — e.g. 256 MiB at 128K f16).
3. Swap the tensor into `layers[ikv]`, update the per-layer type bookkeeping,
   free/decommit the old tensor. Cell metadata (positions/sequences) is
   untouched. Graphs rebuild once on the next decode (type change invalidates
   graph reuse) — a rare, amortized event.
4. Rotation consistency: the stored basis must match `get_rot_k/v(il)`. If the
   downgrade turns AUTO-rotation on (f16→quantized), the conversion pass MUST
   apply the Hadamard rotation inline (cpy through `llama_mul_mat_hadamard`),
   or the layer stays `:norot` for the session (v1).

Policy (server side): thresholds on context fill (e.g. >32K → step 1, >96K →
step 2, ...), applying downgrades in the greedy Pareto ORDER — least-sensitive
(layer, tensor) first, i.e. the degradation schedule IS the Pareto walk. The
schedule ships as a preset string of the same `--cache-layer` grammar, one spec
per threshold.

Known accuracy caveat: re-quantizing already-quantized history compounds error
(f16→q8→q4 ≠ f16→q4). Measure the double-quant penalty before enabling more than
one downgrade step per tensor; if material, restrict the schedule to one step per
tensor (f16 → final type at its threshold), which also halves conversion traffic.
