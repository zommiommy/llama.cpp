#!/usr/bin/env bash
# Per-layer KV importance sweep for hybrid qwen35 models (--cache-layer knob).
#   Sweep A (quant): one layer's KV dropped f16 -> q4_0/q4_0, rest f16.
#   Sweep B (range): one layer's attention windowed to the last W tokens, types f16.
# Output: CSV "config,ppl,err" on stdout / $OUT.
set -euo pipefail

MODEL=${MODEL:-/home/zom/.cache/llama.cpp/Qwen3.8-27B-IQ4_XS.gguf}
CORPUS=${CORPUS:-/tmp/wt2.txt}
CTX=${CTX:-16384}
CHUNKS=${CHUNKS:-6}
WINDOW=${WINDOW:-1024}
LAYERS=${LAYERS:-"3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63"}
BIN=${BIN:-./build/bin/llama-perplexity}
OUT=${OUT:-/tmp/layer-sweep.csv}

run() { # $1 = label, $2 = --cache-layer spec ("" = none)
    local extra=()
    [ -n "$2" ] && extra=(--cache-layer "$2")
    local line
    line=$("$BIN" -m "$MODEL" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" -fa on "${extra[@]}" 2>&1 \
           | grep 'Final estimate' | sed -E 's/.*PPL = ([0-9.]+) \+\/- ([0-9.]+).*/\1,\2/')
    echo "$1,${line:-FAILED,FAILED}" | tee -a "$OUT"
}

: > "$OUT"
echo "config,ppl,err" | tee -a "$OUT"

run "baseline_f16" ""
run "all_q4"       "*=q4_0/q4_0"
run "all_w${WINDOW}" "*=f16:w${WINDOW}"

for il in $LAYERS; do
    run "quant_l${il}" "${il}=q4_0/q4_0"
done

for il in $LAYERS; do
    run "window_l${il}" "${il}=f16:w${WINDOW}"
done

echo "done -> $OUT"
