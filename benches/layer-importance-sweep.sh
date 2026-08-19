#!/usr/bin/env bash
# Per-layer KV importance sweep for hybrid qwen35 models (--cache-layer knob).
#   Sweep A (quant): one layer's KV dropped f16 -> q4_0/q4_0, rest f16.
#   Sweep B (range): one layer's attention windowed to the last W tokens, types f16.
# Metric: KL divergence of each ablated run against the unablated baseline's
# full logit distribution (--kl-divergence), which is far more sensitive than
# corpus perplexity and independent of how predictable the corpus text is.
# NOTE: the baseline logits file is ~2 B x n_vocab x n_tokens (~40 GB for 131K
# tokens at 151K vocab) -- point KLD_BASE at a disk-backed path.
# Output: CSV "config,mean_kld,kld_err,same_top_pct,d_ppl" on stdout / $OUT.
set -euo pipefail

MODEL=${MODEL:-/home/zom/.cache/llama.cpp/Qwen3.8-27B-IQ4_XS.gguf}
CORPUS=${CORPUS:-/tmp/wt2.txt}
CTX=${CTX:-16384}
CHUNKS=${CHUNKS:-6}
WINDOW=${WINDOW:-1024}
LAYERS=${LAYERS:-"3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63"}
BIN=${BIN:-./build/bin/llama-perplexity}
OUT=${OUT:-/tmp/layer-sweep.csv}
KLD_BASE=${KLD_BASE:-/tmp/kld-base.bin}

run() { # $1 = label, $2 = --cache-layer spec ("" = none)
    local extra=()
    if [ -n "$2" ]; then
        extra=(--cache-layer "$2")
    fi
    local log rc kld same dppl
    set +e
    log=$("$BIN" -m "$MODEL" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" -fa on --parse-special \
          --kl-divergence-base "$KLD_BASE" --kl-divergence "${extra[@]}" 2>&1 | tail -60)
    rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        echo "$1,CRASHED(rc=$rc),,," | tee -a "$OUT"
        echo "--- $1 crash tail ---" >> "${OUT}.err"; echo "$log" | tail -20 >> "${OUT}.err"
        return 0
    fi
    kld=$( echo "$log" | grep -iE 'Mean\s+KLD:' | sed -E 's/.*KLD: *([-0-9.eE]+) *. *([-0-9.eE]+).*/\1,\2/' | head -1 || true)
    same=$(echo "$log" | grep -iE 'same top'    | grep -oE '[0-9.]+ *%' | head -1 | tr -d ' %' || true)
    dppl=$(echo "$log" | grep -iE 'ln\(PPL\(Q\)/PPL\(base\)\)' | grep -oE '[-0-9.]+' | head -1 || true)
    echo "$1,${kld:-FAILED,FAILED},${same:-?},${dppl:-?}" | tee -a "$OUT"
}

: > "$OUT"
echo "config,mean_kld,kld_err,same_top_pct,d_ppl" | tee -a "$OUT"

# baseline pass: saves the full-precision (unablated) logits
if [ ! -f "$KLD_BASE" ]; then
    echo "# saving baseline logits to $KLD_BASE ..." >&2
    "$BIN" -m "$MODEL" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" -fa on --parse-special \
        --kl-divergence-base "$KLD_BASE" 2>&1 | grep -E 'Final estimate|ETA' | tail -1 >&2
fi

run "self_check_f16" ""                 # KLD of baseline vs itself: must be ~0

if [ "${SWEEP:-both}" != "window" ]; then
    run "all_q4" "*=q4_0/q4_0"
    for il in $LAYERS; do
        run "quant_l${il}" "${il}=q4_0/q4_0"
    done
fi

if [ "${SWEEP:-both}" != "quant" ]; then
    run "all_w${WINDOW}" "*=f16:w${WINDOW}"
    for il in $LAYERS; do
        run "window_l${il}" "${il}=f16:w${WINDOW}"
    done
fi

echo "done -> $OUT"
