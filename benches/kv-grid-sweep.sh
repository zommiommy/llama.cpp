#!/usr/bin/env bash
# K/V-separated per-layer quantization sensitivity grid (KLD vs f16 baseline).
# For each attention layer, quantize ONLY K or ONLY V to each type in TYPES,
# rest of the cache f16. Output: CSV "config,mean_kld,kld_err".
set -euo pipefail

MODEL=${MODEL:?}
CORPUS=${CORPUS:?}
CTX=${CTX:-32768}
CHUNKS=${CHUNKS:-1}
LAYERS=${LAYERS:-"3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63"}
TYPES=${TYPES:-"q8_0 q5_1 q4_0"}
BIN=${BIN:-./build/bin/llama-perplexity}
OUT=${OUT:?}
KLD_BASE=${KLD_BASE:?}

run() { # $1 = label, $2 = --cache-layer spec
    if [ -f "$OUT" ] && grep -q "^$1," "$OUT"; then
        return 0 # already measured (resume)
    fi
    local log rc kld
    set +e
    log=$("$BIN" -m "$MODEL" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" -fa on --parse-special \
          --kl-divergence-base "$KLD_BASE" --kl-divergence --cache-layer "$2" 2>&1 | tail -60)
    rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        echo "$1,CRASHED(rc=$rc)," | tee -a "$OUT"
        echo "--- $1 ---" >> "${OUT}.err"; echo "$log" | tail -20 >> "${OUT}.err"
        return 0
    fi
    kld=$(echo "$log" | grep -iE 'Mean\s+KLD:' | sed -E 's/.*KLD: *([-0-9.eE]+) *. *([-0-9.eE]+).*/\1,\2/' | head -1 || true)
    echo "$1,${kld:-FAILED,}" | tee -a "$OUT"
}

if [ ! -f "$OUT" ] || [ "${RESUME:-0}" != "1" ]; then
    : > "$OUT"
    echo "config,mean_kld,kld_err" | tee -a "$OUT"
fi

if [ ! -f "$KLD_BASE" ]; then
    echo "# saving baseline logits to $KLD_BASE ..." >&2
    "$BIN" -m "$MODEL" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" -fa on --parse-special \
        --kl-divergence-base "$KLD_BASE" 2>&1 | grep -E 'Final estimate' | tail -1 >&2
fi

for t in $TYPES; do
    for il in $LAYERS; do
        run "k_${t}_l${il}" "${il}=${t}/f16"
        run "v_${t}_l${il}" "${il}=f16/${t}"
    done
done

echo "done -> $OUT"
