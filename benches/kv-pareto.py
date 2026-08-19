#!/usr/bin/env python3
"""Build the accuracy-vs-KV-size Pareto curve from a kv-grid-sweep.sh CSV.

Greedy: start all-f16; repeatedly apply the single (layer, K|V, next-smaller-type)
downgrade with the best bytes_saved / delta_KLD ratio (delta from the measured
single-tensor KLDs, assuming additivity of small KLDs). Emits config points as
--cache-layer specs for validation runs, which measure the TRUE KLD of each point.

Usage: kv-pareto.py grid.csv [--points N] [--n-embd-k 1024 --n-embd-v 1024]
"""
import argparse, csv, sys

LADDER = ["f16", "q8_0", "q5_1", "q4_0"]           # downgrade order
BPE    = {"f16": 2.0, "q8_0": 34/32, "q5_1": 24/32, "q4_0": 18/32}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_csv")
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--n-embd-k", type=int, default=1024)
    ap.add_argument("--n-embd-v", type=int, default=1024)
    args = ap.parse_args()

    # kld[(tensor, layer)][type] = mean_kld vs f16
    kld: dict[tuple[str, int], dict[str, float]] = {}
    layers: set[int] = set()
    for row in csv.DictReader(open(args.grid_csv)):
        cfg = row["config"]
        if not (cfg.startswith("k_") or cfg.startswith("v_")):
            continue
        try:
            tensor, t, l = cfg.split("_", 1)[0], cfg.rsplit("_l", 1)[0][2:], int(cfg.rsplit("_l", 1)[1])
            val = float(row["mean_kld"])
        except ValueError:
            print(f"# skipping {cfg}: {row['mean_kld']}", file=sys.stderr)
            continue
        kld.setdefault((tensor, l), {"f16": 0.0})[t] = val
        layers.add(l)

    n_el = {"k": args.n_embd_k, "v": args.n_embd_v}
    state = {key: 0 for key in kld}                 # index into LADDER
    def size_bpt() -> float:                        # bytes per token, all layers
        return sum(BPE[LADDER[state[(tn, l)]]] * n_el[tn] for (tn, l) in state)
    def est_kld() -> float:
        return sum(kld[key].get(LADDER[state[key]], float("inf")) for key in state)

    size_f16 = size_bpt()
    steps = []                                      # (ratio, key, new_level)
    print(f"# start: {size_f16:.0f} B/token (f16), est KLD 0")
    trail = [(size_f16, 0.0, {})]
    while True:
        best = None
        for key in state:
            i = state[key]
            if i + 1 >= len(LADDER):
                continue
            cur_t, nxt_t = LADDER[i], LADDER[i + 1]
            if nxt_t not in kld[key] or cur_t not in kld[key] and cur_t != "f16":
                continue
            d_kld = kld[key].get(nxt_t, float("inf")) - kld[key].get(cur_t, 0.0)
            d_sz  = (BPE[cur_t] - BPE[nxt_t]) * n_el[key[0]]
            ratio = d_kld / d_sz if d_sz > 0 else float("inf")
            if best is None or ratio < best[0]:
                best = (ratio, key, i + 1, d_kld, d_sz)
        if best is None:
            break
        _, key, lvl, d_kld, d_sz = best
        state[key] = lvl
        trail.append((size_bpt(), est_kld(), dict(state)))

    # sample ~N points evenly in size between f16 and the smallest config
    lo, hi = trail[-1][0], trail[0][0]
    targets = [hi - (hi - lo) * i / (args.points - 1) for i in range(args.points)]
    picked, seen = [], set()
    for tgt in targets:
        pt = min(trail, key=lambda p: abs(p[0] - tgt))
        if id(pt) not in seen:
            seen.add(id(pt)); picked.append(pt)

    print("size_B_per_tok,est_kld,cache_layer_spec")
    for size, e, st in picked:
        # compact spec: group (k_type, v_type) per layer; omit pure-f16 layers
        parts = []
        for l in sorted(layers):
            tk = LADDER[st.get(("k", l), 0)]
            tv = LADDER[st.get(("v", l), 0)]
            if tk == "f16" and tv == "f16":
                continue
            parts.append(f"{l}={tk}/{tv}")
        print(f"{size:.0f},{e:.4f},\"{','.join(parts)}\"")

if __name__ == "__main__":
    main()
