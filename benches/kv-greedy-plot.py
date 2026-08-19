#!/usr/bin/env python3
"""Plot the measured greedy KV-precision descent: cache size vs mean KLD.

Reads WORKDIR/path.jsonl (one chosen downgrade per line) — valid mid-run, so
this can be pointed at a live sweep for partial analysis.

Usage:
    kv-greedy-plot.py ~/models/kv-greedy-tc36 [-o out.png] [--linear]
                      [--anchors "KQ8VQ8:34816:0.782,KQ8VQ4:26624:1.245,..."]

Anchors default to the uniform configs measured on TC-Qwen3.6 @32K (omp corpus).
NixOS: run under `nix-shell -p python3Packages.matplotlib` if needed.
"""
import argparse, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEF_ANCHORS = "KQ8VQ8:34816:0.782,KQ8VQ4:26624:1.245,uniform q5_1:24576:1.098,uniform q4_0:18432:1.836"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("-o", "--out", default=None, help="output PNG (default: WORKDIR/curve.png)")
    ap.add_argument("--anchors", default=DEF_ANCHORS, help='comma list "label:size:kld", "" to disable')
    ap.add_argument("--linear", action="store_true", help="linear y axis (default: symlog)")
    ap.add_argument("--f16-size", type=float, default=65536.0, help="all-f16 bytes/token starting point")
    args = ap.parse_args()

    p_path = os.path.join(args.workdir, "path.jsonl")
    steps = []
    for line in open(p_path):
        try:
            steps.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not steps:
        sys.exit(f"no steps in {p_path} yet")

    sizes = [s["size_bpt"] for s in steps]
    klds  = [s["kld"]      for s in steps]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Measured greedy KV-precision descent (per-tensor, per-layer)")
    ax.plot([s / 1024 for s in sizes], klds, marker=".", ms=4, lw=1.2,
            color="tab:blue", label=f"measured greedy path ({len(steps)} steps; f16 start = {args.f16_size/1024:.0f} KiB, KLD 0)")

    last = steps[-1]

    # log-linear regression ln(KLD) = a*size + b over ALL positive path points.
    # Drawn SOLID over its support only; dotted beyond = extrapolation, trust
    # accordingly. The doubling distance is reported only when the support spans
    # enough size range for the slope to mean anything.
    xs = np.array(sizes, dtype=float); ys = np.array(klds, dtype=float)
    mask = ys > 0
    if mask.sum() >= 4:
        xm, ym = xs[mask], np.log(ys[mask])
        a, b = np.polyfit(xm, ym, 1)
        pred = a * xm + b
        ss_tot = float(np.sum((ym - ym.mean()) ** 2))
        r2 = 1.0 - float(np.sum((ym - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        span_kib = (xm.max() - xm.min()) / 1024
        xsup = np.linspace(xm.min(), xm.max(), 50)
        ax.plot(xsup / 1024, np.exp(a * xsup + b), "-", lw=1.4, color="tab:green",
                label=f"log-linear fit (support {span_kib:.1f} KiB, R²={r2:.3f}): KLD = exp({a*1024:.3f}·KiB {b:+.2f})")
        xext = np.linspace(18432, xm.min(), 100)
        ax.plot(xext / 1024, np.exp(a * xext + b), ":", lw=1.0, color="tab:green",
                label="extrapolation (beyond fit support)")
        if a < 0 and span_kib >= 4.0:
            ax.set_title(f"Measured greedy KV-precision descent — KLD doubles every "
                         f"{-np.log(2)/a/1024:.2f} KiB/token removed (support {span_kib:.1f} KiB)")
    ax.annotate(f'step {last["step"]}: {last["key"][0]}{last["key"][1]}→{last["to"]}',
                xy=(last["size_bpt"] / 1024, last["kld"]),
                xytext=(6, 6), textcoords="offset points", fontsize=8, color="tab:blue")

    if args.anchors:
        for a in args.anchors.split(","):
            label, size, kld = a.rsplit(":", 2)
            ax.scatter(float(size) / 1024, float(kld), marker="*", s=140,
                       color="tab:red", zorder=5)
            ax.annotate(label, xy=(float(size) / 1024, float(kld)),
                        xytext=(6, -10), textcoords="offset points", fontsize=8, color="tab:red")

    if not args.linear:
        ax.set_yscale("log")
    ax.set_xlabel("KV cache size (KiB / token)")
    ax.set_ylabel("mean KLD vs f16 baseline")
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()  # descent reads left→right
    ax.legend(loc="upper left")

    out = args.out or os.path.join(args.workdir, "curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"{out}  ({len(steps)} steps, latest: size={last['size_bpt']:.0f} B/tok kld={last['kld']:.4f})")

if __name__ == "__main__":
    main()
