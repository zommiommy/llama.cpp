#!/usr/bin/env python3
"""Fully-measured greedy KV-precision descent (no additivity assumption).

Start all-f16; at each step MEASURE the true KLD of every eligible single-tensor
downgrade (one ladder step) and apply the one with the lowest added-KLD per byte
saved. Iterate until everything is at the ladder bottom.

Cost: O(#tensors^2 * #levels) measurements (~2300 runs, ~30 h at 32K ctx).

Resumable + crash-safe:
  WORKDIR/measurements.jsonl  append-only log of every run (also the cache)
  WORKDIR/path.jsonl          chosen step per iteration (replayed on resume)
  WORKDIR/curve.csv           rewritten after each step: size_B_per_tok, measured_kld

Partial analysis at any time: the files above are valid after every single run.
"""
import argparse, json, math, os, subprocess, sys, time

LADDER = ["f16", "q8_0", "q5_1", "q4_0"]
BPE    = {"f16": 2.0, "q8_0": 34/32, "q5_1": 24/32, "q4_0": 18/32}

def canon_spec(state: dict, layers: list[int]) -> str:
    parts = []
    for l in layers:
        tk = LADDER[state[("k", l)]]
        tv = LADDER[state[("v", l)]]
        if tk == "f16" and tv == "f16":
            continue
        parts.append(f"{l}={tk}/{tv}")
    return ",".join(parts)

def size_bpt(state: dict, n_el: dict) -> float:
    return sum(BPE[LADDER[lvl]] * n_el[key[0]] for key, lvl in state.items())

def vram_free_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return 1 << 30  # no nvidia-smi: don't block

def measure(args, spec: str, cache: dict, mlog) -> float | None:
    if spec in cache:
        return cache[spec]
    cmd = [args.bin, "-m", args.model, "-f", args.corpus, "-c", str(args.ctx),
           "--chunks", str(args.chunks), "-fa", "on", "--parse-special",
           "--kl-divergence-base", args.kld_base, "--kl-divergence"]
    if spec:
        cmd += ["--cache-layer", spec]
    for attempt in range(1, args.retries + 1):
        while vram_free_mib() < args.min_vram_mib:
            print(f"# waiting for VRAM (< {args.min_vram_mib} MiB free)", flush=True)
            time.sleep(120)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.run_timeout)
        except subprocess.TimeoutExpired:
            print(f"# TIMEOUT attempt {attempt}: {spec}", flush=True)
            continue
        kld = None
        for line in proc.stdout.splitlines() + proc.stderr.splitlines():
            if "Mean" in line and "KLD:" in line:
                try:
                    kld = float(line.split("KLD:")[1].split("±")[0].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        rec = {"ts": time.time(), "spec": spec, "kld": kld, "rc": proc.returncode,
               "sec": round(time.time() - t0, 1), "attempt": attempt}
        mlog.write(json.dumps(rec) + "\n"); mlog.flush(); os.fsync(mlog.fileno())
        if proc.returncode == 0 and kld is not None:
            cache[spec] = kld
            return kld
        print(f"# run failed rc={proc.returncode} attempt {attempt}: {spec}", flush=True)
        time.sleep(30)
    return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--kld-base", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--bin", default="./build/bin/llama-perplexity")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--layers", default="3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63")
    ap.add_argument("--n-embd-k", type=int, default=1024)
    ap.add_argument("--n-embd-v", type=int, default=1024)
    ap.add_argument("--min-vram-mib", type=int, default=18000)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--run-timeout", type=int, default=900)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = run to completion")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split()]
    n_el = {"k": args.n_embd_k, "v": args.n_embd_v}
    os.makedirs(args.workdir, exist_ok=True)
    m_path = os.path.join(args.workdir, "measurements.jsonl")
    p_path = os.path.join(args.workdir, "path.jsonl")
    c_path = os.path.join(args.workdir, "curve.csv")

    # load caches / replay path
    cache: dict[str, float] = {}
    if os.path.exists(m_path):
        for line in open(m_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("rc") == 0 and r.get("kld") is not None:
                cache[r["spec"]] = r["kld"]
    state = {(tn, l): 0 for tn in ("k", "v") for l in layers}
    kld_cur, step0 = 0.0, 0
    if os.path.exists(p_path):
        for line in open(p_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            state[(r["key"][0], r["key"][1])] = r["to_lvl"]
            kld_cur, step0 = r["kld"], r["step"]
        print(f"# resumed at step {step0}, kld={kld_cur:.4f}", flush=True)

    mlog = open(m_path, "a")
    plog = open(p_path, "a")

    step = step0
    while True:
        cands = [k for k, lvl in state.items() if lvl + 1 < len(LADDER)]
        if not cands or (args.max_steps and step - step0 >= args.max_steps):
            break
        step += 1
        best = None
        for key in sorted(cands, key=lambda k: (k[1], k[0])):
            state[key] += 1
            spec = canon_spec(state, layers)
            kld = measure(args, spec, cache, mlog)
            state[key] -= 1
            if kld is None:
                print(f"# skipping unmeasurable candidate {key}", flush=True)
                continue
            d_kld  = kld - kld_cur
            d_size = (BPE[LADDER[state[key]]] - BPE[LADDER[state[key] + 1]]) * n_el[key[0]]
            ratio  = d_kld / d_size
            print(f"# step {step} cand {key} -> {LADDER[state[key]+1]}: kld={kld:.4f} d={d_kld:+.4f} ratio={ratio:+.6f}", flush=True)
            if best is None or ratio < best[0]:
                best = (ratio, key, kld)
        if best is None:
            print("# no measurable candidates left; stopping", flush=True)
            break
        ratio, key, kld = best
        state[key] += 1
        kld_cur = kld
        rec = {"step": step, "key": [key[0], key[1]], "to_lvl": state[key],
               "to": LADDER[state[key]], "spec": canon_spec(state, layers),
               "kld": kld, "ratio": ratio, "size_bpt": size_bpt(state, n_el)}
        plog.write(json.dumps(rec) + "\n"); plog.flush(); os.fsync(plog.fileno())
        print(f"== step {step}: {key} -> {LADDER[state[key]]} | kld={kld:.4f} size={rec['size_bpt']:.0f} B/tok ==", flush=True)

        with open(c_path, "w") as c:
            c.write("size_B_per_tok,measured_kld,spec\n")
            c.write(f"{2.0 * sum(n_el.values()) * len(layers):.0f},0.0,\n")
            for line in open(p_path):
                r = json.loads(line)
                c.write(f"{r['size_bpt']:.0f},{r['kld']:.6f},\"{r['spec']}\"\n")

    print("done", flush=True)

if __name__ == "__main__":
    main()
