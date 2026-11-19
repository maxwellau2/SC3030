"""Cuttlefish proof-of-concept: rank races by wall-clock reaction time vs by virtual time.

Two traders A and B run the same strategy; B executes exactly k extra instructions before
submit_trade, so A should win EVERY race. We measure, on this machine:
  * wall-clock reaction time (perf_counter_ns from event delivery to submit) - what a
    reaction-time stopwatch (DBO) or a timestamp (CloudEx) effectively sees
  * virtual reaction time (instruction-cost count) - what Cuttlefish ranks by
Run (from experiments/):  .venv/bin/python -I poc/cuttlefish/demo.py
"""
import gc, json, os, platform, random, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vm import Participant, moving_average_strategy  # noqa: E402
import native  # noqa: E402

KS = [1, 10, 100, 1000, 10000]
EVENTS = 2000
SPACING = 1_000_000        # virtual time between event releases (>> any strategy cost)
JITTER_RUNS = 20000


def market(n, seed=7):
    rng = random.Random(seed)
    p, out = 50.0, []
    for _ in range(n):
        p = max(1.0, p + rng.gauss(0, 1.0))
        out.append({"price": p})
    return out


def race(k, events, seed_order=0):
    """Run A (base) and B (base + k NOPs) on every event; alternate who runs first."""
    A = Participant("A", moving_average_strategy(0))
    B = Participant("B", moving_average_strategy(k))
    wall_a, wall_b = [], []
    gc.disable()
    try:
        for i, ev in enumerate(events):
            D = (i + 1) * SPACING
            order = (A, B) if (i + seed_order) % 2 == 0 else (B, A)
            for p in order:
                t0 = time.perf_counter_ns()
                p.on_event(ev, D)
                rt = p.trades[-1][1] - t0
                (wall_a if p is A else wall_b).append(rt)
    finally:
        gc.enable()
    va = np.array([t[0] for t in A.trades]) - (np.arange(len(events)) + 1) * SPACING
    vb = np.array([t[0] for t in B.trades]) - (np.arange(len(events)) + 1) * SPACING
    wa, wb = np.array(wall_a), np.array(wall_b)
    return {
        "wall_A_wins": float(np.mean(wa < wb) + 0.5 * np.mean(wa == wb)),  # ties = coin flip
        "wall_ties": float(np.mean(wa == wb)),
        "virtual_A_wins": float(np.mean(va < vb)),
        "virtual_gap_units": float(np.median(vb - va)),
        "wall_gap_median_ns": float(np.median(wb - wa)),
        "_wall_outcome": (wa < wb).tolist(),
        "_virtual_outcome": (va < vb).tolist(),
    }


def jitter(events, runs=JITTER_RUNS):
    """Identical program, identical input, run repeatedly: spread of wall-clock time."""
    P = Participant("J", moving_average_strategy(0))
    ev = events[0]
    ts = []
    gc.disable()
    try:
        for i in range(runs):
            t0 = time.perf_counter_ns()
            P.on_event(ev, (i + 1) * SPACING)
            ts.append(P.trades[-1][1] - t0)
    finally:
        gc.enable()
    ts = np.array(ts, dtype=float)
    q = np.percentile(ts, [25, 50, 75, 99, 99.9])
    return {"median_ns": q[1], "iqr_ns": q[2] - q[0], "p99_ns": q[3], "p99_9_ns": q[4],
            "p99_9_minus_median_ns": q[4] - q[1], "max_ns": float(ts.max()), "runs": runs}


def start_noise(n):
    """Noisy neighbours: n separate busy-loop processes competing for CPU."""
    return [subprocess.Popen([sys.executable, "-I", "-c", "while True: pass"]) for _ in range(n)]


def stop_noise(procs):
    for p in procs:
        p.kill()
    for p in procs:
        p.wait()


def main():
    t_start = time.time()
    events = market(EVENTS)
    res = {"machine": {"platform": platform.platform(), "python": platform.python_version(),
                       "cpus": os.cpu_count(),
                       "perf_counter_resolution_s": time.get_clock_info("perf_counter").resolution},
           "events_per_race": EVENTS, "ks": KS, "quiet": {}, "loaded": {}}

    # warm-up + build native race
    race(1, events[:200])
    native.build(KS)

    for cond in ("quiet", "loaded"):
        procs = start_noise(os.cpu_count() or 4) if cond == "loaded" else []
        try:
            if procs:
                time.sleep(1.0)
            res[cond]["jitter"] = jitter(events)
            for k in KS:
                r = race(k, events)
                res[cond][str(k)] = {kk: v for kk, v in r.items() if not kk.startswith("_")}
                print(f"[{cond:6s}] k={k:>5}: A wins wall-clock {100*r['wall_A_wins']:6.2f}% "
                      f"(ties {100*r['wall_ties']:.1f}%) | virtual {100*r['virtual_A_wins']:.1f}% "
                      f"| median wall gap {r['wall_gap_median_ns']:.0f} ns", flush=True)
            nat = native.run()
            res[cond]["native"] = nat
            for r in nat["results"]:
                print(f"[{cond:6s}] NATIVE k={r['k']:>5}: A wins wall-clock {100*r['A_wins']:6.2f}% "
                      f"(ties {100*r['tie_share']:.1f}%) | mean wall gap {r['mean_gap_ns']:.1f} ns", flush=True)
            nj = nat["jitter"]
            print(f"[{cond:6s}] NATIVE identical-program jitter: median {nj['median_ns']} ns, IQR {nj['iqr_ns']} ns, "
                  f"p99 {nj['p99_ns']} ns, p99.9 {nj['p99_9_ns']} ns, max {nj['max_ns']} ns", flush=True)
            j = res[cond]["jitter"]
            print(f"[{cond:6s}] identical-program jitter: median {j['median_ns']:.0f} ns, IQR {j['iqr_ns']:.0f} ns, "
                  f"p99 {j['p99_ns']:.0f} ns, p99.9 {j['p99_9_ns']:.0f} ns", flush=True)
        finally:
            stop_noise(procs)

    # determinism: same race twice
    r1, r2 = race(10, events), race(10, events)
    res["determinism_k10"] = {
        "virtual_outcomes_identical": r1["_virtual_outcome"] == r2["_virtual_outcome"],
        "wall_outcome_agreement": float(np.mean(np.array(r1["_wall_outcome"]) == np.array(r2["_wall_outcome"]))),
    }
    print("determinism (k=10, two identical runs):", res["determinism_k10"])
    res["runtime_s"] = time.time() - t_start
    json.dump(res, open(HERE / "results.json", "w"), indent=2, default=float)
    print(f"saved results.json in {res['runtime_s']:.1f}s")
    plot(res)


THEMES = {
    "dark": dict(bg="#0b1220", ink="#e5e7eb", ink2="#94a3b8", grid="#1e293b", ref="#94a3b8",
                 wall="#3987e5", vt="#0ca30c"),
    "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", grid="#e7e5e4", ref="#6b7280",
                  wall="#2a78d6", vt="#008300"),
}


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for th, c in THEMES.items():
        plt.rcParams.update({
            "figure.facecolor": c["bg"], "axes.facecolor": c["bg"], "savefig.facecolor": c["bg"],
            "axes.edgecolor": c["grid"], "axes.labelcolor": c["ink2"], "text.color": c["ink"],
            "xtick.color": c["ink2"], "ytick.color": c["ink2"], "grid.color": c["grid"],
            "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
            "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
            "lines.linewidth": 2.2, "lines.markersize": 7, "legend.frameon": False, "legend.fontsize": 11.5,
        })
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        x = np.array(KS)
        vt = [100 * res["quiet"][str(k)]["virtual_A_wins"] for k in KS]
        panels = [
            ("Python VM (≈110 ns per instruction)",
             [100 * res["quiet"][str(k)]["wall_A_wins"] for k in KS],
             [100 * res["loaded"][str(k)]["wall_A_wins"] for k in KS], res["quiet"]["jitter"], "Python"),
            ("Native arm64 C (≈0.25 ns per instruction)",
             [100 * r["A_wins"] for r in res["quiet"]["native"]["results"]],
             [100 * r["A_wins"] for r in res["loaded"]["native"]["results"]], res["quiet"]["native"]["jitter"], "native"),
        ]
        for ax, (title, wq, wl, jit, kind) in zip(axes, panels):
            ax.plot(x, vt, "-o", color=c["vt"], label="rank by virtual time (Cuttlefish)")
            ax.plot(x, wq, "-o", color=c["wall"], label="rank by wall-clock time, idle")
            ax.plot(x, wl, "--o", color=c["wall"], label="… with noisy-neighbour load")
            ax.axhline(50, color=c["ref"], ls=":", lw=1)
            ax.text(1.05, 51.5, "coin flip", color=c["ink2"], fontsize=10)
            ax.set_xscale("log"); ax.set_ylim(40, 103)
            ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
            ax.set_xlabel("extra instructions run by the slower trader (k)")
            ax.set_title(title, fontsize=13)
            iqr = jit["iqr_ns"]; p999 = jit["p99_9_ns"]
            note = (f"same program re-run: IQR {iqr:.0f} ns, p99.9 {p999:.0f} ns" +
                    (" (clock tick 41.7 ns)" if kind == "native" else ""))
            ax.text(0.98, 0.05, note, transform=ax.transAxes, ha="right", fontsize=9.5, color=c["ink2"])
        axes[0].set_ylabel("races won by the faster trader (%)")
        axes[0].legend(loc="center right")
        fig.suptitle("PoC: virtual time ranks every race correctly; wall-clock can't see small gaps",
                     x=0.01, ha="left", fontweight="bold", fontsize=15, color=c["ink"])
        fig.tight_layout()
        fig.savefig(HERE / f"cuttlefish_poc_{th}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
