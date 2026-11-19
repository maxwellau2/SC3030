"""Run all experiments on the authors' real Azure latency trace.

  E0  Reproduction of DBO paper's simulation results (Direct, CloudEx thresholds, DBO, MaxRTT)
  E1  NEW: CloudEx under clock-synchronisation error            (tests Beyond Lamport's critique)
  E2  NEW: Probabilistic ordering (Beyond Lamport) vs CloudEx's deterministic timestamps
  E3  NEW: "Fairness has a radius": one distant participant      (DBO Theorem 3 + straggler rule)
  E4  NEW: Compute jitter vs reaction-time ranking               (Cuttlefish's argument)
  E5  NEW: Placement advantage (static per-VM latency offsets)   (crypto-style AZ racing)

Usage (from experiments/):  .venv/bin/python -I fairlab/run_experiments.py [E0 E1 ...]
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from azure_trace import load_latency_trace  # noqa: E402
import sim  # noqa: E402

OUT = HERE.parent / "results"
OUT.mkdir(exist_ok=True)
T = 1_000_000
N = 10
SEEDS = range(10)
TRACE = load_latency_trace()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(name, rows):
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{name}.csv", index=False)
    log(f"saved {name}.csv ({len(df)} rows)")
    return df


# ------------------------------------------------------------------ E0
def e0():
    env = sim.make_env(TRACE, n=N, T=T)
    rows = []
    runs = [sim.direct(env), sim.dbo(env, 20, 25), sim.max_rtt(env)]
    runs += [sim.cloudex(env, d, d) for d in (10, 15, 20, 30, 40, 50, 70, 100)]
    for r in runs:
        st = sim.latency_stats(r)
        rows.append({"algo": r.name, "fairness": sim.fairness(env, r), **st})
        log(rows[-1])
    return save("e0_reproduction", rows)


# ------------------------------------------------------------------ E1 + E2
SIGMAS = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]  # us


def e1_e2():
    env = sim.make_env(TRACE, n=N, T=T)
    base = {"direct": sim.fairness(env, sim.direct(env)), "dbo": sim.fairness(env, sim.dbo(env, 20, 25))}
    rows1, rows2 = [], []
    # (a) exchange-released data: gateway release and stamp share one clock
    for d in (15, 30, 50):
        for sg in SIGMAS:
            for seed in SEEDS:
                th = np.random.default_rng(seed).normal(0.0, sg, N) if sg > 0 else np.zeros(N)
                r = sim.cloudex(env, d, d, offsets=th)
                f = sim.fairness(env, r)
                rows1.append({"scenario": "released", "hold_us": d, "sigma_us": sg, "seed": seed,
                              "fairness": f, "mean_latency": float(np.mean(r.latency)),
                              "dbo_fairness": base["dbo"], "direct_fairness": base["direct"]})
            log(f"E1a hold={d} sigma={sg}: fairness mean={np.mean([x['fairness'] for x in rows1[-len(SEEDS):]]):.4f}")
    # (b) externally-triggered events: offsets no longer cancel
    for sg in SIGMAS:
        for seed in SEEDS:
            th = np.random.default_rng(seed).normal(0.0, sg, N) if sg > 0 else np.zeros(N)
            r = sim.cloudex_external(env, 50, th)
            f = sim.fairness(env, r)
            rows1.append({"scenario": "external", "hold_us": 50, "sigma_us": sg, "seed": seed,
                          "fairness": f, "mean_latency": np.nan,
                          "dbo_fairness": base["dbo"], "direct_fairness": base["direct"]})
            ts = sim.tommy_scores(env, r.extra["stamps"], sg)
            rows2.append({"sigma_us": sg, "seed": seed, "cloudex_correct": f,
                          "cloudex_wrong": 1 - f, **{f"tommy_{k}": v for k, v in ts.items()}})
        log(f"E1b external sigma={sg}: fairness mean={np.mean([x['fairness'] for x in rows1[-len(SEEDS):]]):.4f}")
    save("e1_clock_error", rows1)
    save("e2_probabilistic", rows2)


# ------------------------------------------------------------------ E3
EXTRA = [0, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]  # us one-way


def e3():
    k = 5  # the distant participant (mid-range response time)
    others = [i for i in range(N) if i != k]
    rt = sim.response_times(N)
    # only races the distant trader DESERVES to win (it reacts faster than these 5 rivals);
    # judging all 9 pairs would credit it for correctly losing to the 4 faster rivals.
    pairs_k = [(k, j) for j in others if rt[k] < rt[j]]
    rows = []
    for D in EXTRA:
        extra = np.zeros(N); extra[k] = D
        env = sim.make_env(TRACE, n=N, T=T, extra_owd=extra)
        r_wait = sim.dbo(env, 20, 25)
        r_excl = sim.dbo(env, 20, 25, exclude=(k,))
        r_dir = sim.direct(env)
        rows.append({
            "extra_owd_us": D, "extra_rtt_us": 2 * D,
            "wait_others_mean_latency": float(np.mean(r_wait.latency[others])),
            "wait_fairness_all": sim.fairness(env, r_wait),
            "excl_others_mean_latency": float(np.mean(r_excl.latency[others])),
            "excl_fairness_distant_pairs": sim.fairness_by_exec(env, r_excl, pairs_k),
            "direct_fairness_distant_pairs": sim.fairness(env, r_dir, pairs_k),
        })
        log(rows[-1])
    return save("e3_radius", rows)


# ------------------------------------------------------------------ E4
GAPS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]  # us (1 ns ... 3 us)
JIT_SIGMA = 20e-3 / 1.349      # normal core with IQR = 20 ns (Cuttlefish Fig. 1)
JIT_TAIL_P, JIT_TAIL_MEAN = 0.003, 1.0  # => P(jitter > 1 us) ~ 0.1% (Cuttlefish: >1 us at p99.9)


def jitter(rng, size):
    j = rng.normal(0.0, JIT_SIGMA, size)
    tail = rng.random(size) < JIT_TAIL_P
    j[tail] += rng.exponential(JIT_TAIL_MEAN, tail.sum())
    return j


def e4():
    rows = []
    for gap in GAPS:
        rt = np.array([5.0 + gap, 5.0])  # participant 1 is faster by `gap`
        for seed in range(5):
            rng = np.random.default_rng(100 + seed)
            rt_act = rt[:, None] + jitter(rng, (2, T))
            env = sim.make_env(TRACE, n=2, T=T, rt=rt, rt_actual=rt_act)
            r = sim.dbo(env, 20, 25)
            rows.append({"gap_us": gap, "seed": seed,
                         "dbo_fairness_with_jitter": sim.fairness(env, r),
                         "virtual_time_fairness": 1.0})  # ranking by instruction count is deterministic
        log(f"E4 gap={gap}us: DBO fairness={np.mean([x['dbo_fairness_with_jitter'] for x in rows[-5:]]):.4f}")
    # empirical check of the jitter model
    rng = np.random.default_rng(0); j = jitter(rng, 2_000_000)
    q = np.percentile(j, [25, 75, 99.9])
    json.dump({"iqr_ns": float((q[1] - q[0]) * 1e3), "p99_9_us": float(q[2])},
              open(OUT / "e4_jitter_model.json", "w"))
    return save("e4_compute_jitter", rows)


# ------------------------------------------------------------------ E5
SPREADS = [0, 0.5, 1, 2, 5, 10, 20]  # us: per-VM static extra one-way delay ~ U(0, B)


def e5():
    rows = []
    for B in SPREADS:
        for seed in SEEDS:
            extra = np.random.default_rng(1000 + seed).uniform(0, B, N) if B > 0 else np.zeros(N)
            env = sim.make_env(TRACE, n=N, T=T, extra_owd=extra)
            for r in (sim.direct(env), sim.cloudex(env, 15, 15), sim.cloudex(env, 30, 30), sim.dbo(env, 20, 25)):
                rows.append({"spread_us": B, "seed": seed, "algo": r.name,
                             "fairness": sim.fairness(env, r), "mean_latency": float(np.mean(r.latency))})
        log(f"E5 spread={B}us done")
    return save("e5_placement", rows)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["E0", "E1", "E3", "E4", "E5"]
    fns = {"E0": e0, "E1": e1_e2, "E2": e1_e2, "E3": e3, "E4": e4, "E5": e5}
    for name in dict.fromkeys(todo):
        t0 = time.time(); log("=== start", name); fns[name](); log(f"=== {name} done in {time.time()-t0:.0f}s")
