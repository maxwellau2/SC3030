"""PoC of probabilistic fair ordering ("Beyond Lamport", HotNets'25).

(a) Online probabilistic sequencer. External events (news / another venue) hit all
    clients at the same true time; client i reacts after rt_i and stamps the message
    with its OWN clock, whose error is drawn from a distribution N(mu_i, sigma_i) that the
    sequencer knows. Messages + heartbeats travel over real Azure-trace links (FIFO).
    The sequencer emits a message once it is (with prob. p_safe) truly earlier than
    anything still in flight, orders emitted messages by estimated true time, and
    starts a new batch only when P(previous before next) > p_thresh; otherwise it is a
    tie. Baseline: plain timestamp order (what CloudEx/Onyx do).
(b) Intransitivity: with non-Gaussian clock-error distributions, "likely-happened-before"
    can form a cycle (A>B, B>C, C>A). Probabilities computed by FFT convolution, as in the
    paper (§3.3). Gaussians are checked to stay transitive.
"""
import sys, json, heapq
from pathlib import Path
from itertools import permutations
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core import Sim, Link, response_times  # noqa: E402
from protocols import FifoLink  # noqa: E402

OUT = HERE.parent / "results"


def online(sigma=1.0, n=10, points=3000, G=10.0, hb=5.0, p_thresh=0.75, p_safe=0.999, seed=0):
    sim = Sim()
    rng = np.random.default_rng(seed)
    rts = response_times(n)
    sig = sigma * rng.uniform(0.5, 1.5, n)          # each client's (known) clock-error spread
    mu = np.zeros(n)
    z = norm.ppf(p_safe)
    links = [FifoLink(sim, Link(k=i)) for i in range(n)]
    last_hb = [-np.inf] * n
    pending = []                                      # heap by upper bound of true time
    out = []                                          # (batch, est, client, point, true, emit_t)
    state = {"batch": 0, "prev": None}

    def stamp(i):
        return sim.now + rng.normal(mu[i], sig[i])

    def trigger(k):
        g = sim.now
        for i in range(n):
            sim.after(rts[i], send_msg, i, k, g + rts[i])
        if k + 1 < points:
            sim.after(G, trigger, k + 1)

    def send_msg(i, k, true_t):
        links[i].send(seq_in, "msg", i, stamp(i), k, true_t)

    def heartbeat(i):
        links[i].send(seq_in, "hb", i, stamp(i), None, None)
        sim.after(hb, heartbeat, i)

    def seq_in(kind, i, T, k, true_t):
        last_hb[i] = max(last_hb[i], T)
        if kind == "msg":
            est = T - mu[i]
            heapq.heappush(pending, (est + z * sig[i], est, i, k, true_t))
        # any future message from client j has true time >= last_hb[j] - mu_j - z*sig_j (w.p. p_safe)
        frontier = min(last_hb[j] - mu[j] - z * sig[j] for j in range(n))
        ready = []
        while pending and pending[0][0] < frontier:
            ready.append(heapq.heappop(pending))
        for _, est, ii, kk, tt in sorted(ready, key=lambda r: r[1]):
            prev = state["prev"]
            if prev is not None:
                p = norm.cdf((est - prev[0]) / np.hypot(sig[prev[1]], sig[ii]))
                if p > p_thresh:
                    state["batch"] += 1
            state["prev"] = (est, ii)
            out.append((state["batch"], est, ii, kk, tt, sim.now))

    sim.at(0.0, trigger, 0)
    for i in range(n):
        sim.at(0.0, heartbeat, i)
    sim.run(points * G + 3000)

    # score races: same trigger point, all client pairs
    by_pt = {}
    for b, est, i, k, tt, et in out:
        by_pt.setdefault(k, {})[i] = (b, est, tt, et)
    c = w = t = 0
    cn = wn = 0
    lat = []
    for k, d in by_pt.items():
        if len(d) < n:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                f, s = (i, j) if rts[i] < rts[j] else (j, i)
                bf, bs = d[f][0], d[s][0]
                c += bf < bs; w += bf > bs; t += bf == bs
                cn += d[f][1] < d[s][1]; wn += d[f][1] > d[s][1]
        lat += [v[3] - v[2] for v in d.values()]
    tot = c + w + t
    return {"sigma_us": sigma, "p_thresh": p_thresh,
            "prob_correct": c / tot, "prob_wrong": w / tot, "prob_tie": t / tot,
            "timestamp_correct": cn / tot, "timestamp_wrong": wn / tot,
            "median_emit_latency_us": float(np.median(lat)), "races": tot}


def intransitivity():
    """Clock-error PMFs shaped like Efron-style non-transitive dice (values in 100 ns).
    All three messages carry the SAME timestamp; true time = stamp - error, so
    P(X truly before Y) = P(err_X > err_Y), computed via FFT convolution of
    pmf_X with the reversed pmf_Y (distribution of err_X - err_Y)."""
    dice = {"A": [2, 4, 9], "B": [1, 6, 8], "C": [3, 5, 7]}
    L = 16

    def pmf(vals):
        p = np.zeros(L); p[vals] = 1 / len(vals); return p

    def p_before(x, y):
        conv = np.real(np.fft.ifft(np.fft.fft(pmf(dice[x]), 2 * L) * np.fft.fft(pmf(dice[y])[::-1], 2 * L)))
        diff = np.arange(2 * L) - (L - 1)            # index -> err_x - err_y
        return float(np.sum(conv[diff > 0])), float(np.sum(conv[diff == 0]))

    res = {f"P({x} before {y})": round(p_before(x, y)[0], 4) for x, y in [("A", "B"), ("B", "C"), ("C", "A")]}
    # Gaussian sanity check: random means/sigmas never produce a cycle
    rng = np.random.default_rng(0)
    cycles = 0
    for _ in range(20000):
        m, s = rng.normal(0, 1, 3), rng.uniform(0.2, 2, 3)
        P = lambda a, b: norm.cdf((m[a] - m[b]) / np.hypot(s[a], s[b]))  # P(a truly before b) when err_a > err_b
        cycles += any(P(a, b) > .5 and P(b, c) > .5 and P(c, a) > .5 for a, b, c in permutations(range(3)))
    res["gaussian_trials"] = 20000
    res["gaussian_cycles_found"] = cycles
    return res


if __name__ == "__main__":
    res = {"online": [], "threshold_sweep": [], "intransitivity": intransitivity()}
    print(res["intransitivity"], flush=True)
    for s in (0.1, 0.5, 1.0, 2.0, 5.0):
        r = online(sigma=s); res["online"].append(r); print(r, flush=True)
    for pt in (0.6, 0.9, 0.99):
        r = online(sigma=1.0, p_thresh=pt); res["threshold_sweep"].append(r); print(r, flush=True)
    (OUT / "poc_tommy.json").write_text(json.dumps(res, indent=1))
