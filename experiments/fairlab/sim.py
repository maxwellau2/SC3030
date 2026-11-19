"""Vectorised port of the DBO authors' simulator (github.com/eash3010/dbo-simulation),
plus our extensions. Units: microseconds. One market data point per microsecond.

Faithful ports (validated against the original classes in validate.py):
    direct, cloudex (perfect clocks), dbo, max_rtt
Our extensions (NOT in the original code):
    cloudex(..., offsets=theta)  per-gateway clock-sync error
    tommy_*                      probabilistic ordering (Beyond Lamport, HotNets'25)
    extra one-way delay per participant (straggler / "fairness radius")
    compute jitter on response times (Cuttlefish's argument)
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import norm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbo-simulation"))
from traces.trace_indices import rand_idx1  # noqa: E402  (authors' fixed random offsets)

MIN_RT, MAX_RT = 4.0, 19.0  # authors' single_run.py


def response_times(n):
    """Authors' choice: evenly spaced, MP0 slowest ... MP(n-1) fastest. Gap = 15/n us."""
    return np.array([MIN_RT + (n - i - 1) * ((MAX_RT - MIN_RT) / n) for i in range(n)])


def random_window(trace, start, length):
    """util.generate_random_trace (wrap-around window)."""
    if start + length < trace.shape[0]:
        return trace[start:start + length]
    return np.concatenate([trace[start:], trace[:length - (trace.shape[0] - start)]])


@dataclass
class Env:
    T: int                      # number of data points (= horizon in us)
    n: int
    rt: np.ndarray              # nominal response times (n,)
    fw: np.ndarray              # (n, 2T) one-way delay CES->RB, indexed by int(time)
    rv: np.ndarray              # (n, 2T) one-way delay RB->CES
    rt_actual: np.ndarray = None  # (n, T) per-trade response time incl. compute jitter (default = nominal)

    def __post_init__(self):
        if self.rt_actual is None:
            self.rt_actual = np.repeat(self.rt[:, None], self.T, axis=1)


def make_env(trace, n=10, T=1_000_000, extra_owd=None, rt=None, rt_actual=None):
    rt = response_times(n) if rt is None else np.asarray(rt, float)
    fw = np.empty((n, 2 * T))
    for i in range(n):
        w = random_window(trace, rand_idx1[i], 2 * T)
        fw[i] = w / 2.0
    rv = fw.copy()
    if extra_owd is not None:
        fw += np.asarray(extra_owd, float)[:, None]
        rv += np.asarray(extra_owd, float)[:, None]
    return Env(T=T, n=n, rt=rt, fw=fw, rv=rv, rt_actual=rt_actual)


def _lookup(arr_row, t):
    """arr[int(t)] as the original code does."""
    return arr_row[t.astype(np.int64)]


@dataclass
class Result:
    name: str
    ordering: np.ndarray        # (n, T) values on a total-order line (smaller = earlier)
    latency: np.ndarray         # (n, T-26)
    extra: dict = field(default_factory=dict)


def _latency(env, exec_time, i):
    g = np.arange(env.T, dtype=np.float64)
    lat = exec_time - g - env.rt[i]
    return lat[: -int(25.0 + 1)]


# ---------------------------------------------------------------- algorithms
def direct(env):
    g = np.arange(env.T, dtype=np.float64)
    order, lat = [], []
    for i in range(env.n):
        r = g + env.fw[i, : env.T]
        s = r + env.rt_actual[i]
        recv = s + _lookup(env.rv[i], s)
        order.append(recv)
        lat.append(_latency(env, recv, i))
    return Result("Direct", np.array(order), np.array(lat))


def cloudex(env, d_o, d_i, offsets=None):
    """CloudEx hold/release + timestamp sequencer.
    offsets[i] = clock error of gateway/RB i (local = true + offset). The OB/CES clock is
    the reference. offsets=None reproduces the authors' perfect-clock model exactly.
    Release: gateway releases when its LOCAL clock reads g + d_o  -> true time g + d_o - theta.
    Stamp:   trade stamped with LOCAL time s + theta.
    OB:      processes in stamp order once its clock passes stamp + d_i; a trade arriving
             later than that is processed on arrival (as in the original model)."""
    th = np.zeros(env.n) if offsets is None else np.asarray(offsets, float)
    g = np.arange(env.T, dtype=np.float64)
    order, lat, stamps = [], [], []
    for i in range(env.n):
        r = g + env.fw[i, : env.T]
        d = np.maximum(g + d_o - th[i], r)
        s = d + env.rt_actual[i]
        recv = s + _lookup(env.rv[i], s)
        stamp = s + th[i]
        o = np.maximum(stamp + d_i, recv)
        order.append(o)
        stamps.append(stamp)
        lat.append(_latency(env, o, i))
    name = f"CloudEx({d_o:g}|{d_i:g})"
    return Result(name, np.array(order), np.array(lat), {"stamps": np.array(stamps)})


def dbo_delivery(env, i, delta, batch):
    """Vectorised DBO RB pacing: D_b = max(R_b, D_{b-1} + delta) with R_b the arrival
    time of the batch's last point; closed form via a running maximum."""
    g = np.arange(env.T, dtype=np.float64)
    r = g + env.fw[i, : env.T]
    b = (g // batch).astype(np.int64)
    nb = b[-1] + 1
    last_idx = np.minimum((np.arange(nb) + 1) * batch - 1, env.T - 1)
    R = r[last_idx]
    j = np.arange(nb, dtype=np.float64)
    D = j * delta + np.maximum.accumulate(np.maximum(R - j * delta, -100.0 + delta))
    return D[b]


def dbo(env, delta=20, batch=25, exclude=()):
    """exclude: participants the OB stops waiting for (DBO's straggler mitigation, §4.2.1).
    With exclude=() this is the authors' algorithm exactly. With exclusions (our extension),
    a trade is executed at max(its arrival at the OB, the heartbeat time of the waited-for
    participants); 'exec' and 'dc' are returned so fairness can be judged by execution order."""
    T = env.T
    d_all, ord_all, ack_all, recv_all = [], [], [], []
    for i in range(env.n):
        d = dbo_delivery(env, i, delta, batch)
        s = d + env.rt_actual[i]
        ind = np.searchsorted(d, s, side="right")
        ind = np.minimum(ind, T - 1)
        o = (ind - 1) * float(T) + (s - d[ind - 1])
        ack = d + _lookup(env.rv[i], d)
        recv_all.append(s + _lookup(env.rv[i], s))
        d_all.append(d); ord_all.append(o); ack_all.append(ack)
    keep = [i for i in range(env.n) if i not in set(exclude)]
    A = np.max(np.array(ack_all)[keep], axis=0)
    lat, ex_all = [], []
    for i in range(env.n):
        x = (ord_all[i] / T).astype(np.int64)
        ex = A[np.minimum(x + 1, T - 1)]
        if exclude:
            ex = np.maximum(ex, recv_all[i])
        ex_all.append(ex)
        lat.append(_latency(env, ex, i))
    name = f"DBO({delta}|{batch})" + (f" excl{list(exclude)}" if exclude else "")
    return Result(name, np.array(ord_all), np.array(lat),
                  {"exec": np.array(ex_all), "dc": np.array(ord_all)})


def fairness_by_exec(env, res, pairs):
    """Fraction of data points where the faster participant's trade executes first
    (execution time, ties broken by delivery clock). Used with DBO straggler exclusion."""
    ex, dc = res.extra["exec"], res.extra["dc"]
    vals = []
    for i, j in pairs:
        f, s = (i, j) if env.rt[i] < env.rt[j] else (j, i)
        first = (ex[f] < ex[s]) | ((ex[f] == ex[s]) & (dc[f] < dc[s]))
        vals.append(np.mean(first))
    return float(np.mean(vals))


def max_rtt(env):
    T = env.T
    g = np.arange(T, dtype=np.float64)
    d_all, ack_all, ord_all = [], [], []
    for i in range(env.n):
        d = g + env.fw[i, :T]
        s = d + env.rt_actual[i]
        ord_all.append(g * T + (s - d))
        d_all.append(d)
        ack_all.append(d + _lookup(env.rv[i], d))
    lat = []
    for i in range(env.n):
        rt_i = ord_all[i] - g * T
        ex = np.full(T, -1.0)
        for j in range(env.n):
            target = d_all[j] + rt_i
            ind = np.minimum(np.searchsorted(d_all[j], target, side="left"), T - 1)
            ind = np.maximum(ind, np.arange(T))
            ex = np.maximum(ex, ack_all[j][ind])
        lat.append(_latency(env, ex, i))
    return Result("MaxRTT", np.array(ord_all), np.array(lat))


def cloudex_external(env, d_i, offsets):
    """Our extension: an EXTERNAL trigger (e.g. news, another venue's price) reaches every
    trader at the same true time g -- it is NOT released by the gateways, so a gateway's
    clock offset no longer cancels between release and stamp. Trades are stamped by the
    gateway clock (true + theta) and sequenced as in CloudEx."""
    th = np.asarray(offsets, float)
    g = np.arange(env.T, dtype=np.float64)
    order, stamps = [], []
    for i in range(env.n):
        s = g + env.rt_actual[i]
        recv = s + _lookup(env.rv[i], s)
        stamp = s + th[i]
        order.append(np.maximum(stamp + d_i, recv))
        stamps.append(stamp)
    return Result(f"CloudEx-external(|{d_i:g})", np.array(order), np.zeros((env.n, 1)),
                  {"stamps": np.array(stamps)})


# ---------------------------------------------------------------- metrics
def fairness(env, res, pairs=None):
    """Authors' get_win_fraction: over all participant pairs, the fraction of data points
    where the trade of the (nominally) faster participant is ordered strictly first."""
    o = res.ordering
    vals = []
    idx = pairs if pairs is not None else [(i, j) for i in range(env.n) for j in range(i + 1, env.n)]
    for i, j in idx:
        fast, slow = (i, j) if env.rt[i] < env.rt[j] else (j, i)
        vals.append(np.mean(o[fast] < o[slow]))
    return float(np.mean(vals))


def latency_stats(res):
    L = res.latency
    return {"mean": float(np.mean(L)), "p99": float(np.percentile(L.flatten(), 99)),
            "max": float(np.max(L))}


def tommy_scores(env, stamps, sigma, p_thresh=0.75):
    """Beyond Lamport-style probabilistic ordering applied to CloudEx timestamps.
    For each pair and data point: P(fast really first) = Phi((stamp_slow - stamp_fast)/(sigma*sqrt2))
    using the known offset distribution N(0, sigma). Ordered if max(P,1-P) > p_thresh, else tie.
    Returns shares of correct / wrong / tie and the rank-agreement score (correct - wrong)."""
    c = w = t = 0.0
    m = 0
    for i in range(env.n):
        for j in range(i + 1, env.n):
            fast, slow = (i, j) if env.rt[i] < env.rt[j] else (j, i)
            gap = stamps[slow] - stamps[fast]
            if sigma > 0:
                p = norm.cdf(gap / (sigma * np.sqrt(2.0)))
            else:
                p = (gap > 0).astype(float) + 0.5 * (gap == 0)
            tie = np.maximum(p, 1 - p) <= p_thresh
            c += np.mean((~tie) & (p > 0.5)); w += np.mean((~tie) & (p < 0.5)); t += np.mean(tie)
            m += 1
    c, w, t = c / m, w / m, t / m
    return {"correct": c, "wrong": w, "tie": t, "RAS": c - w}
