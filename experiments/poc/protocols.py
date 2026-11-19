"""Message-level proofs of concept of CloudEx, Onyx and DBO.

Each function builds the paper's components as event-driven nodes, replays real Azure
link delays (poc.core.Link) and returns metrics. These are PoCs of the *mechanisms*,
not reimplementations of the authors' systems (no DPDK, no real NICs, no matching engine
beyond recording execution order).
"""
import heapq, math
from collections import defaultdict
import numpy as np
from core import Sim, Link, Sender, response_times, pair_fairness

INF = float("inf")


class FifoLink:
    """TCP-like ordered channel over a trace-driven Link (no overtaking)."""
    def __init__(self, sim, link):
        self.sim, self.link, self.last = sim, link, 0.0

    def send(self, fn, *args, depart=None):
        t = self.sim.now if depart is None else depart
        arr = max(t + self.link.delay(t), self.last)
        self.last = arr
        self.sim.at(arr, fn, *args)


class Recorder:
    def __init__(self):
        self.rank = {}           # (trader, point) -> execution position
        self.exec_t = {}         # (trader, point) -> execution time
        self.n = 0

    def execute(self, trader, point, now):
        self.rank[(trader, point)] = self.n
        self.exec_t[(trader, point)] = now
        self.n += 1


def _latency(rec, gen, rts):
    lat = [rec.exec_t[k] - gen[k[1]] - rts[k[0]] for k in rec.exec_t]
    return float(np.mean(lat)), float(np.percentile(lat, 99))


def _throughput_timeline(rec, horizon, bin_us=1000):
    t = np.array(sorted(rec.exec_t.values()))
    edges = np.arange(0, horizon + bin_us, bin_us)
    return np.histogram(t, edges)[0].tolist()


# ===================================================================== CloudEx
def cloudex(n=10, points=4000, G=10.0, d_h=15.0, d_s=15.0, sigma=0.0, seed=0,
            fail=None, fail_at=INF):
    """CloudEx (HotOS'21): gateway hold/release at t_M + d_h on the gateway's synced
    clock; gateway timestamps orders; sequencer releases an order once CES clock >=
    stamp + d_s (timeout). `fail`: a gateway that dies at `fail_at`."""
    sim, rec = Sim(), Recorder()
    rng = np.random.default_rng(seed)
    th = rng.normal(0, sigma, n) if sigma > 0 else np.zeros(n)
    rts = response_times(n)
    links = [Link(k=i) for i in range(n)]
    gen, late, release_t = {}, 0, defaultdict(dict)
    q = []

    def alive(i):
        return not (fail == i and sim.now >= fail_at)

    def tick(k):
        gen[k] = sim.now
        for i in range(n):
            sim.at(sim.now + links[i].delay(sim.now), gw_md, i, k)
        if k + 1 < points:
            sim.after(G, tick, k + 1)

    def gw_md(i, k):
        nonlocal late
        if not alive(i):
            return
        due = gen[k] + d_h - th[i]           # local clock reads t_M + d_h
        late += sim.now > due
        sim.at(due, release, i, k)

    def release(i, k):
        release_t[k][i] = sim.now
        sim.after(rts[i], trade, i, k)

    def trade(i, k):
        if not alive(i):
            return
        stamp = sim.now + th[i]
        sim.at(sim.now + links[i].delay(sim.now), ces_order, stamp, i, k)

    def ces_order(stamp, i, k):
        heapq.heappush(q, (stamp, i, k))
        sim.at(stamp + d_s, sequencer)

    def sequencer():
        while q and q[0][0] + d_s <= sim.now:
            _, i, k = heapq.heappop(q)
            rec.execute(i, k, sim.now)

    sim.at(0.0, tick, 0)
    horizon = points * G + 2000
    sim.run(horizon)
    mean, p99 = _latency(rec, gen, rts)
    dws = [max(v.values()) - min(v.values()) for v in release_t.values() if len(v) == n]
    return {"protocol": f"CloudEx(d_h={d_h},d_s={d_s},sigma={sigma})",
            "fairness": pair_fairness(rec.rank, rts, range(points)),
            "mean_latency": mean, "p99_latency": p99,
            "late_deliveries_pct": 100 * late / (n * points),
            "p_dws_le_1us": 100 * float(np.mean(np.array(dws) <= 1.0)) if dws else None,
            "executed": rec.n, "timeline": _throughput_timeline(rec, horizon)}


# ===================================================================== Onyx
def onyx(n=40, points=3000, G=10.0, F=6, H=1, spray=True, mode="tree", ser=0.5,
         sigma=0.1, hb=5.0, seed=0, fail=None, fail_at=INF, window=200):
    """Onyx (SIGCOMM'25). mode='tree': CES -> P proxies -> leaf gateways (fanout F) with
    round-robin packet spraying and proxy hedging (H sibling copies; first copy wins);
    mode='unicast': CES sends directly to every leaf (serialised NIC).
    Outbound: hold-and-release to deadline t_M + D, D = max over leaves of the p95 of
    recently measured delivery delays (feedback). Inbound: orders travel back up the tree;
    the CES sequencer is wait-for-one (release smallest stamp only after every live gateway
    has sent something with a later stamp; gateways send heartbeats every `hb` us)."""
    sim, rec = Sim(), Recorder()
    rng = np.random.default_rng(seed)
    th = rng.normal(0, sigma, n) if sigma > 0 else np.zeros(n)
    rts = response_times(n)
    P = math.ceil(n / F) if mode == "tree" else 0
    ces = Sender(sim, ser)
    prox_tx = [Sender(sim, ser) for _ in range(P)]
    k_link = iter(range(10_000))
    ces_prox = [Link(k=next(k_link)) for _ in range(P)]
    prox_leaf = [[Link(k=next(k_link)) for _ in range(n)] for _ in range(P)]
    ces_leaf = [Link(k=next(k_link)) for _ in range(n)]
    home = [j // F for j in range(n)] if P else [None] * n
    up_leaf = [FifoLink(sim, prox_leaf[home[j]][j] if P else ces_leaf[j]) for j in range(n)]
    up_prox = [FifoLink(sim, ces_prox[p]) for p in range(P)]
    gen, first_arr, release_t = {}, defaultdict(dict), defaultdict(dict)
    recent = defaultdict(list)
    D = {"est": 100.0}
    late = 0
    last_stamp = [-INF] * n
    q = []

    def alive(i):
        return not (fail == i and sim.now >= fail_at)

    def groups(p, k):
        base = (p + k) % P if spray else p
        return [(base + h) % P for h in range(H + 1)]

    def tick(k):
        gen[k] = sim.now
        deadline = sim.now + D["est"]
        if mode == "tree":
            for p in range(P):
                ces.send(ces_prox[p], proxy_md, p, k, deadline)
        else:
            for j in range(n):
                ces.send(ces_leaf[j], leaf_md, j, k, deadline)
        if k % window == window - 1:   # feedback: refresh the deadline estimate
            est = [np.percentile(v[-window:], 95) for v in recent.values() if v]
            if est:
                D["est"] = max(est)
        if k + 1 < points:
            sim.after(G, tick, k + 1)

    def proxy_md(p, k, deadline):
        for g in groups(p, k):
            for j in range(g * F, min((g + 1) * F, n)):
                prox_tx[p].send(prox_leaf[p][j], leaf_md, j, k, deadline)

    def leaf_md(j, k, deadline):
        nonlocal late
        if k in first_arr[j] or not alive(j):
            return                                  # duplicate (hedging) or dead
        first_arr[j][k] = sim.now
        recent[j].append(sim.now - gen[k])
        due = deadline - th[j]
        late += sim.now > due
        sim.at(due, release, j, k)

    def release(j, k):
        release_t[k][j] = sim.now
        sim.after(rts[j], trade, j, k)

    def trade(j, k):
        if alive(j):
            send_up(("order", sim.now + th[j], j, k))

    def heartbeat(j):
        if alive(j):
            send_up(("hb", sim.now + th[j], j, None))
            sim.after(hb, heartbeat, j)

    def send_up(msg):
        j = msg[2]
        if P:
            up_leaf[j].send(proxy_up, home[j], msg)
        else:
            up_leaf[j].send(ces_in, msg)

    def proxy_up(p, msg):
        up_prox[p].send(ces_in, msg)

    def ces_in(msg):
        kind, stamp, j, k = msg
        last_stamp[j] = max(last_stamp[j], stamp)
        if kind == "order":
            heapq.heappush(q, (stamp, j, k))
        horizon_ok = min(last_stamp)               # wait-for-one: needs EVERY gateway
        while q and q[0][0] < horizon_ok:
            _, jj, kk = heapq.heappop(q)
            rec.execute(jj, kk, sim.now)

    sim.at(0.0, tick, 0)
    for j in range(n):
        sim.at(0.0, heartbeat, j)
    horizon = points * G + 3000
    sim.run(horizon)
    oml = [max(v.values()) - gen[k] for k, v in release_t.items() if len(v) == n]
    dws = [max(v.values()) - min(v.values()) for k, v in release_t.items() if len(v) == n]
    arr = [max(v.values()) - gen[k] for k, v in first_arr_by_point(first_arr).items() if len(v) == n]
    mean, p99 = _latency(rec, gen, rts) if rec.n else (None, None)
    return {"protocol": f"Onyx(mode={mode},F={F},H={H},spray={spray})",
            "fairness": pair_fairness(rec.rank, rts, range(points)),
            "median_last_arrival_us": float(np.median(arr)) if arr else None,
            "median_OML_us": float(np.median(oml)) if oml else None,
            "p_dws_le_1us": 100 * float(np.mean(np.array(dws) <= 1.0)) if dws else None,
            "late_deliveries_pct": 100 * late / (n * points),
            "mean_latency": mean, "p99_latency": p99, "executed": rec.n,
            "timeline": _throughput_timeline(rec, horizon)}


def first_arr_by_point(first_arr):
    out = defaultdict(dict)
    for j, d in first_arr.items():
        for k, t in d.items():
            out[k][j] = t
    return out


# ===================================================================== DBO
def dbo(n=10, points=8000, G=5.0, delta=20.0, batch=25.0, tau=5.0, extra=None,
        straggler_timeout=None, fail=None, fail_at=INF):
    """DBO (SIGCOMM'23). Release buffer (RB) per trader: batches market data into
    `batch`-us batches, delivers a batch when complete and >= delta after the previous
    one, stamps each trade with the delivery clock DC = (batch id, time since delivery),
    and sends heartbeats carrying its current DC every `tau` us. Ordering buffer (OB):
    priority queue by DC; forwards a trade once every (non-straggler) RB has reported a
    larger DC. `straggler_timeout`: OB stops waiting for an RB whose heartbeats are
    older than this (DBO §4.2.1). No clock synchronisation anywhere."""
    sim, rec = Sim(), Recorder()
    rts = response_times(n)
    extra = extra if extra is not None else [0.0] * n
    links = [Link(extra=extra[i], k=i) for i in range(n)]
    down = [FifoLink(sim, links[i]) for i in range(n)]
    up = [FifoLink(sim, links[i]) for i in range(n)]
    gen = {}
    per_batch = int(round(batch / G))
    rb_pending = [defaultdict(list) for _ in range(n)]
    rb_last = [(-1, -INF)] * n          # (batch id, delivery time) of last delivered batch
    rb_last_deliv = [-INF] * n
    last_dc = [(-1, -INF)] * n          # latest DC heard by the OB from each RB
    last_heard = [0.0] * n
    q = []

    def alive(i):
        return not (fail == i and sim.now >= fail_at)

    def tick(k):
        gen[k] = sim.now
        for i in range(n):
            down[i].send(rb_md, i, k)
        if k + 1 < points:
            sim.after(G, tick, k + 1)

    def rb_md(i, k):
        if not alive(i):
            return
        b = k // per_batch
        rb_pending[i][b].append(k)
        if len(rb_pending[i][b]) == per_batch or k == points - 1:
            sim.at(max(sim.now, rb_last_deliv[i] + delta), rb_deliver, i, b)
            rb_last_deliv[i] = max(sim.now, rb_last_deliv[i] + delta)

    def rb_deliver(i, b):
        rb_last[i] = (b, sim.now)
        for k in rb_pending[i].pop(b):
            sim.after(rts[i], rb_trade, i, k)

    def dc_now(i):
        b, t = rb_last[i]
        return (b, sim.now - t)

    def rb_trade(i, k):
        if alive(i):
            up[i].send(ob_in, i, "order", dc_now(i), k)

    def rb_heartbeat(i):
        if alive(i):
            up[i].send(ob_in, i, "hb", dc_now(i), None)
            sim.after(tau, rb_heartbeat, i)

    rtt_est = [0.0] * n

    def ob_in(i, kind, dc, k):
        last_dc[i] = max(last_dc[i], dc)
        last_heard[i] = sim.now
        b, elapsed = dc
        if b >= 0:  # round trip seen via this message: now - (when batch b's data was generated) - elapsed
            last_pt = min((b + 1) * per_batch - 1, points - 1)
            rtt_est[i] = sim.now - gen[last_pt] - elapsed
        if kind == "order":
            heapq.heappush(q, (dc, i, k))
        forward()

    def waiting_for():
        """DBO §4.2.1: stop waiting for an RB whose measured round trip exceeds the
        threshold (or that has gone silent for longer than it)."""
        if straggler_timeout is None:
            return range(n)
        return [i for i in range(n) if sim.now - last_heard[i] <= straggler_timeout
                and rtt_est[i] <= straggler_timeout]

    def forward():
        w = list(waiting_for())
        if not w:
            return
        frontier = min(last_dc[i] for i in w)
        while q and q[0][0] < frontier:
            dc, i, k = heapq.heappop(q)
            rec.execute(i, k, sim.now)

    def ob_watchdog():
        forward()
        sim.after(50.0, ob_watchdog)

    sim.at(0.0, tick, 0)
    for i in range(n):
        sim.at(0.0, rb_heartbeat, i)
    if straggler_timeout is not None:
        sim.at(0.0, ob_watchdog)
    horizon = points * G + 3000 + 2 * max(extra)
    sim.run(horizon)
    mean, p99 = _latency(rec, gen, rts) if rec.n else (None, None)
    others = [i for i in range(n) if extra[i] == 0]
    lat_others = [rec.exec_t[k] - gen[k[1]] - rts[k[0]] for k in rec.exec_t if k[0] in others]
    return {"protocol": f"DBO(delta={delta},batch={batch},tau={tau})",
            "fairness": pair_fairness(rec.rank, rts, range(points)),
            "mean_latency": mean, "p99_latency": p99,
            "others_mean_latency": float(np.mean(lat_others)) if lat_others else None,
            "executed": rec.n, "timeline": _throughput_timeline(rec, horizon)}
