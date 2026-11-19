"""Tiny discrete-event engine for the protocol proofs-of-concept.

Every PoC is message-level: exchange (CES), gateways / release buffers, proxies and
sequencers are separate nodes that exchange timestamped messages over links whose
one-way delays are replayed from the DBO authors' real Azure trace.
Units: microseconds.
"""
import heapq, itertools, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fairlab"))
from azure_trace import load_latency_trace  # noqa: E402
from sim import random_window  # noqa: E402
sys.path.insert(0, str(ROOT / "dbo-simulation"))
from traces.trace_indices import rand_idx1  # noqa: E402

TRACE = load_latency_trace()


class Sim:
    def __init__(self):
        self.now = 0.0
        self._q = []
        self._seq = itertools.count()

    def at(self, t, fn, *args):
        heapq.heappush(self._q, (max(t, self.now), next(self._seq), fn, args))

    def after(self, dt, fn, *args):
        self.at(self.now + dt, fn, *args)

    def run(self, until):
        while self._q and self._q[0][0] <= until:
            t, _, fn, args = heapq.heappop(self._q)
            self.now = t
            fn(*args)


class Link:
    """One-way link whose delay at real time t is half the Azure RTT at t (as in the
    authors' simulator), optionally plus a fixed extra delay. `k` picks the window."""
    _next = 0

    def __init__(self, extra=0.0, horizon=2_000_000, k=None):
        k = Link._next if k is None else k
        Link._next += 1
        self.owd = random_window(TRACE, rand_idx1[k % len(rand_idx1)], horizon) / 2.0
        self.extra = extra

    def delay(self, t):
        return self.owd[int(t) % len(self.owd)] + self.extra


class Sender:
    """Serialises messages out of one NIC: each send occupies the NIC for `ser` us.
    This is what makes plain unicast to many receivers slow (Onyx's argument)."""
    def __init__(self, sim, ser=0.0):
        self.sim, self.ser, self.free = sim, ser, 0.0

    def send(self, link, fn, *args):
        start = max(self.sim.now, self.free)
        self.free = start + self.ser
        depart = self.free
        self.sim.at(depart + link.delay(depart), fn, *args)


def response_times(n):
    return [4.0 + (n - i - 1) * (15.0 / n) for i in range(n)]


def pair_fairness(exec_rank, rts, points):
    """exec_rank[(trader, point)] = position in the matching engine's execution sequence.
    Returns the fraction of (pair, point) races in which the faster trader executed first,
    counting only races where both trades executed."""
    n = len(rts)
    good = tot = 0
    for p in points:
        for i in range(n):
            for j in range(i + 1, n):
                a, b = exec_rank.get((i, p)), exec_rank.get((j, p))
                if a is None or b is None:
                    continue
                fast, slow = (a, b) if rts[i] < rts[j] else (b, a)
                good += fast < slow
                tot += 1
    return good / tot if tot else float("nan")
