"""Fast loader for the DBO authors' Azure latency trace.

Reproduces exactly the 2M-microsecond RTT slice that the authors' `single_run.py` /
`run_simulation.py` use (MP1's RTTs, interpolated to 1 us steps, around index
53,927,275), but with vectorised numpy instead of a pure-Python loop over the
whole 15-minute trace.

Authors' definitions (util.read_cloud_trace):
    e2e      = execution_time - generation_time - response_time
    ob_delay = execution_time - ces_recv_time
    rtt      = e2e - pacing_delay - ob_delay
             = ces_recv_time - generation_time - response_time - pacing_delay
Interpolation (util.data_generation): rtt_arr[k] is the linear interpolation of
MP1's (generation_time - first_generation_time, rtt) samples at time step k+1.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRACE_ZIP = ROOT / "dbo-simulation" / "traces" / "direct.zip"
CACHE = ROOT / "cache"

TRACE_DD_IDX = 53927275  # fixed by the authors "to reproduce the results"


def load_mp1_samples():
    cache = CACHE / "mp1_samples.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"]
    cols = ["generation_time", "mp_id", "response_time", "pacing_delay", "ces_recv_time"]
    df = pd.read_csv(TRACE_ZIP, usecols=cols)
    df = df[df["mp_id"] == 1].sort_values(by="generation_time", kind="stable")
    rtt = (df["ces_recv_time"] - df["generation_time"] - df["response_time"] - df["pacing_delay"]).to_numpy()
    x = df["generation_time"].to_numpy()
    x = x - x[0]
    CACHE.mkdir(exist_ok=True)
    np.savez(cache, x=x, y=rtt)
    return x, rtt


def load_latency_trace():
    """Return the 2,000,000-step RTT slice (us) used by the authors."""
    cache = CACHE / "latency_trace.npy"
    if cache.exists():
        return np.load(cache)
    x, y = load_mp1_samples()
    lo, hi = TRACE_DD_IDX - 150000, TRACE_DD_IDX + 1850000
    steps = np.arange(lo, hi, dtype=np.float64) + 1.0  # rtt_arr[k] <-> time step k+1
    trace = np.interp(steps, x, y)
    np.save(cache, trace)
    return trace


if __name__ == "__main__":
    t = load_latency_trace()
    print(len(t), "steps; RTT us: min %.1f median %.1f p99 %.1f max %.1f"
          % (t.min(), np.median(t), np.percentile(t, 99), t.max()))
