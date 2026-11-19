"""Check our vectorised port against the authors' original (pure-Python) classes on the
same inputs. Run from experiments/:  .venv/bin/python -I fairlab/validate.py"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dbo-simulation"))

from azure_trace import load_latency_trace  # noqa: E402
import sim  # noqa: E402
from algorithms.dbo import DBO  # noqa: E402
from algorithms.cloudex import Cloudex  # noqa: E402
from algorithms.direct import DirectDelivery  # noqa: E402
from algorithms.max_rtt import MaxRTT  # noqa: E402

T, N = 60_000, 10
trace = load_latency_trace()
env = sim.make_env(trace, n=N, T=T)
g_time = list(range(T))
fw = [list(env.fw[i]) for i in range(N)]
rv = [list(env.rv[i]) for i in range(N)]
rts = list(env.rt)

cases = [
    ("Direct", DirectDelivery(), lambda: sim.direct(env)),
    ("CloudEx(15|15)", Cloudex(15, 15), lambda: sim.cloudex(env, 15, 15)),
    ("DBO(20|25)", DBO(20, 25, 0), lambda: sim.dbo(env, 20, 25)),
    ("MaxRTT", MaxRTT(), lambda: sim.max_rtt(env)),
]
ok = True
for name, orig, ours in cases:
    t0 = time.time()
    orig.set_simulation_environment(g_time, T, N, fw, rv, rts, 1)
    orig.run_simulation()
    t1 = time.time()
    r = ours()
    t2 = time.time()
    o_ord = np.array(orig.ordering_arr)
    o_lat = np.array(orig.latency_arr)
    d_ord = np.max(np.abs(o_ord - r.ordering))
    d_lat = np.max(np.abs(o_lat - r.latency))
    f_o, f_v = orig.get_win_fraction(), sim.fairness(env, r)
    good = d_ord < 1e-6 and d_lat < 1e-6 and abs(f_o - f_v) < 1e-12
    ok &= good
    print(f"{name:16s} max|dOrder|={d_ord:.2e} max|dLat|={d_lat:.2e} "
          f"fair orig={f_o:.6f} ours={f_v:.6f} | mean lat orig={orig.get_mean_latency():.3f} "
          f"ours={sim.latency_stats(r)['mean']:.3f} | {t1-t0:.1f}s vs {t2-t1:.2f}s | {'OK' if good else 'MISMATCH'}",
          flush=True)
print("ALL MATCH" if ok else "SOME MISMATCH")
