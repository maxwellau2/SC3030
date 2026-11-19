"""Run the message-level PoCs. From experiments/:  .venv/bin/python -I poc/run_poc.py [cloudex onyx dbo]"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocols as P  # noqa: E402

OUT = HERE.parent / "results"
OUT.mkdir(exist_ok=True)


def show(r):
    keys = [k for k in r if k not in ("timeline",)]
    print("  " + " | ".join(f"{k}={r[k]:.4g}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys), flush=True)
    return r


def run_cloudex():
    res = {}
    res["base_hold15"] = show(P.cloudex(d_h=15, d_s=15))
    res["hold30"] = show(P.cloudex(d_h=30, d_s=30))
    res["hold15_sigma1us"] = show(P.cloudex(d_h=15, d_s=15, sigma=1.0))
    res["gateway_fail"] = show(P.cloudex(d_h=15, d_s=15, fail=3, fail_at=20_000))
    return res


def run_onyx():
    res = {}
    kw = dict(n=40, points=2000, ser=0.2)  # 0.2 us/send: unicast to 40 = 8 us < 10 us interval (sustainable)
    res["unicast"] = show(P.onyx(mode="unicast", **kw))
    res["tree"] = show(P.onyx(mode="tree", spray=False, H=0, **kw))
    res["tree_spray"] = show(P.onyx(mode="tree", spray=True, H=0, **kw))
    res["tree_spray_hedge"] = show(P.onyx(mode="tree", spray=True, H=1, **kw))
    res["gateway_fail"] = show(P.onyx(mode="tree", spray=True, H=1, fail=7, fail_at=10_000, **kw))
    return res


def run_dbo():
    res = {}
    res["base"] = show(P.dbo())
    res["rb_fail_no_timeout"] = show(P.dbo(fail=3, fail_at=20_000))
    res["rb_fail_timeout200"] = show(P.dbo(fail=3, fail_at=20_000, straggler_timeout=200.0))
    far = [0.0] * 10; far[5] = 500.0
    res["far_trader_wait"] = show(P.dbo(extra=far))
    res["far_trader_timeout"] = show(P.dbo(extra=far, straggler_timeout=200.0))
    return res


if __name__ == "__main__":
    which = sys.argv[1:] or ["cloudex", "onyx", "dbo"]
    path = OUT / "poc_results.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    for w in which:
        t0 = time.time(); print(f"== {w}", flush=True)
        allres[w] = globals()[f"run_{w}"]()
        print(f"   ({time.time() - t0:.0f}s)", flush=True)
        path.write_text(json.dumps(allres, indent=1))
