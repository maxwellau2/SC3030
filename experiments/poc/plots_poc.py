"""Figures for the protocol PoCs (dark for slides, light for LaTeX)."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fairlab"))
from plots import THEMES, style, save  # noqa: E402

RES = HERE.parent / "results"


def liveness(th):
    c = style(th)
    r = json.loads((RES / "poc_results.json").read_text())
    series = [
        ("CloudEx (timeout sequencer)", r["cloudex"]["gateway_fail"]["timeline"], 20, c["CX15"], "-"),
        ("Onyx (wait for every gateway)", r["onyx"]["gateway_fail"]["timeline"], 10, c["CX50"], "-"),
        ("DBO (wait for every RB)", r["dbo"]["rb_fail_no_timeout"]["timeline"], 20, c["DBO"], "-"),
        ("DBO + straggler timeout", r["dbo"]["rb_fail_timeout200"]["timeline"], 20, c["DBO"], "--"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, tl, fail_ms, col, ls in series:
        tl = np.array(tl, float)
        pre = tl[max(0, fail_ms - 8):fail_ms - 1].mean()
        x = np.arange(len(tl)) - fail_ms
        keep = (x >= -6) & (x <= 9)
        ax.step(x[keep], 100 * tl[keep] / pre, where="post", color=col, ls=ls, lw=2.4, label=name)
    ax.axvline(0, color=c["ink2"], lw=1, ls=":")
    ax.text(0.15, 30, "one participant\ndies here", color=c["ink2"], fontsize=10)
    ax.text(8.9, 96, "CloudEx & DBO+timeout: keep going\n(9 of 10 traders left)", ha="right", fontsize=10, color=c["ink"])
    ax.text(8.9, 5, "Onyx & DBO: frozen", ha="right", fontsize=10, color=c["ink"])
    ax.set_ylim(-5, 125)
    ax.set_xlabel("time relative to the failure (ms)")
    ax.set_ylabel("orders executed (% of normal rate)")
    ax.set_title("PoC: one dead participant freezes wait-for-everyone sequencers")
    ax.legend(loc="center left", fontsize=10.5)
    save(fig, "p1_liveness", th)


def tommy(th):
    c = style(th)
    r = json.loads((RES / "poc_tommy.json").read_text())["online"]
    s = np.array([x["sigma_us"] for x in r])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(s, [100 * x["timestamp_wrong"] for x in r], "-o", color=c["CX50"], label="timestamp order: wrong")
    ax.plot(s, [100 * x["prob_wrong"] for x in r], "-o", color=c["TOMMY"], label="probabilistic: wrong")
    ax.plot(s, [100 * x["prob_tie"] for x in r], "--o", color=c["TOMMY"], label="probabilistic: declared tie")
    for x in r:
        if x["sigma_us"] in (0.1, 1.0, 5.0):
            ax.annotate(f"emit +{x['median_emit_latency_us']:.0f} µs", (x["sigma_us"], 100 * x["prob_tie"]),
                        xytext=(6, 8), textcoords="offset points", fontsize=9.5, color=c["ink2"])
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.5, 1, 2, 5]); ax.set_xticklabels(["100 ns", "500 ns", "1 µs", "2 µs", "5 µs"])
    ax.set_xlabel("client clock error (σ)"); ax.set_ylabel("share of races (%)")
    ax.set_title("PoC: online probabilistic sequencer on real Azure links")
    ax.legend(loc="upper left")
    save(fig, "p2_probabilistic_sequencer", th)


if __name__ == "__main__":
    for th in THEMES:
        liveness(th); tommy(th); print("ok", th)
