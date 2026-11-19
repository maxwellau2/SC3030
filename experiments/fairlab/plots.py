"""Plot experiment results. Dark theme (slides, surface #0b1220) and light theme (LaTeX).
Colours: validated reference categorical palette, one fixed colour per entity."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"

THEMES = {
    "dark": dict(bg="#0b1220", ink="#e5e7eb", ink2="#94a3b8", grid="#1e293b", ref="#94a3b8",
                 DBO="#3987e5", CX15="#d95926", CX30="#199e70", CX50="#c98500",
                 TOMMY="#d55181", VT="#0ca30c"),
    "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", grid="#e7e5e4", ref="#6b7280",
                  DBO="#2a78d6", CX15="#eb6834", CX30="#1baf7a", CX50="#eda100",
                  TOMMY="#e87ba4", VT="#008300"),
}


def style(th):
    c = THEMES[th]
    plt.rcParams.update({
        "figure.facecolor": c["bg"], "axes.facecolor": c["bg"], "savefig.facecolor": c["bg"],
        "axes.edgecolor": c["grid"], "axes.labelcolor": c["ink2"], "text.color": c["ink"],
        "xtick.color": c["ink2"], "ytick.color": c["ink2"], "grid.color": c["grid"],
        "axes.grid": True, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 2.2, "lines.markersize": 7, "legend.frameon": False, "legend.fontsize": 11.5,
    })
    return c


def label_end(ax, x, y, text, color, dy=0, ha="left"):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=color,
                fontsize=11.5, fontweight="bold", va="center", ha=ha)


def save(fig, name, th):
    out = FIG / th
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


SYM_ZERO = 0.003  # where sigma=0 is drawn on log axes


# ---------------------------------------------------------------- F0 trade-off (E0)
def f0(th):
    c = style(th)
    df = pd.read_csv(RES / "e0_reproduction.csv")
    cx = df[df.algo.str.startswith("CloudEx")].copy()
    cx["d"] = cx.algo.str.extract(r"\((\d+)\|").astype(int)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cx["mean"], 100 * (1 - cx.fairness), "-o", color=c["CX15"], label="CloudEx (hold 10–100 µs)")
    for _, r in cx[cx.d.isin([15, 30, 100])].iterrows():
        ax.annotate(f"hold {r.d} µs", (r["mean"], 100 * (1 - r.fairness)), xytext=(6, 8),
                    textcoords="offset points", color=c["ink2"], fontsize=10.5)
    for name, key, col in [("DBO", "DBO(20|25)", c["DBO"]), ("MaxRTT (bound)", "MaxRTT", c["ink2"]),
                           ("Direct", "Direct", c["ref"])]:
        r = df[df.algo == key].iloc[0]
        y = max(100 * (1 - r.fairness), 0.004)
        ax.plot(r["mean"], y, "D" if name == "DBO" else "s", color=col, ms=9)
        label_end(ax, r["mean"], y, name + (" (0% wrong)" if r.fairness == 1 else ""), col,
                  dy={"DBO": -13, "MaxRTT (bound)": 13}.get(name, 0))
    ax.set_yscale("log"); ax.set_ylim(0.0015, 5)
    ax.set_xlabel("mean end-to-end latency (µs)"); ax.set_ylabel("races ordered wrongly (%)")
    ax.set_title("Reproduced: CloudEx trades latency for fairness")
    ax.legend(loc="upper right")
    save(fig, "f0_tradeoff", th)


# ---------------------------------------------------------------- F1 clock error (E1)
def f1(th):
    c = style(th)
    df = pd.read_csv(RES / "e1_clock_error.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    rel = df[df.scenario == "released"]
    for d, key in [(15, "CX15"), (30, "CX30"), (50, "CX50")]:
        g = rel[rel.hold_us == d].groupby("sigma_us").fairness
        x = np.array([s if s > 0 else SYM_ZERO for s in g.mean().index])
        a1.fill_between(x, 100 * g.min().values, 100 * g.max().values, color=c[key], alpha=0.18, lw=0)
        a1.plot(x, 100 * g.mean().values, "-o", color=c[key], label=f"CloudEx, hold {d} µs")
    a1.set_title("Exchange data: release + stamp share a clock")
    ext = df[df.scenario == "external"].groupby("sigma_us").fairness
    x = np.array([s if s > 0 else SYM_ZERO for s in ext.mean().index])
    a2.fill_between(x, 100 * ext.min().values, 100 * ext.max().values, color=c["CX50"], alpha=0.18, lw=0)
    a2.plot(x, 100 * ext.mean().values, "-o", color=c["CX50"], label="CloudEx, hold 50 µs")
    a2.set_title("External trigger (news, other venue)")
    dbo = 100 * df.dbo_fairness.iloc[0]; dr = 100 * df.direct_fairness.iloc[0]
    for ax in (a1, a2):
        ax.axhline(dbo, color=c["DBO"], lw=2, label="DBO (no clocks)")
        ax.axhline(dr, color=c["ref"], lw=1.4, ls="--", label=f"Direct, no fairness ({dr:.1f}%)")
        ax.axvspan(0.1, 0.2, color=c["ink2"], alpha=0.12, lw=0)
        ax.axvline(1.5, color=c["ink2"], lw=1, ls=":")
        ax.set_xscale("log"); ax.set_ylim(50, 101.5)
        ax.set_xticks([SYM_ZERO, 0.01, 0.1, 1, 10]); ax.set_xticklabels(["perfect", "10 ns", "100 ns", "1 µs", "10 µs"])
        ax.set_xlabel("clock-sync error between gateways (σ)")
        ax.legend(loc="lower left")
    a2.text(0.14, 76, "CloudEx's\nHuygens sync\np99≈159 ns", ha="center", fontsize=9.5, color=c["ink2"])
    a2.text(1.65, 76, "trader gap\n1.5 µs", fontsize=9.5, color=c["ink2"])
    a1.set_ylabel("races ordered correctly (%)")
    fig.suptitle("New: clock error is harmless for released data, costly for external events",
                 x=0.01, ha="left", fontweight="bold", fontsize=15, color=c["ink"])
    fig.tight_layout()
    save(fig, "f1_clock_error", th)


# ---------------------------------------------------------------- F2 probabilistic (E2)
def f2(th):
    c = style(th)
    df = pd.read_csv(RES / "e2_probabilistic.csv").groupby("sigma_us").mean()
    x = np.array([s if s > 0 else SYM_ZERO for s in df.index])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, 100 * df.cloudex_wrong, "-o", color=c["CX50"], label="CloudEx (trust timestamps): wrong")
    ax.plot(x, 100 * df.tommy_wrong, "-o", color=c["TOMMY"], label="Beyond Lamport (p > 0.75): wrong")
    ax.plot(x, 100 * df.tommy_tie, "--", color=c["TOMMY"], label="Beyond Lamport: declared tie")
    ax.set_xscale("log")
    ax.set_xticks([SYM_ZERO, 0.01, 0.1, 1, 10]); ax.set_xticklabels(["perfect", "10 ns", "100 ns", "1 µs", "10 µs"])
    ax.set_xlabel("clock-sync error between gateways (σ)"); ax.set_ylabel("share of trader pairs (%)")
    ax.set_title("New: probabilistic ordering swaps wrong orders for ties (external events)")
    ax.legend(loc="upper left")
    save(fig, "f2_probabilistic", th)


# ---------------------------------------------------------------- F3 radius (E3)
def f3(th):
    c = style(th)
    df = pd.read_csv(RES / "e3_radius.csv")
    x = np.array([v if v > 0 else 5 for v in df.extra_rtt_us])
    ticks, tl = [5, 10, 100, 1e3, 1e4, 1e5], ["0", "10", "100", "1 ms", "10 ms", "100 ms"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.plot(x, df.wait_others_mean_latency, "-o", color=c["DBO"], label="DBO waits for everyone")
    a1.plot(x, df.excl_others_mean_latency, "--o", color=c["ref"], label="DBO stops waiting (straggler rule)")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("extra round trip of ONE distant trader (µs)"); a1.set_ylabel("everyone else's mean latency (µs)")
    a1.set_title("Everyone waits for the farthest trader")
    a1.legend(loc="upper left")
    a1.axvline(200_000, color=c["ink2"], ls=":", lw=1)
    a1.text(170_000, 300, "≈ US↔Singapore\nround trip", ha="right", fontsize=10, color=c["ink2"])
    for a in (a1, a2):
        a.set_xticks(ticks); a.set_xticklabels(tl)
    a2.plot(x, 100 * df.excl_fairness_distant_pairs, "-o", color=c["ref"], label="DBO + straggler rule")
    a2.plot(x, 100 * df.direct_fairness_distant_pairs, "-o", color=c["CX15"], label="Direct (first-come-first-served)")
    a2.set_xscale("log"); a2.set_ylim(-3, 103); a2.set_xticks(ticks); a2.set_xticklabels(tl)
    a2.set_xlabel("extra round trip of the distant trader (µs)"); a2.set_ylabel("races it deserved to win that it won (%)")
    a2.set_title("…or the farthest trader loses fairness")
    a2.legend(loc="center right")
    fig.suptitle("New: fairness has a radius", x=0.01, ha="left", fontweight="bold", fontsize=16, color=c["ink"])
    fig.tight_layout()
    save(fig, "f3_radius", th)


# ---------------------------------------------------------------- F4 compute jitter (E4)
def f4(th):
    c = style(th)
    df = pd.read_csv(RES / "e4_compute_jitter.csv")
    g = df.groupby("gap_us").dbo_fairness_with_jitter
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = g.mean().index.values
    ax.fill_between(x, 100 * g.min().values, 100 * g.max().values, color=c["DBO"], alpha=0.18, lw=0)
    ax.plot(x, 100 * g.mean().values, "-o", color=c["DBO"], label="DBO: rank by measured reaction time (incl. CPU jitter)")
    ax.plot(x, 100 * np.ones_like(x), "-", color=c["VT"], label="Cuttlefish: rank by instruction count")
    ax.axhline(50, color=c["ref"], ls=":", lw=1); ax.text(x[0], 51.5, "coin flip", color=c["ink2"], fontsize=10)
    ax.set_xscale("log"); ax.set_ylim(40, 103)
    ax.set_xticks([0.001, 0.01, 0.1, 1]); ax.set_xticklabels(["1 ns", "10 ns", "100 ns", "1 µs"])
    ax.set_xlabel("true difference in reaction time between two traders")
    ax.set_ylabel("races won by the faster trader (%)")
    ax.set_title("New: CPU jitter hides tiny speed differences")
    ax.legend(loc="lower right")
    meta = json.load(open(RES / "e4_jitter_model.json"))
    ax.text(0.99, 0.30, f"jitter model: IQR {meta['iqr_ns']:.0f} ns, p99.9 {meta['p99_9_us']:.1f} µs\n(calibrated to Cuttlefish Fig. 1)",
            transform=ax.transAxes, ha="right", fontsize=9.5, color=c["ink2"])
    save(fig, "f4_compute_jitter", th)


# ---------------------------------------------------------------- F5 placement (E5)
def f5(th):
    c = style(th)
    df = pd.read_csv(RES / "e5_placement.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo, col, ls, nm in [("Direct", c["ref"], "--", "Direct (crypto-style FCFS)"),
                              ("CloudEx(15|15)", c["CX15"], "-", "CloudEx, hold 15 µs"),
                              ("CloudEx(30|30)", c["CX30"], "-", "CloudEx, hold 30 µs"),
                              ("DBO(20|25)", c["DBO"], "-", "DBO")]:
        g = df[df.algo == algo].groupby("spread_us").fairness
        x = g.mean().index.values
        ax.fill_between(x, 100 * g.min().values, 100 * g.max().values, color=col, alpha=0.15, lw=0)
        ax.plot(x, 100 * g.mean().values, ls, marker="o", color=col, label=nm)
    ax.set_ylim(40, 102)
    ax.set_xlabel("spread of fixed per-VM network distance (µs)")
    ax.set_ylabel("races ordered correctly (%)")
    ax.set_title("New: being closer beats being faster, unless the exchange corrects")
    ax.legend(loc="lower left")
    save(fig, "f5_placement", th)


if __name__ == "__main__":
    which = sys.argv[1:] or ["f0", "f1", "f2", "f3", "f4", "f5"]
    for th in THEMES:
        for f in which:
            try:
                globals()[f](th)
                print("ok", th, f)
            except FileNotFoundError as e:
                print("skip", th, f, "(missing", Path(e.filename).name + ")")
