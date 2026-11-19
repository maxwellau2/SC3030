"""Single source for the presentation charts (Plotly).

  dark  -> slides/charts/charts-data.js   (interactive, embedded in reveal.js deck)
  light -> experiments/figures/plotly/*.pdf (static, for the LaTeX doc)

Colours: validated reference categorical palette, one fixed colour per entity
(DBO blue, CloudEx orange/aqua/yellow by hold, probabilistic magenta, virtual time
green, Direct/FCFS neutral grey reference).
Run from experiments/:  .venv/bin/python -I fairlab/charts.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SLIDES = ROOT.parent / "slides"
PDF_OUT = ROOT / "figures" / "plotly"

T = {
    "dark": dict(bg="rgba(0,0,0,0)", ink="#e5e7eb", ink2="#94a3b8", grid="#1e293b", zero="#334155",
                 DBO="#3987e5", CX15="#d95926", CX30="#199e70", CX50="#c98500", TOMMY="#d55181",
                 VT="#0ca30c", REF="#94a3b8", font=16),
    "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", grid="#e7e5e4", zero="#d6d3d1",
                  DBO="#2a78d6", CX15="#eb6834", CX30="#1baf7a", CX50="#eda100", TOMMY="#e87ba4",
                  VT="#008300", REF="#6b7280", font=15),
}
ZERO = 0.003  # where "perfect clocks" sits on log axes
SIG_TICKS = dict(tickvals=[ZERO, 0.01, 0.1, 1, 10], ticktext=["perfect", "10 ns", "100 ns", "1 µs", "10 µs"])


def rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


def base(fig, th, title=None, legend=None, height=None):
    c = T[th]
    fig.update_layout(
        template="none", paper_bgcolor=c["bg"], plot_bgcolor=c["bg"],
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif", size=c["font"], color=c["ink2"]),
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=c["font"] + 3, color=c["ink"])) if title else None,
        margin=dict(l=75, r=25, t=60 if title else 35, b=85 if th == "light" else 60),
        legend=legend or dict(orientation="h", y=-0.22, x=0, font=dict(color=c["ink"])),
        hoverlabel=dict(font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif")),
        height=height,
    )
    fig.update_xaxes(gridcolor=c["grid"], zerolinecolor=c["zero"], linecolor=c["zero"], showline=True,
                     tickfont=dict(color=c["ink2"]), title_font=dict(color=c["ink2"]))
    fig.update_yaxes(gridcolor=c["grid"], zerolinecolor=c["zero"], linecolor=c["zero"], showline=True,
                     tickfont=dict(color=c["ink2"]), title_font=dict(color=c["ink2"]))
    fig.update_annotations(font=dict(color=c["ink"]))
    return fig


def inside(th, x, y, xa="left", ya="top"):
    """Legend placed inside an empty region of the plot (avoids collisions on narrow charts)."""
    c = T[th]
    bg = "rgba(11,18,32,0.85)" if th == "dark" else "rgba(255,255,255,0.9)"
    return dict(x=x, y=y, xanchor=xa, yanchor=ya, bgcolor=bg, bordercolor=c["grid"], borderwidth=1,
                font=dict(color=c["ink"], size=c["font"] - 2), orientation="v")


def line(fig, x, y, name, color, dash="solid", row=None, col=None, hover="%{y:.2f}%", custom=None,
         showlegend=True, shape="linear", marker=True, legendgroup=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(
        x=x, y=y, name=name, mode="lines+markers" if marker else "lines",
        line=dict(color=color, width=2.5, dash=dash, shape=shape), marker=dict(size=8, color=color),
        customdata=custom, hovertemplate=f"<b>{name}</b><br>{hover}<extra></extra>",
        showlegend=showlegend, legendgroup=legendgroup or name), **kw)


def band(fig, x, lo, hi, color, row=None, col=None, legendgroup=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip",
                             showlegend=False, legendgroup=legendgroup), **kw)
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(color, 0.16), hoverinfo="skip", showlegend=False,
                             legendgroup=legendgroup), **kw)


def sig_label(s):
    return "perfect" if s == 0 else (f"{s*1000:g} ns" if s < 1 else f"{s:g} µs")


# ------------------------------------------------------------------ charts
def clock_error(th):
    c = T[th]
    df = pd.read_csv(RES / "e1_clock_error.csv")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=("Exchange data: release & stamp share one clock",
                                        "External trigger (news, another venue)"))
    rel = df[df.scenario == "released"]
    for d, key in [(15, "CX15"), (30, "CX30"), (50, "CX50")]:
        g = rel[rel.hold_us == d].groupby("sigma_us").fairness
        s = g.mean().index.values
        x = np.where(s == 0, ZERO, s)
        band(fig, x, 100 * g.min().values, 100 * g.max().values, c[key], 1, 1, f"cx{d}")
        line(fig, x, 100 * g.mean().values, f"CloudEx, hold {d} µs", c[key], row=1, col=1,
             custom=[sig_label(v) for v in s], hover="clock error %{customdata}<br>%{y:.2f}% of races correct",
             legendgroup=f"cx{d}")
    ext = df[df.scenario == "external"].groupby("sigma_us").fairness
    s = ext.mean().index.values
    x = np.where(s == 0, ZERO, s)
    band(fig, x, 100 * ext.min().values, 100 * ext.max().values, c["CX50"], 1, 2, "cx50")
    line(fig, x, 100 * ext.mean().values, "CloudEx, hold 50 µs", c["CX50"], row=1, col=2,
         custom=[sig_label(v) for v in s], hover="clock error %{customdata}<br>%{y:.2f}% of races correct",
         showlegend=False, legendgroup="cx50")
    dbo, dr = 100 * df.dbo_fairness.iloc[0], 100 * df.direct_fairness.iloc[0]
    xs = [ZERO, 10]
    for col in (1, 2):
        line(fig, xs, [dbo, dbo], "DBO (no clocks)", c["DBO"], row=1, col=col, marker=False,
             showlegend=col == 1, hover="100% at any clock error")
        line(fig, xs, [dr, dr], "Direct / FCFS", c["REF"], dash="dash", row=1, col=col, marker=False,
             showlegend=col == 1, hover=f"{dr:.1f}% (no fairness mechanism)")
        fig.add_vrect(x0=0.1, x1=0.2, fillcolor=rgba(c["ink2"], 0.12), line_width=0, row=1, col=col)
        fig.add_vline(x=1.5, line=dict(color=c["ink2"], width=1, dash="dot"), row=1, col=col)
    fig.add_annotation(x=np.log10(0.14), y=56, text="CloudEx sync<br>p99 ≈ 159 ns", showarrow=False,
                       font=dict(size=c["font"] - 3, color=c["ink2"]), row=1, col=2)
    fig.add_annotation(x=np.log10(1.6), y=56, text="trader gap<br>1.5 µs", showarrow=False, xanchor="left",
                       font=dict(size=c["font"] - 3, color=c["ink2"]), row=1, col=2)
    fig.update_xaxes(type="log", title_text="clock-sync error between gateways (σ)", **SIG_TICKS)
    fig.update_yaxes(range=[50, 101.5], title_text="races ordered correctly (%)", row=1, col=1)
    base(fig, th, legend=dict(orientation="h", y=-0.3, x=0, font=dict(color=T[th]["ink"])))
    fig.update_layout(margin=dict(b=110 if th == "light" else 95))
    return fig


def placement(th):
    c = T[th]
    df = pd.read_csv(RES / "e5_placement.csv")
    fig = go.Figure()
    for algo, key, dash, nm in [("Direct", "REF", "dash", "Direct / FCFS"),
                                ("CloudEx(15|15)", "CX15", "solid", "CloudEx, hold 15 µs"),
                                ("CloudEx(30|30)", "CX30", "solid", "CloudEx, hold 30 µs"),
                                ("DBO(20|25)", "DBO", "solid", "DBO")]:
        g = df[df.algo == algo].groupby("spread_us").fairness
        x = g.mean().index.values
        band(fig, x, 100 * g.min().values, 100 * g.max().values, c[key], legendgroup=nm)
        line(fig, x, 100 * g.mean().values, nm, c[key], dash=dash, legendgroup=nm,
             hover="VMs up to %{x} µs apart<br>%{y:.1f}% of races correct (mean of 10 placements)")
    fig.add_annotation(x=20, y=57.6, text="paper's real Azure run: 57.6%", showarrow=True, ax=-90, ay=28,
                       arrowcolor=c["ink2"], font=dict(size=c["font"] - 2, color=c["ink2"]))
    fig.update_xaxes(title_text="spread of fixed per-VM network distance (µs)")
    fig.update_yaxes(title_text="races ordered correctly (%)", range=[38, 102])
    return base(fig, th, legend=inside(th, 0.01, 0.02, "left", "bottom"))


def radius(th):
    c = T[th]
    df = pd.read_csv(RES / "e3_radius.csv")
    x = np.where(df.extra_rtt_us == 0, 5, df.extra_rtt_us)
    lab = [("0" if v == 0 else (f"{v:g} µs" if v < 1000 else f"{v/1000:g} ms")) for v in df.extra_rtt_us]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Everyone waits for the farthest trader…",
                                        "…or the farthest trader loses"))
    line(fig, x, df.wait_others_mean_latency, "DBO waits for everyone", c["DBO"], row=1, col=1,
         custom=lab, hover="distant trader +%{customdata} round trip<br>others' mean latency %{y:,.0f} µs")
    line(fig, x, df.excl_others_mean_latency, "DBO straggler rule", c["REF"], dash="dash",
         row=1, col=1, custom=lab, hover="+%{customdata}: others' mean latency %{y:,.0f} µs")
    line(fig, x, 100 * df.excl_fairness_distant_pairs, "DBO straggler rule", c["REF"],
         dash="dash", row=1, col=2, custom=lab, showlegend=False,
         hover="+%{customdata}: distant trader won %{y:.1f}% of races it deserved")
    line(fig, x, 100 * df.direct_fairness_distant_pairs, "Direct / FCFS", c["CX15"], row=1, col=2,
         custom=lab, hover="+%{customdata}: distant trader won %{y:.1f}% of races it deserved")
    fig.add_vline(x=200_000, line=dict(color=c["ink2"], width=1, dash="dot"), row=1, col=1)
    fig.add_annotation(x=np.log10(200_000), y=np.log10(800), text="≈ US↔Singapore", showarrow=False,
                       xanchor="right", font=dict(size=c["font"] - 3, color=c["ink2"]), row=1, col=1)
    ticks = dict(tickvals=[5, 10, 100, 1e3, 1e4, 1e5], ticktext=["0", "10 µs", "100 µs", "1 ms", "10 ms", "100 ms"])
    fig.update_xaxes(type="log", title_text="extra round trip of ONE distant trader", **ticks)
    fig.update_yaxes(type="log", title_text="others' mean latency", row=1, col=1,
                     tickvals=[100, 1e3, 1e4, 1e5], ticktext=["100 µs", "1 ms", "10 ms", "100 ms"])
    fig.update_yaxes(title_text="races it deserved, won (%)", range=[-3, 103], row=1, col=2)
    return base(fig, th, legend=inside(th, 0.99, 0.6, "right", "middle"))


def liveness(th):
    c = T[th]
    r = json.loads((RES / "poc_results.json").read_text())
    fig = go.Figure()
    for name, tl, fail_ms, key, dash in [
        ("CloudEx (timeout)", r["cloudex"]["gateway_fail"]["timeline"], 20, "CX15", "solid"),
        ("Onyx (waits for all)", r["onyx"]["gateway_fail"]["timeline"], 10, "CX50", "solid"),
        ("DBO (waits for all)", r["dbo"]["rb_fail_no_timeout"]["timeline"], 20, "DBO", "solid"),
        ("DBO + straggler timeout", r["dbo"]["rb_fail_timeout200"]["timeline"], 20, "DBO", "dot")]:
        tl = np.array(tl, float)
        pre = tl[max(0, fail_ms - 8):fail_ms - 1].mean()
        xx = np.arange(len(tl)) - fail_ms
        keep = (xx >= -5) & (xx <= 8)
        line(fig, xx[keep], 100 * tl[keep] / pre, name, c[key], dash=dash, shape="hv", marker=False,
             hover="t = %{x} ms after failure<br>%{y:.0f}% of normal order rate")
    fig.add_vline(x=0, line=dict(color=c["ink2"], width=1, dash="dot"))
    fig.add_annotation(x=0.2, y=20, text="one participant<br>dies", showarrow=False, xanchor="left",
                       font=dict(size=c["font"] - 2, color=c["ink2"]))
    fig.add_annotation(x=8, y=97, text="keep going (9 of 10 left)", showarrow=False, xanchor="right",
                       font=dict(size=c["font"] - 2, color=c["ink"]))
    fig.add_annotation(x=8, y=7, text="frozen", showarrow=False, xanchor="right",
                       font=dict(size=c["font"] - 2, color=c["ink"]))
    fig.update_xaxes(title_text="time relative to the failure (ms)")
    fig.update_yaxes(title_text="orders executed (% of normal)", range=[-5, 120])
    return base(fig, th, legend=inside(th, 0.42, 0.72, "left", "top"))


def cuttlefish(th):
    c = T[th]
    r = json.loads((ROOT / "poc" / "cuttlefish" / "results.json").read_text())
    fig = go.Figure()
    ks = [x["k"] for x in r["quiet"]["native"]["results"]]
    line(fig, ks, [100.0] * len(ks), "instruction count (Cuttlefish)", c["VT"],
         hover="%{x} extra instructions: %{y:.0f}% correct, identical every rerun")
    line(fig, ks, [100 * x["A_wins"] for x in r["quiet"]["native"]["results"]],
         "wall clock, idle", c["DBO"], hover="%{x} extra instructions: %{y:.1f}% correct")
    line(fig, ks, [100 * x["A_wins"] for x in r["loaded"]["native"]["results"]],
         "wall clock, noisy neighbours", c["DBO"], dash="dot", hover="%{x} extra instructions: %{y:.1f}% correct")
    fig.add_hline(y=50, line=dict(color=c["ink2"], width=1, dash="dot"))
    fig.add_annotation(x=0, y=53, text="coin flip", showarrow=False, xanchor="left",
                       font=dict(size=c["font"] - 2, color=c["ink2"]))
    fig.update_xaxes(type="log", title_text="extra instructions run by the slower trader (native arm64)",
                     tickvals=ks, ticktext=[f"{k:,}" for k in ks])
    fig.update_yaxes(title_text="races won by the faster trader (%)", range=[40, 104])
    return base(fig, th, legend=inside(th, 0.99, 0.2, "right", "bottom"))


def prob_sequencer(th):
    c = T[th]
    r = json.loads((RES / "poc_tommy.json").read_text())["online"]
    s = [x["sigma_us"] for x in r]
    lab = [sig_label(v) for v in s]
    emit = [f"{x['median_emit_latency_us']:.0f}" for x in r]
    fig = go.Figure()
    line(fig, s, [100 * x["timestamp_wrong"] for x in r], "timestamp order: wrong", c["CX50"],
         custom=lab, hover="σ = %{customdata}<br>%{y:.2f}% of races wrong")
    line(fig, s, [100 * x["prob_wrong"] for x in r], "probabilistic: wrong", c["TOMMY"],
         custom=lab, hover="σ = %{customdata}<br>%{y:.2f}% of races wrong")
    line(fig, s, [100 * x["prob_tie"] for x in r], "probabilistic: declared tie", c["TOMMY"], dash="dash",
         custom=np.stack([lab, emit], axis=-1), hover="σ = %{customdata[0]}<br>%{y:.1f}% ties<br>emit after %{customdata[1]} µs")
    fig.update_xaxes(type="log", title_text="client clock error (σ)", tickvals=s, ticktext=lab)
    fig.update_yaxes(title_text="share of races (%)", rangemode="tozero")
    return base(fig, th, legend=inside(th, 0.01, 0.99))


def tradeoff(th):
    c = T[th]
    df = pd.read_csv(RES / "e0_reproduction.csv")
    cx = df[df.algo.str.startswith("CloudEx")].copy()
    cx["d"] = cx.algo.str.extract(r"\((\d+)\|").astype(int)
    fig = go.Figure()
    line(fig, cx["mean"], 100 * (1 - cx.fairness), "CloudEx (hold 10–100 µs)", c["CX15"],
         custom=cx.d, hover="hold %{customdata} µs<br>%{x:.0f} µs mean latency<br>%{y:.3f}% wrong")
    for nm, key, col, sym, pos in [("DBO", "DBO(20|25)", c["DBO"], "diamond", "bottom center"),
                                   ("MaxRTT bound", "MaxRTT", c["ink2"], "square", "top right"),
                                   ("Direct / FCFS", "Direct", c["REF"], "square", "top right")]:
        rr = df[df.algo == key].iloc[0]
        y = max(100 * (1 - rr.fairness), 0.003)
        fig.add_trace(go.Scatter(x=[rr["mean"]], y=[y], mode="markers+text", name=nm,
                                 marker=dict(size=12, color=col, symbol=sym), showlegend=False,
                                 text=[nm + (" (0% wrong)" if rr.fairness == 1 else "")], textposition=pos,
                                 textfont=dict(color=col), hovertemplate=f"<b>{nm}</b><br>%{{x:.1f}} µs, {100*(1-rr.fairness):.3f}% wrong<extra></extra>"))
    fig.update_xaxes(title_text="mean end-to-end latency (µs)")
    fig.update_yaxes(type="log", title_text="races ordered wrongly (%)", range=[-3.0, 0.7],
                     tickvals=[0.01, 0.1, 1], ticktext=["0.01%", "0.1%", "1%"])
    return base(fig, th, legend=dict(x=0.62, y=0.95))


CHARTS = {"clock_error": clock_error, "placement": placement, "radius": radius,
          "liveness": liveness, "cuttlefish": cuttlefish, "prob_sequencer": prob_sequencer,
          "tradeoff": tradeoff}
PDF_SIZE = {"clock_error": (1150, 470), "radius": (1150, 470)}

if __name__ == "__main__":
    # dark, interactive -> slides
    (SLIDES / "charts").mkdir(parents=True, exist_ok=True)
    (SLIDES / "vendor").mkdir(parents=True, exist_ok=True)
    data = {k: json.loads(f("dark").to_json()) for k, f in CHARTS.items()}
    (SLIDES / "charts" / "charts-data.js").write_text("window.FAIR_CHARTS = " + json.dumps(data) + ";\n")
    import plotly
    (SLIDES / "vendor" / "plotly.min.js").write_text(get_plotlyjs())
    print("slides: charts-data.js + vendor/plotly.min.js (plotly.js bundled with plotly", plotly.__version__ + ")")
    # light, static -> LaTeX
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    for k, f in CHARTS.items():
        w, h = PDF_SIZE.get(k, (820, 470))
        f("light").write_image(PDF_OUT / f"{k}.pdf", width=w, height=h)
        f("light").write_image(PDF_OUT / f"{k}.png", width=w, height=h, scale=2)
        print("latex:", k)
