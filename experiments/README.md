# Experiments and proofs of concept

Everything runs locally on the DBO authors' **real 15-minute Azure latency trace**. No cloud instances are needed.

## Reproduce everything

```sh
make          # from the repo root or experiments/: setup + data + validate + experiments + PoCs + figures (~8 min)
make help     # individual targets: validate, experiments, e0..e5, poc, figures, clean, ...
```

`make` does the following:
- Creates `.venv` with the pinned packages in `requirements.txt` (Python 3.12, using uv or pip).
- Clones the DBO authors' simulator, [eash3010/dbo-simulation](https://github.com/eash3010/dbo-simulation), at the commit used here. It's the only paper with public code.
- Downloads their 313 MB Azure trace (`direct.zip`, release v1) and verifies its SHA-256 checksum.

You need `git`, `curl` and a C compiler. The PDF/PNG chart export (kaleido) also needs Chrome or Chromium.

The committed `results/` and `figures/` are the outputs of exactly this run:
- E0–E5 and the CloudEx/Onyx/DBO and Beyond-Lamport PoCs are seeded and deterministic.
- The Cuttlefish native timing (`poc/cuttlefish/results.json`) is a real measurement, so it will differ slightly between machines.

## Layout

| Path | What it is |
|---|---|
| `fairlab/azure_trace.py` | Fast loader. It rebuilds the authors' exact 2M-µs round-trip slice of the trace. |
| `fairlab/sim.py` | A vectorised port of the authors' Direct, CloudEx, DBO and MaxRTT algorithms, plus our extensions: clock error, external triggers, straggler exclusion, compute jitter, and probabilistic scoring. |
| `fairlab/validate.py` | Checks the port against the authors' original classes. The difference is **0.0** for ordering, latency and fairness on all four algorithms. |
| `fairlab/run_experiments.py` | Experiments E0–E5. Results go to `results/*.csv`. |
| `fairlab/plots.py` | Charts F0–F5, written to `figures/{dark,light}/` (exploratory matplotlib versions). |
| `fairlab/charts.py` | **The presentation charts**, all built in Plotly from one source: the interactive dark versions go to `../slides/charts/charts-data.js` (with plotly.js vendored in `../slides/vendor/`), and the light versions go to `figures/plotly/*.pdf\|png` for the LaTeX doc. |
| `poc/core.py`, `poc/protocols.py` | Message-level event-driven prototypes of **CloudEx**, **Onyx** and **DBO**, running over trace-driven links. |
| `poc/tommy.py` | Probabilistic sequencer (Beyond Lamport) plus the intransitivity demo using FFT convolution. |
| `poc/cuttlefish/` | Instruction-counting VM plus a native arm64 race timed on this Mac. See its README. |
| `poc/run_poc.py`, `poc/plots_poc.py` | PoC scenarios and charts P1–P2. |

## Run steps by hand

These are what the Makefile runs. Run them from `experiments/`, using `.venv/bin/python -I`:

```sh
fairlab/validate.py            # port == original (about 25 s)
fairlab/run_experiments.py     # E0..E5 (about 3 min)
fairlab/plots.py
poc/run_poc.py                 # CloudEx, Onyx, DBO PoCs (about 2 min)
poc/tommy.py
poc/plots_poc.py
poc/cuttlefish/demo.py
```

## Headline results

All figures are for 10 traders over 1M market-data points unless noted.

| Exp | Question | Result |
|---|---|---|
| E0 | Reproduce the paper's simulation | **Exact match** with `simulation.dat`: DBO 57.71 µs and 100% fair; MaxRTT 31.12 µs; CloudEx(15) 99.514%. CloudEx never reaches 100%, even with a 100 µs hold (99.94%). |
| E1a | CloudEx under clock-sync error, exchange-released data | Robust. Release and stamp use the same gateway clock, so a static error **cancels**. With a 50 µs hold it stays at 99.92% up to σ = 10 µs. It only hurts when the hold is too short and data arrives late: a 15 µs hold falls to 73.6% at 10 µs. |
| E1b | Same, for an **external** trigger (news, another venue) | Not robust: 96.2% at σ = 1 µs, 83.1% at 3 µs, 61.3% at 10 µs. |
| E2 | Probabilistic ordering on the E1b case | At σ = 10 µs: wrong orders drop from 38.7% to 19.1%, with 45% declared ties. |
| E3 | One distant trader (DBO) | Everyone else's latency ≈ 54 µs + that trader's extra round trip (200 ms gives 200,054 µs). With the straggler rule the others stay at 57 µs, and the distant trader wins 1.2% of the races it deserved at +60 µs, 0% beyond that. |
| E4 | Compute jitter, using a model calibrated to Cuttlefish Fig. 1 | Ranking by measured reaction time: 52% (1 ns gap), 68% (10 ns), 92% (30 ns), 99.7% (100 ns). Ranking by instruction count is 100% by construction. |
| E5 | Fixed per-VM distance spread | First-come-first-served falls from 97.8% to 59.7% at a 20 µs spread (the paper measured 57.6% on real Azure). DBO stays at 100%. CloudEx(30) holds until the spread exceeds its hold (95.96% at 20 µs). |
| PoC | CloudEx / Onyx / DBO when one participant fails | CloudEx keeps going on its timeout. Onyx and DBO (wait for everyone) **freeze**. DBO with its straggler timeout recovers. |
| PoC | Onyx tree vs unicast, 40 receivers, 0.2 µs per send | Median arrival is worse because of the extra hop (30.3 vs 21.9 µs), but p99 latency is better (136 vs 243 µs) and 96.8% of messages land within 1 µs of each other. When unicast is overloaded (0.5 µs per send), it explodes to 5.8 ms while the tree stays at about 35 µs (`results/poc_onyx_overload_ser0.5_log.txt`). |
| PoC | Probabilistic sequencer, online, at σ = 1 µs | Wrong orders 1.45% vs 3.47% for timestamp order, with 10.5% ties and emission latency 25 µs (vs 18 µs at σ = 100 ns). |
| PoC | Intransitivity | Dice-shaped clock errors give P(A<B) = P(B<C) = P(C<A) = 0.556. Gaussian errors: 0 cycles in 20,000 trials. |
| PoC | Cuttlefish, native arm64 | Wall-clock ranking is a coin flip (≈50%) for gaps of ≤100 instructions and 100% at ≥1000. Virtual time is 100% at every gap and deterministic. |

## Caveats

- **These are simulations and prototypes of the mechanisms, not the authors' systems.** There is no DPDK, no real smartNICs, no matching engine, and no real cloud clocks.
- **The authors' simulator gives every trader windows of one trader's trace.** E5 adds the per-VM asymmetry it lacks.
- **Clock error is modelled as a static per-gateway offset in E1/E2** (realistic over the µs between release and stamp). The Beyond Lamport PoC draws a fresh error for each message, following that paper's model.
- **The E4 jitter model is synthetic**, calibrated to the summary statistics in Cuttlefish's Fig. 1. The Cuttlefish PoC's native run is a real measurement on one Mac.
- **Onyx PoC simplifications:** delay-estimate feedback is instantaneous, and failure handling is not implemented beyond what the paper describes.
