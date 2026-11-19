# Cuttlefish PoC: virtual time vs wall-clock ranking

**What it shows.** Cuttlefish (AFT '25) ranks trades by *virtual time*: each instruction has a fixed public cost, so the same code on the same input always gets the same stamp. Here, two traders race on 2,000 events. Trader B runs exactly *k* more instructions than trader A, so A should win every race. Each race is ranked two ways: by measured wall-clock reaction time (what a stopwatch, i.e. DBO, or a timestamp, i.e. CloudEx, actually sees) and by virtual time.

**Files**
- `vm.py`: a tiny eBPF-flavoured register VM. It has a fixed cost table, a virtual clock that follows the paper's rule (clock = max(release time D(x), clock) + instruction costs), map state, and `SUBMIT`. It comes with three example strategies (threshold, moving average, padded).
- `native.py`: generates and compiles `native_race.c`. The same race runs as native arm64 code, with exactly *k* dependent `add` instructions, timed with `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`.
- `demo.py`: runs everything on an idle machine and on a loaded one (one busy process per CPU core). It writes `results.json` and `cuttlefish_poc_{dark,light}.png`.

**Run** (from `experiments/`): `.venv/bin/python -I poc/cuttlefish/demo.py`. This takes about 12 s on an M4 and needs `cc`.

**Result on this Mac (M4, macOS).**
- Virtual time ranks **100%** of races correctly for every *k*, and the result is identical across reruns.
- Native wall-clock ranking is a **coin flip (≈50%) for k ≤ 100**, then 100% for k ≥ 1000. Small amounts of work are invisible to the clock: its tick is 41.7 ns, and out-of-order execution hides about 100 dependent adds.
- The Python VM resolves even k = 1 (~99%), because one interpreted instruction costs ~110 ns, which exceeds its ~40–80 ns jitter IQR.

**Caveats.**
- This is a proof-of-concept, not the paper's system: there is no eBPF verifier, JIT, network or ordering buffer.
- The Python VM inflates per-instruction cost by about 400×, which makes its wall-clock races unrealistically easy. The native run is the realistic one.
- macOS's user-space clock has 41.7 ns resolution, so the measured IQR is quantised. The paper measured an IQR of 20 ns and >1 µs at p99.9 on Azure VMs, with a different methodology.
- "Noisy neighbour" here means local busy processes, not cloud tenancy.
- The flip side, which Cuttlefish also implies: virtual time *charges* for 100 extra instructions that cost no visible real time, so its cost model and real cost can differ.
