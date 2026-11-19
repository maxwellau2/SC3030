"""Native (C, arm64) version of the race, so the instruction-cost : timing-jitter ratio is
realistic (a Python VM instruction costs ~100 ns; a native instruction < 1 ns).

Generates native_race.c with, for each k, a strategy that executes EXACTLY k extra dependent
`add` instructions (unrolled inline asm) before "submitting", compiles it with cc -O2, runs
races timed with clock_gettime_nsec_np(CLOCK_UPTIME_RAW), and returns parsed JSON.
"""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, BIN = HERE / "native_race.c", HERE / "native_race"


def gen_c(ks, races, jitter_runs):
    fns = []
    for k in [0] + ks:
        body = "\n".join('    __asm__ volatile("add %0, %0, #1" : "+r"(x));' for _ in range(k))
        fns.append(f"""
__attribute__((noinline)) static uint64_t strat_{k}(double price, double *ewma) {{
    *ewma += (price - *ewma) / 8.0;          /* same base work as the Python strategy */
    uint64_t x = (uint64_t)price;
{body}
    return (price > *ewma ? 2 : 1) + (x & 0);  /* "submit_trade" */
}}""")
    table = ", ".join(f"strat_{k}" for k in ks)
    ks_c = ", ".join(str(k) for k in ks)
    return f"""#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
typedef uint64_t (*strat_t)(double, double*);
{''.join(fns)}
static inline uint64_t now(void) {{ return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }}
static int cmp(const void *a, const void *b) {{ uint64_t x=*(uint64_t*)a, y=*(uint64_t*)b; return (x>y)-(x<y); }}
int main(void) {{
    strat_t B[] = {{ {table} }};
    int K[] = {{ {ks_c} }};
    int nk = sizeof(K)/sizeof(K[0]);
    const long R = {races}, J = {jitter_runs};
    double ea = 50, eb = 50, price = 50; volatile uint64_t sink = 0;
    srand(7);
    printf("{{\\"races\\": %ld, \\"results\\": [", R);
    for (int ki = 0; ki < nk; ki++) {{
        long win = 0, tie = 0; double gap_sum = 0;
        for (long i = 0; i < R; i++) {{
            price += ((rand() % 2001) - 1000) / 1000.0; if (price < 1) price = 1;
            uint64_t ta, tb, t0;
            if (i % 2 == 0) {{
                t0 = now(); sink += strat_0(price, &ea); ta = now() - t0;
                t0 = now(); sink += B[ki](price, &eb); tb = now() - t0;
            }} else {{
                t0 = now(); sink += B[ki](price, &eb); tb = now() - t0;
                t0 = now(); sink += strat_0(price, &ea); ta = now() - t0;
            }}
            if (ta < tb) win++; else if (ta == tb) tie++;
            gap_sum += (double)tb - (double)ta;
        }}
        printf("%s{{\\"k\\": %d, \\"A_wins_strict\\": %ld, \\"ties\\": %ld, \\"mean_gap_ns\\": %.3f}}",
               ki ? ", " : "", K[ki], win, tie, gap_sum / R);
    }}
    uint64_t *ts = malloc(sizeof(uint64_t) * J);
    for (long i = 0; i < J; i++) {{ uint64_t t0 = now(); sink += strat_0(price, &ea); ts[i] = now() - t0; }}
    qsort(ts, J, sizeof(uint64_t), cmp);
    printf("], \\"jitter\\": {{\\"runs\\": %ld, \\"p25_ns\\": %llu, \\"median_ns\\": %llu, \\"p75_ns\\": %llu, "
           "\\"p99_ns\\": %llu, \\"p99_9_ns\\": %llu, \\"max_ns\\": %llu}}, \\"sink\\": %llu}}\\n",
           J, ts[J/4], ts[J/2], ts[3*J/4], ts[(long)(J*0.99)], ts[(long)(J*0.999)], ts[J-1], (unsigned long long)sink);
    return 0;
}}
"""


def build(ks, races=200_000, jitter_runs=200_000):
    SRC.write_text(gen_c(ks, races, jitter_runs))
    subprocess.run(["cc", "-O2", "-o", str(BIN), str(SRC)], check=True)


def run():
    out = subprocess.run([str(BIN)], check=True, capture_output=True, text=True).stdout
    d = json.loads(out)
    R = d["races"]
    for r in d["results"]:
        r["A_wins"] = (r["A_wins_strict"] + 0.5 * r["ties"]) / R   # ties = coin flip
        r["tie_share"] = r["ties"] / R
    j = d["jitter"]; j["iqr_ns"] = j["p75_ns"] - j["p25_ns"]
    return d


if __name__ == "__main__":
    build([1, 10, 100, 1000, 10000])
    print(json.dumps(run(), indent=1))
