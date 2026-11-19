"""A tiny eBPF-flavoured register VM with Cuttlefish-style *virtual time*.

Cuttlefish (Yu, Goyal, Marinos, Liu - AFT 2025) runs each trader's strategy as eBPF
bytecode, charges every instruction a fixed, public cost, and ranks trades by the
virtual time of the submit_trade call instead of by wall-clock time. This file is a
toy re-implementation of that idea for a proof-of-concept - NOT the paper's system
(no verifier, no JIT, no x64 codegen, no networking).

Program format: a list of tuples, e.g.
    [("LOADF", 0, "price"), ("CONST", 1, 100), ("JGT", 0, 1, "sell"), ...,
     ("LABEL", "sell"), ("SUBMIT", 2), ("RET",)]
Registers r0..r7 hold ints/floats. Labels are resolved by `assemble`.
"""
import time

# Fixed, public cost per opcode, in virtual-time units (same on every machine).
COST = {
    "LOADF": 1,   # load a field of the market-data event into a register
    "CONST": 1,   # load an immediate
    "MOV": 1,
    "ADD": 1, "SUB": 1, "MUL": 1,
    "DIV": 4,     # dearer ALU op
    "JMP": 1, "JLT": 1, "JGT": 1, "JEQ": 1,   # compare-and-branch
    "MGET": 5, "MPUT": 5,                     # key-value map (eBPF-map-like state)
    "NOP": 1,
    "SUBMIT": 1,  # submit_trade(): trade is stamped with the virtual time of this call
    "RET": 1,
}


def assemble(prog):
    """Resolve ('LABEL', name) pseudo-ops into jump targets. Returns a list of tuples."""
    labels, out = {}, []
    for ins in prog:
        if ins[0] == "LABEL":
            labels[ins[1]] = len(out)
        else:
            if ins[0] not in COST:
                raise ValueError(f"unknown opcode {ins[0]}")
            out.append(ins)
    resolved = []
    for ins in out:
        op = ins[0]
        if op == "JMP":
            resolved.append((op, labels[ins[1]]))
        elif op in ("JLT", "JGT", "JEQ"):
            resolved.append((op, ins[1], ins[2], labels[ins[3]]))
        else:
            resolved.append(ins)
    return resolved


class Participant:
    """One trader: a program, private map state, and a virtual clock."""

    def __init__(self, name, program):
        self.name = name
        self.code = assemble(program)
        self.state = {}          # eBPF-map-like key/value store, persists across events
        self.vclock = 0          # virtual time
        self.trades = []         # (virtual_time, wall_ns, name, payload)

    def on_event(self, event, release_vt):
        """Handle one market-data event released at virtual time `release_vt`.
        Rule from the paper: clock = max(D(x), clock); each instruction adds its cost."""
        self.vclock = max(release_vt, self.vclock)
        regs = [0] * 8
        code, state, pc = self.code, self.state, 0
        vt = self.vclock
        while True:
            ins = code[pc]
            op = ins[0]
            vt += COST[op]
            pc += 1
            if op == "LOADF":
                regs[ins[1]] = event[ins[2]]
            elif op == "CONST":
                regs[ins[1]] = ins[2]
            elif op == "MOV":
                regs[ins[1]] = regs[ins[2]]
            elif op == "ADD":
                regs[ins[1]] = regs[ins[2]] + regs[ins[3]]
            elif op == "SUB":
                regs[ins[1]] = regs[ins[2]] - regs[ins[3]]
            elif op == "MUL":
                regs[ins[1]] = regs[ins[2]] * regs[ins[3]]
            elif op == "DIV":
                regs[ins[1]] = regs[ins[2]] / regs[ins[3]] if regs[ins[3]] else 0
            elif op == "JMP":
                pc = ins[1]
            elif op == "JLT":
                if regs[ins[1]] < regs[ins[2]]:
                    pc = ins[3]
            elif op == "JGT":
                if regs[ins[1]] > regs[ins[2]]:
                    pc = ins[3]
            elif op == "JEQ":
                if regs[ins[1]] == regs[ins[2]]:
                    pc = ins[3]
            elif op == "MGET":
                regs[ins[1]] = state.get(ins[2], 0)
            elif op == "MPUT":
                state[ins[1]] = regs[ins[2]]
            elif op == "NOP":
                pass
            elif op == "SUBMIT":
                # trade stamped with the virtual time of this call (and, for comparison,
                # the real wall-clock time - which is what a stopwatch would see)
                self.trades.append((vt, time.perf_counter_ns(), self.name, regs[ins[1]]))
            elif op == "RET":
                break
        self.vclock = vt


# ---------------------------------------------------------------- example strategies
def threshold_strategy():
    """Paper's Fig. 7 style: sell above 100, buy below 10, save the last price."""
    return [
        ("LOADF", 0, "price"),
        ("CONST", 1, 100), ("JGT", 0, 1, "sell"),
        ("CONST", 1, 10), ("JLT", 0, 1, "buy"),
        ("JMP", "save"),
        ("LABEL", "sell"), ("CONST", 2, 1), ("SUBMIT", 2), ("JMP", "save"),
        ("LABEL", "buy"), ("CONST", 2, 2), ("SUBMIT", 2),
        ("LABEL", "save"), ("MPUT", "last", 0),
        ("RET",),
    ]


def moving_average_strategy(extra_nops=0):
    """Always trades: buy (2) if price > EWMA else sell (1). EWMA kept in map state.
    `extra_nops` inserts exactly that many extra instructions BEFORE submit_trade, so
    two traders can differ by an exact, known amount of work."""
    prog = [
        ("LOADF", 0, "price"),
        ("MGET", 1, "ewma"),          # r1 = ewma
        ("SUB", 2, 0, 1),             # r2 = price - ewma
        ("CONST", 3, 8), ("DIV", 2, 2, 3),
        ("ADD", 1, 1, 2),             # ewma += (price - ewma) / 8
        ("MPUT", "ewma", 1),
    ]
    prog += [("NOP",)] * extra_nops
    prog += [
        ("JGT", 0, 1, "buy"),
        ("CONST", 4, 1), ("SUBMIT", 4), ("RET",),
        ("LABEL", "buy"), ("CONST", 4, 2), ("SUBMIT", 4), ("RET",),
    ]
    return prog
