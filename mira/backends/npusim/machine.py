"""MNPU-1: a small, simulated neural processing unit.

It's modeled on real NPUs (TPU, AMD AIE, Apple ANE-style designs), simplified
enough to read in one sitting:

    ┌──────────────────────────── MNPU-1 ───────────────────────────┐
    │   DMA engine  <──────────────── DRAM (tensors, fp16) ────────┐  │
    │      │                                                     │  │
    │      v                                                     │  │
    │   SRAM scratchpad  (512 KiB, fp16, 32 banks x 16 KiB)  ───────┘  │
    │      │        ^                                                │
    │      v        │                                                │
    │   MXU: 32x32 systolic array ──> Accumulators (256 KiB, fp32)    │
    │   VPU: 32-lane vector unit  <──> (epilogues, activations, ...)  │
    └────────────────────────────────────────────────────────────────┘

Programs are a single instruction stream. Each instruction goes to one of three
engines (dma, mxu, vpu). Each engine executes its own instructions in order,
but the engines run *concurrently*. A hardware scoreboard tracks every SRAM /
accumulator bank and DRAM tensor, and delays an instruction until:
  * RAW: everything it reads has been written, and
  * WAR/WAW: nobody is still reading or writing what it's about to overwrite.
That's what makes double buffering work: the DMA engine fills buffer 1 while
the MXU is busy with buffer 0.

The simulator is both functional (it produces real numbers, rounding to fp16
whenever data lands in SRAM) and timed (it counts cycles per engine).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from ...ops import BINARY, UNARY


@dataclass(frozen=True)
class Config:
    clock_ghz: float = 1.0
    sram_bytes: int = 512 * 1024
    acc_bytes: int = 256 * 1024
    bank_bytes: int = 16 * 1024
    mxu_dim: int = 32               # 32x32 systolic array = 1024 fp16 MACs/cycle
    vpu_lanes: int = 32
    dma_bytes_per_cycle: int = 64   # 64 GB/s at 1 GHz
    dma_latency: int = 200          # cycles to start a transfer

    @property
    def sram_elems(self) -> int:
        return self.sram_bytes // 2

    @property
    def acc_elems(self) -> int:
        return self.acc_bytes // 4

    @property
    def sram_bank_elems(self) -> int:
        return self.bank_bytes // 2

    @property
    def acc_bank_elems(self) -> int:
        return self.bank_bytes // 4


# ------------------------------------------------------------------ operands

@dataclass(frozen=True)
class Tile:
    """A contiguous row-major rows x cols block in SRAM (fp16) or the accumulators (fp32)."""
    space: str        # "sram" | "acc"
    addr: int         # element address
    rows: int
    cols: int

    @property
    def size(self) -> int:
        return self.rows * self.cols

    def __str__(self) -> str:
        return f"{self.space}[{self.addr:#07x}]({self.rows}x{self.cols})"


@dataclass(frozen=True)
class DramRegion:
    """A rectangle of a DRAM tensor, viewed as 2D [rows, cols]."""
    storage: str
    view_cols: int
    r0: int
    nr: int
    c0: int
    nc: int

    def __str__(self) -> str:
        return f"{self.storage}[{self.r0}:{self.r0 + self.nr}, {self.c0}:{self.c0 + self.nc}]"


# broadcast modes for vector-unit sources
FULL, ROW, SCALAR = "full", "row", "scalar"


# ------------------------------------------------------------------ instructions

@dataclass
class Instr:
    engine: str = field(init=False, default="")
    comment: str = field(init=False, default="")


@dataclass
class Load(Instr):
    dst: Tile
    src: DramRegion

    def __post_init__(self):
        self.engine = "dma"

    def __str__(self):
        return f"dma.load    {self.dst} <- {self.src}"


@dataclass
class Store(Instr):
    dst: DramRegion
    src: Tile

    def __post_init__(self):
        self.engine = "dma"

    def __str__(self):
        return f"dma.store   {self.dst} <- {self.src}"


@dataclass
class MatMul(Instr):
    acc: Tile
    a: Tile
    b: Tile
    accumulate: bool

    def __post_init__(self):
        self.engine = "mxu"

    def __str__(self):
        return f"mxu.matmul  {self.acc} {'+=' if self.accumulate else '='} {self.a} @ {self.b}"


@dataclass
class Vec(Instr):
    """dst = fn(*srcs), elementwise, with per-source broadcast (full / row / scalar)."""
    fn: str
    dst: Tile
    srcs: list[tuple[Tile, str]]

    def __post_init__(self):
        self.engine = "vpu"

    def __str__(self):
        srcs = ", ".join(f"{t}" + ("" if m == FULL else f".{m}") for t, m in self.srcs)
        return f"vpu.{self.fn:<8}{self.dst} <- {srcs}"


@dataclass
class RowOp(Instr):
    """Row-wise fused vector kernels: softmax over each row; layernorm with gamma/beta rows."""
    fn: str                 # "softmax" | "layernorm"
    dst: Tile
    src: Tile
    extra: list[Tile]
    eps: float = 0.0

    def __post_init__(self):
        self.engine = "vpu"

    def __str__(self):
        extra = "".join(f", {t}.row" for t in self.extra)
        return f"vpu.{self.fn:<8}{self.dst} <- {self.src}{extra}"


Program = list[Instr]
SPECIAL = {"exp", "log", "sqrt", "tanh", "sigmoid", "gelu", "div", "pow"}


# ------------------------------------------------------------------ the machine

@dataclass
class Stats:
    cycles: int = 0
    busy: dict[str, int] = field(default_factory=lambda: {"dma": 0, "mxu": 0, "vpu": 0})
    instrs: dict[str, int] = field(default_factory=lambda: {"dma": 0, "mxu": 0, "vpu": 0})
    dram_bytes: int = 0
    macs: int = 0

    def summary(self, cfg: Config) -> str:
        us = self.cycles / (cfg.clock_ghz * 1e3)
        peak = cfg.mxu_dim ** 2
        lines = [f"cycles: {self.cycles:,}  ({us:.1f} us at {cfg.clock_ghz} GHz)"]
        for e in ("dma", "mxu", "vpu"):
            util = 100 * self.busy[e] / self.cycles if self.cycles else 0
            lines.append(f"  {e}: {self.instrs[e]:6,} instrs, busy {self.busy[e]:>10,} cycles ({util:5.1f}%)")
        if self.cycles:
            lines.append(f"  MACs {self.macs:,}  -> {self.macs / self.cycles:,.0f} MAC/cycle "
                         f"({100 * self.macs / self.cycles / peak:.1f}% of {peak} peak)")
        lines.append(f"  DRAM traffic {self.dram_bytes / 1024:,.1f} KiB")
        return "\n".join(lines)


class Machine:
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.sram = np.zeros(cfg.sram_elems, dtype=np.float16)
        self.acc = np.zeros(cfg.acc_elems, dtype=np.float32)
        self.dram: dict[str, np.ndarray] = {}   # flat fp16 storage per tensor

    # ----- memory access helpers

    def _mem(self, t: Tile) -> np.ndarray:
        mem = self.sram if t.space == "sram" else self.acc
        if t.addr < 0 or t.addr + t.size > mem.size:
            raise RuntimeError(f"out-of-bounds {t.space} access: {t}")
        return mem

    def read(self, t: Tile) -> np.ndarray:
        return self._mem(t)[t.addr:t.addr + t.size].reshape(t.rows, t.cols).astype(np.float32)

    def write(self, t: Tile, value: np.ndarray) -> None:
        mem = self._mem(t)
        mem[t.addr:t.addr + t.size] = np.broadcast_to(value, (t.rows, t.cols)).ravel().astype(mem.dtype)

    def dram_view(self, r: DramRegion) -> np.ndarray:
        flat = self.dram[r.storage]
        return flat.reshape(-1, r.view_cols)[r.r0:r.r0 + r.nr, r.c0:r.c0 + r.nc]

    def banks(self, t: Tile) -> set[str]:
        bank = self.cfg.sram_bank_elems if t.space == "sram" else self.cfg.acc_bank_elems
        return {f"{t.space}{b}" for b in range(t.addr // bank, (t.addr + t.size - 1) // bank + 1)}

    # ----- semantics + cost of one instruction

    def execute(self, ins: Instr) -> tuple[int, set[str], set[str]]:
        """Run `ins` functionally; return (cycles, resources read, resources written)."""
        cfg = self.cfg
        if isinstance(ins, Load):
            data = self.dram_view(ins.src)
            self.write(ins.dst, data.reshape(ins.dst.rows, ins.dst.cols))
            nbytes = data.size * 2
            self.stats.dram_bytes += nbytes
            return (cfg.dma_latency + math.ceil(nbytes / cfg.dma_bytes_per_cycle),
                    {f"dram:{ins.src.storage}"}, self.banks(ins.dst))
        if isinstance(ins, Store):
            self.dram_view(ins.dst)[...] = self.read(ins.src).astype(np.float16)
            nbytes = ins.src.size * 2
            self.stats.dram_bytes += nbytes
            return (cfg.dma_latency + math.ceil(nbytes / cfg.dma_bytes_per_cycle),
                    self.banks(ins.src), {f"dram:{ins.dst.storage}"})
        if isinstance(ins, MatMul):
            m, k = ins.a.rows, ins.a.cols
            n = ins.b.cols
            prod = self.read(ins.a) @ self.read(ins.b)
            self.write(ins.acc, self.read(ins.acc) + prod if ins.accumulate else prod)
            self.stats.macs += m * k * n
            d = cfg.mxu_dim
            # weight-stationary systolic array: load a d x d block of B, stream m rows of A through it
            cycles = math.ceil(k / d) * math.ceil(n / d) * (m + d) + 16
            reads = self.banks(ins.a) | self.banks(ins.b) | (self.banks(ins.acc) if ins.accumulate else set())
            return cycles, reads, self.banks(ins.acc)
        if isinstance(ins, Vec):
            rows, cols = ins.dst.rows, ins.dst.cols
            args = []
            for t, mode in ins.srcs:
                v = self.read(t)
                args.append(v if mode == FULL else np.broadcast_to(v[:1] if mode == ROW else v[:1, :1], (rows, cols)))
            if ins.fn == "copy":
                out = args[0]
            elif ins.fn in UNARY:
                out = UNARY[ins.fn](args[0])
            else:
                out = BINARY[ins.fn](args[0], args[1])
            self.write(ins.dst, out)
            cost = 4 if ins.fn in SPECIAL else 1
            cycles = cost * math.ceil(rows * cols / cfg.vpu_lanes) + 8
            reads = set().union(*(self.banks(t) for t, _ in ins.srcs))
            return cycles, reads, self.banks(ins.dst)
        if isinstance(ins, RowOp):
            x = self.read(ins.src)
            if ins.fn == "softmax":
                e = np.exp(x - x.max(axis=1, keepdims=True))
                out = e / e.sum(axis=1, keepdims=True)
                passes = 3 + 4   # max, sub/exp (special), sum, div
            else:
                g, b = (self.read(t)[:1] for t in ins.extra)
                mu = x.mean(axis=1, keepdims=True)
                var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
                out = (x - mu) / np.sqrt(var + ins.eps) * g + b
                passes = 5 + 4
            self.write(ins.dst, out)
            cycles = passes * math.ceil(x.size / cfg.vpu_lanes) + 16
            reads = self.banks(ins.src).union(*(self.banks(t) for t in ins.extra))
            return cycles, reads, self.banks(ins.dst)
        raise TypeError(ins)

    # ----- run a program

    def run(self, program: Program, trace: Optional[list] = None) -> Stats:
        self.stats = Stats()
        engine_free = {"dma": 0, "mxu": 0, "vpu": 0}
        last_write: dict[str, int] = {}   # resource -> cycle its latest write completes
        last_read: dict[str, int] = {}    # resource -> cycle its latest read completes
        end_all = 0
        for ins in program:
            cycles, reads, writes = self.execute(ins)
            start = engine_free[ins.engine]
            for r in reads:                              # RAW
                start = max(start, last_write.get(r, 0))
            for w in writes:                             # WAW + WAR
                start = max(start, last_write.get(w, 0), last_read.get(w, 0))
            end = start + cycles
            engine_free[ins.engine] = end
            for r in reads:
                last_read[r] = max(last_read.get(r, 0), end)
            for w in writes:
                last_write[w] = end
            self.stats.busy[ins.engine] += cycles
            self.stats.instrs[ins.engine] += 1
            end_all = max(end_all, end)
            if trace is not None:
                trace.append((ins, start, end))
        self.stats.cycles = end_all
        return self.stats


def format_timeline(trace: list, width: int = 72, limit: int = 40) -> str:
    """ASCII Gantt chart of the first `limit` instructions, one row per instruction."""
    if not trace:
        return ""
    rows = trace[:limit]
    t_end = max(e for _, _, e in rows) or 1
    scale = width / t_end
    lines = [f"{'engine':<6} {'start':>8} {'end':>8}  timeline (0 .. {t_end:,} cycles)"]
    for ins, s, e in rows:
        a = int(s * scale)
        b = max(a + 1, int(e * scale))
        mark = {"dma": "=", "mxu": "#", "vpu": "~"}[ins.engine]
        lines.append(f"{ins.engine:<6} {s:>8,} {e:>8,}  |{' ' * a}{mark * (b - a)}{' ' * (width - b)}|  "
                     f"{str(ins).split()[0]}")
    if len(trace) > limit:
        lines.append(f"... {len(trace) - limit:,} more instructions")
    return "\n".join(lines)


Operand = Union[Tile, DramRegion]
