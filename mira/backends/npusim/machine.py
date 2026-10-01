"""MNPU-1: a small, simulated neural processing unit.

It's modeled on real NPUs (TPU, AMD AIE, Apple ANE-style designs), simplified
enough to read in one sitting:

    ┌──────────────────────────── MNPU-1 ───────────────────────────┐
    │   DMA engine  <──────────────── DRAM (tensors) ───────────────┐  │
    │      │   strided, transposing, and int8-dequantizing transfers │  │
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
whenever data lands in SRAM) and timed (it counts cycles per engine). In fast
mode it skips the arithmetic and only counts cycles.

Timing is a model, not a measurement: MNPU-1 doesn't exist in silicon. The
constants in `Config` are plausible for a small edge NPU and can be changed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import NamedTuple, Optional

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
    dma_latency: int = 200          # cycles from issuing a transfer until its data lands (overlappable)

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

    @classmethod
    def from_overrides(cls, items: dict[str, str]) -> "Config":
        """Config(**{'mxu_dim': '64', ...}) with values converted to the right types."""
        types = {f.name: f.type for f in fields(cls)}
        kw = {}
        for k, v in items.items():
            if k not in types:
                raise ValueError(f"unknown MNPU-1 parameter '{k}' (known: {', '.join(types)})")
            kw[k] = float(v) if types[k] in (float, "float") else int(v)
        return cls(**kw)


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
    """A (possibly strided) rectangle of a DRAM tensor, viewed as 2D [rows, cols].

    Rows r0, r0 + row_stride, ... (nr of them) and columns c0, c0 + col_stride, ...
    (nc of them). Strides let the DMA engine gather im2col patches, permute rows,
    and read every other pixel without any compute.
    """
    storage: str
    view_cols: int
    r0: int
    nr: int
    c0: int
    nc: int
    row_stride: int = 1
    col_stride: int = 1

    def __str__(self) -> str:
        rs = f":{self.row_stride}" if self.row_stride != 1 else ""
        cs = f":{self.col_stride}" if self.col_stride != 1 else ""
        return (f"{self.storage}[{self.r0}:{self.r0 + self.nr * self.row_stride}{rs}, "
                f"{self.c0}:{self.c0 + self.nc * self.col_stride}{cs}]")


# broadcast modes for vector-unit sources (how a source tile maps onto the dst tile)
FULL, ROW, COL, SCALAR = "full", "row", "col", "scalar"


# ------------------------------------------------------------------ instructions

@dataclass
class Instr:
    engine: str = field(init=False, default="")
    comment: str = field(init=False, default="")
    op: str = field(init=False, default="")      # the IR op this instruction implements (for tools)


@dataclass
class Load(Instr):
    """DRAM -> SRAM. Optionally transposes, and optionally dequantizes int8 data by a scale vector."""
    dst: Tile
    src: DramRegion
    transpose: bool = False
    scale: Optional[DramRegion] = None
    scale_axis: int = 1          # scale per dst column (1) or per dst row (0)

    def __post_init__(self):
        self.engine = "dma"

    def __str__(self):
        extra = " (transpose)" if self.transpose else ""
        extra += f" * {self.scale}" if self.scale else ""
        return f"dma.load    {self.dst} <- {self.src}{extra}"


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
    """dst = fn(*srcs), elementwise, with per-source broadcast (full / row / col / scalar)."""
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
    """Row-wise vector kernels: softmax / layernorm (dst like src), rsum / rmean / rmax (dst is rows x 1)."""
    fn: str
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


def _broadcast(v: np.ndarray, mode: str, shape: tuple[int, int]) -> np.ndarray:
    if mode == FULL:
        return v
    if mode == ROW:
        return np.broadcast_to(v[:1], shape)
    if mode == COL:
        return np.broadcast_to(v[:, :1], shape)
    return np.broadcast_to(v[:1, :1], shape)


class Machine:
    def __init__(self, cfg: Config = Config(), functional: bool = True):
        self.cfg = cfg
        self.functional = functional
        self.sram = np.zeros(cfg.sram_elems if functional else 0, dtype=np.float16)
        self.acc = np.zeros(cfg.acc_elems if functional else 0, dtype=np.float32)
        self.dram: dict[str, np.ndarray] = {}   # flat storage per tensor (fp16, or int8 for quantized weights)
        self.itemsize: dict[str, int] = {}

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
        return flat.reshape(-1, r.view_cols)[r.r0:r.r0 + r.nr * r.row_stride:r.row_stride,
                                              r.c0:r.c0 + r.nc * r.col_stride:r.col_stride]

    def banks(self, t: Tile) -> set[str]:
        bank = self.cfg.sram_bank_elems if t.space == "sram" else self.cfg.acc_bank_elems
        return {f"{t.space}{b}" for b in range(t.addr // bank, (t.addr + t.size - 1) // bank + 1)}

    def dma_cycles(self, r: DramRegion, itemsize: int, element_wise: bool) -> tuple[int, int]:
        """(cycles, bytes) for moving region r. Contiguous rows stream at full bandwidth;
        strided columns or transposes move one element per cycle."""
        nbytes = r.nr * r.nc * itemsize
        if element_wise or r.col_stride != 1:
            cycles = r.nr * r.nc
        elif r.nc == r.view_cols and r.row_stride == 1:
            cycles = math.ceil(nbytes / self.cfg.dma_bytes_per_cycle)
        else:   # one burst per row
            cycles = r.nr * math.ceil(r.nc * itemsize / self.cfg.dma_bytes_per_cycle)
        return cycles, nbytes

    # ----- semantics + cost of one instruction

    def execute(self, ins: Instr) -> tuple[int, set[str], set[str]]:
        """Run `ins` (functionally unless in fast mode); return (busy cycles, resources read, resources written).

        For DMA, busy cycles are the transfer time only; the startup latency is added by
        `run`, because the engine keeps several transfers in flight and overlaps their latency.
        """
        cfg, fn = self.cfg, self.functional
        if isinstance(ins, Load):
            size = self.itemsize.get(ins.src.storage, 2)
            cycles, nbytes = self.dma_cycles(ins.src, size, ins.transpose)
            reads = {f"dram:{ins.src.storage}"}
            if fn:
                data = self.dram_view(ins.src).astype(np.float32)
                if ins.transpose:
                    data = data.T
                if ins.scale is not None:
                    s = self.dram_view(ins.scale).astype(np.float32).reshape(-1)
                    data = data * (s[None, :] if ins.scale_axis == 1 else s[:, None])
                self.write(ins.dst, data.reshape(ins.dst.rows, ins.dst.cols))
            if ins.scale is not None:
                c2, b2 = self.dma_cycles(ins.scale, 2, False)
                cycles, nbytes = cycles + c2, nbytes + b2
                reads.add(f"dram:{ins.scale.storage}")
            self.stats.dram_bytes += nbytes
            return cycles, reads, self.banks(ins.dst)
        if isinstance(ins, Store):
            if fn:
                self.dram_view(ins.dst)[...] = self.read(ins.src).astype(np.float16)
            cycles, nbytes = self.dma_cycles(ins.dst, 2, False)
            self.stats.dram_bytes += nbytes
            return cycles, self.banks(ins.src), {f"dram:{ins.dst.storage}"}
        if isinstance(ins, MatMul):
            m, k = ins.a.rows, ins.a.cols
            n = ins.b.cols
            if fn:
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
            if fn:
                args = [_broadcast(self.read(t), mode, (rows, cols)) for t, mode in ins.srcs]
                if ins.fn == "copy":
                    out = args[0]
                elif ins.fn == "where":
                    out = np.where(args[0] != 0, args[1], args[2])
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
            passes = {"softmax": 7, "layernorm": 9, "rsum": 1, "rmean": 2, "rmax": 1}[ins.fn]
            if fn:
                x = self.read(ins.src)
                if ins.fn == "softmax":
                    e = np.exp(x - x.max(axis=1, keepdims=True))
                    out = e / e.sum(axis=1, keepdims=True)
                elif ins.fn == "layernorm":
                    g, b = (self.read(t)[:1] for t in ins.extra)
                    mu = x.mean(axis=1, keepdims=True)
                    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
                    out = (x - mu) / np.sqrt(var + ins.eps) * g + b
                else:
                    out = {"rsum": np.sum, "rmean": np.mean, "rmax": np.max}[ins.fn](x, axis=1, keepdims=True)
                self.write(ins.dst, out)
            cycles = passes * math.ceil(ins.src.size / cfg.vpu_lanes) + 16
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
            latency = self.cfg.dma_latency if ins.engine == "dma" else 0
            engine_free[ins.engine] = start + cycles     # the engine can issue its next request now
            end = start + cycles + latency               # ...but this one's data lands later
            for r in reads:
                last_read[r] = max(last_read.get(r, 0), end)
            for w in writes:
                last_write[w] = end
            self.stats.busy[ins.engine] += cycles
            self.stats.instrs[ins.engine] += 1
            end_all = max(end_all, end)
            if trace is not None:
                trace.append(TraceEntry(ins, start, start + cycles, end,
                                        frozenset(r for r in reads if not r.startswith("dram:")),
                                        frozenset(w for w in writes if not w.startswith("dram:"))))
        self.stats.cycles = end_all
        return self.stats


class TraceEntry(NamedTuple):
    """One executed instruction: when it started, when its engine was free again, when its result
    was ready (later than busy_end for DMA, whose latency overlaps other work), and which on-chip
    banks it read and wrote."""
    ins: Instr
    start: int
    busy_end: int
    end: int
    reads: frozenset
    writes: frozenset


def format_timeline(trace: list, width: int = 72, limit: int = 40) -> str:
    """ASCII Gantt chart of the first `limit` instructions, one row per instruction."""
    if not trace:
        return ""
    rows = trace[:limit]
    t_end = max(t.end for t in rows) or 1
    scale = width / t_end
    lines = [f"{'engine':<6} {'start':>8} {'end':>8}  timeline (0 .. {t_end:,} cycles)"]
    for ins, s, _, e, _, _ in rows:
        a = int(s * scale)
        b = max(a + 1, int(e * scale))
        mark = {"dma": "=", "mxu": "#", "vpu": "~"}[ins.engine]
        lines.append(f"{ins.engine:<6} {s:>8,} {e:>8,}  |{' ' * a}{mark * (b - a)}{' ' * (width - b)}|  "
                     f"{str(ins).split()[0]}")
    if len(trace) > limit:
        lines.append(f"... {len(trace) - limit:,} more instructions")
    return "\n".join(lines)
