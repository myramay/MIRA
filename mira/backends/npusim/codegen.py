"""Code generation for MNPU-1: IR subgraph -> instruction stream.

For each op this module:
  1. picks tile sizes that fit on-chip memory (a small cost model: minimize DRAM traffic),
  2. plans SRAM / accumulator buffers (bank-aligned bump allocation),
  3. emits DMA loads, compute, and stores, alternating between two buffer
     "slots" (double buffering) so loads of tile i+1 overlap compute on tile i.

Data layout: every tensor lives in DRAM as flat fp16, viewed as 2D
[prod(leading dims), last dim]. `reshape` is free: the output aliases the
input's storage with a different 2D view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ... import ir
from ...ops import BINARY, UNARY
from .machine import (FULL, ROW, SCALAR, Config, DramRegion, Instr, Load, MatMul, Program, RowOp, Store, Tile,
                      Vec)

MAX_BUF = 32 * 1024   # elements per elementwise buffer (4 SRAM banks)


# ------------------------------------------------------------------ what the NPU supports

def view2d(shape: tuple[int, ...]) -> tuple[int, int]:
    if not shape:
        return 1, 1
    return int(np.prod(shape[:-1], dtype=np.int64)), shape[-1]


def bcast_mode(operand: tuple[int, ...], result: tuple[int, ...]) -> Optional[str]:
    """How `operand` broadcasts onto `result` in the 2D view, if the NPU can do it."""
    if operand == result:
        return FULL
    if int(np.prod(operand, dtype=np.int64)) == 1:
        return SCALAR
    if operand and operand[-1] == result[-1] and all(d == 1 for d in operand[:-1]):
        return ROW
    return None


def check(op: ir.Op, cfg: Config) -> Optional[str]:
    """None if MNPU-1 can run `op`, else a human-readable reason."""
    if op.result.type.dtype != "f16" or any(v.type.dtype != "f16" for v in op.inputs):
        return "MNPU-1 only computes in fp16 (compile with precision f16)"
    k = op.kind
    out = op.result.type.shape
    if k == "matmul":
        x, w = op.inputs[0].type.shape, op.inputs[1].type.shape
        if len(w) != 2 and x[:-2] != w[:-2]:
            return "matmul: batched operands must have identical batch dims (no broadcasting)"
        for fn, idx, _ in op.attrs.get("epilogue", ()):
            if idx is not None and bcast_mode(op.inputs[idx].type.shape, out) is None:
                return f"matmul epilogue '{fn}': operand broadcast pattern not supported"
        return None
    if k in UNARY:
        return None
    if k in BINARY:
        for v in op.inputs:
            if bcast_mode(v.type.shape, out) is None:
                return f"{k}: broadcast of {v.type} to {op.result.type} needs a strided DMA MNPU-1 lacks"
        return None
    if k == "reshape":
        return None
    if k in ("softmax", "layernorm"):
        if k == "softmax" and op.attrs["axis"] % max(len(out), 1) != len(out) - 1:
            return "softmax: the vector unit only reduces along the last axis"
        if view2d(out)[1] > MAX_BUF:
            return f"{k}: a row of {view2d(out)[1]} elements doesn't fit a vector buffer"
        return None
    reasons = {
        "conv2d": "conv2d: needs im2col lowering (not implemented for MNPU-1)",
        "maxpool2d": "maxpool2d: no windowed-reduction unit",
        "transpose": "transpose: needs a strided/transposing DMA",
        "concat": "concat: not implemented",
        "cast": "cast: MNPU-1 is fp16-only",
    }
    if k.startswith("reduce_"):
        return f"{k}: no general reduction unit (only row softmax/layernorm)"
    return reasons.get(k, f"{k}: not supported by MNPU-1")


# ------------------------------------------------------------------ memory planning

class Allocator:
    """Bump allocator that aligns every buffer to a bank boundary.

    Bank alignment means two buffers never share a bank, so the scoreboard never
    sees a false dependency between them.
    """

    def __init__(self, space: str, capacity: int, bank: int):
        self.space, self.capacity, self.bank = space, capacity, bank
        self.top = 0

    def alloc(self, rows: int, cols: int) -> Tile:
        size = math.ceil(rows * cols / self.bank) * self.bank
        if self.top + size > self.capacity:
            raise MemoryError(f"{self.space} exhausted")
        t = Tile(self.space, self.top, rows, cols)
        self.top += size
        return t

    def banks_for(self, elems: int) -> int:
        return math.ceil(elems / self.bank)


def tile_candidates(d: int) -> list[int]:
    c = {min(d, 1 << p) for p in range(5, 13)}
    return sorted(c)


def choose_matmul_tiles(M: int, K: int, N: int, n_epi: int, slots: int, cfg: Config) -> tuple[int, int, int]:
    """Pick (tm, tk, tn) minimizing DRAM traffic subject to SRAM/accumulator capacity.

    DRAM traffic (in elements) for an output-stationary loop nest is roughly
        A is re-read once per column of tiles:  M*K * ceil(N/tn)
        B is re-read once per row of tiles:     K*N * ceil(M/tm)
    so bigger tiles = less traffic, until they stop fitting on chip.
    """
    sb, ab = cfg.sram_bank_elems, cfg.acc_bank_elems
    best = None
    for tm in tile_candidates(M):
        for tn in tile_candidates(N):
            for tk in tile_candidates(K):
                sram_banks = slots * (math.ceil(tm * tk / sb) + math.ceil(tk * tn / sb)
                                      + (n_epi + 1) * math.ceil(tm * tn / sb))
                acc_banks = slots * math.ceil(tm * tn / ab)
                if sram_banks * sb > cfg.sram_elems or acc_banks * ab > cfg.acc_elems:
                    continue
                traffic = M * K * math.ceil(N / tn) + K * N * math.ceil(M / tm)
                score = (traffic, math.ceil(K / tk), -tm * tn)
                if best is None or score < best[0]:
                    best = (score, (tm, tk, tn))
    if best is None:
        raise MemoryError("no matmul tiling fits on chip")
    return best[1]


# ------------------------------------------------------------------ code generation

@dataclass
class Compiled:
    program: Program
    storage: dict[ir.Value, tuple[str, tuple[int, int]]]   # value -> (storage name, 2D view)
    sizes: dict[str, int]                                   # storage name -> elements
    consts: dict[str, np.ndarray]                           # storage preloaded at load time
    notes: list[str] = field(default_factory=list)


class CodeGen:
    def __init__(self, cfg: Config, double_buffer: bool = True):
        self.cfg = cfg
        self.slots = 2 if double_buffer else 1
        self.prog: Program = []
        self.storage: dict[ir.Value, tuple[str, tuple[int, int]]] = {}
        self.sizes: dict[str, int] = {}
        self.consts: dict[str, np.ndarray] = {}
        self.notes: list[str] = []

    # ----- DRAM

    def place(self, v: ir.Value, alias_of: Optional[ir.Value] = None) -> None:
        if alias_of is not None:
            self.storage[v] = (self.storage[alias_of][0], view2d(v.type.shape))
            return
        name = f"t{v.id}" if v.name is None else f"in_{v.name}"
        self.storage[v] = (name, view2d(v.type.shape))
        self.sizes[name] = v.type.numel

    def region(self, v: ir.Value, r0: int, nr: int, c0: int, nc: int) -> DramRegion:
        name, (_, cols) = self.storage[v]
        return DramRegion(name, cols, r0, nr, c0, nc)

    def emit(self, ins: Instr, comment: str = "") -> None:
        ins.comment = comment
        self.prog.append(ins)

    def fresh(self) -> tuple[Allocator, Allocator]:
        c = self.cfg
        return (Allocator("sram", c.sram_elems, c.sram_bank_elems),
                Allocator("acc", c.acc_elems, c.acc_bank_elems))

    def operand_region(self, v: ir.Value, mode: str, r0: int, nr: int, c0: int, nc: int) -> DramRegion:
        if mode == FULL:
            return self.region(v, r0, nr, c0, nc)
        if mode == ROW:
            return self.region(v, 0, 1, c0, nc)
        return self.region(v, 0, 1, 0, 1)

    # ----- graph

    def compile(self, g: ir.Graph) -> Compiled:
        for v in g.inputs:
            self.place(v)
        for op in g.ops:
            if op.is_const:
                self.place(op.result)
                self.consts[self.storage[op.result][0]] = op.attrs["value"].astype(np.float16).ravel()
                continue
            if op.kind == "reshape":
                self.place(op.result, alias_of=op.inputs[0])
                continue
            self.place(op.result)
            if op.kind == "matmul":
                self.matmul(op)
            elif op.kind in ("softmax", "layernorm"):
                self.rowop(op)
            else:
                self.elementwise(op)
        return Compiled(self.prog, self.storage, self.sizes, self.consts, self.notes)

    # ----- kernels

    def matmul(self, op: ir.Op) -> None:
        x, w = op.inputs[0], op.inputs[1]
        out = op.result
        epi = op.attrs.get("epilogue", ())
        K, N = w.type.shape[-2], w.type.shape[-1]
        if w.type.rank == 2:        # shared weights: fold all leading dims of x into M
            batches, M = 1, int(np.prod(x.type.shape[:-1], dtype=np.int64))
        else:                       # true batched matmul: loop over the batch
            batches, M = int(np.prod(x.type.shape[:-2], dtype=np.int64)), x.type.shape[-2]
        n_epi = sum(1 for _, idx, _ in epi if idx is not None)
        tm, tk, tn = choose_matmul_tiles(M, K, N, n_epi, self.slots, self.cfg)
        self.notes.append(f"matmul {out!r}: M={M} K={K} N={N} x{batches} -> tiles tm={tm} tk={tk} tn={tn}"
                          + (f", epilogue {[s[0] for s in epi]}" if epi else ""))

        sram, acc = self.fresh()
        A = [sram.alloc(tm, tk) for _ in range(self.slots)]
        B = [sram.alloc(tk, tn) for _ in range(self.slots)]
        E = [[sram.alloc(tm, tn) for _ in range(n_epi)] for _ in range(self.slots)]
        O = [sram.alloc(tm, tn) for _ in range(self.slots)]
        C = [acc.alloc(tm, tn) for _ in range(self.slots)]

        ab = tile_no = 0
        for b in range(batches):
            for i in range(0, M, tm):
                mm = min(tm, M - i)
                for j in range(0, N, tn):
                    nn = min(tn, N - j)
                    s = tile_no % self.slots
                    c = Tile("acc", C[s].addr, mm, nn)
                    for kk in range(0, K, tk):
                        kn = min(tk, K - kk)
                        t = ab % self.slots
                        a_t = Tile("sram", A[t].addr, mm, kn)
                        b_t = Tile("sram", B[t].addr, kn, nn)
                        self.emit(Load(a_t, self.region(x, b * M + i, mm, kk, kn)),
                                  f"A tile ({i},{kk}) slot {t}")
                        w_row = (b * K if w.type.rank > 2 else 0) + kk
                        self.emit(Load(b_t, self.region(w, w_row, kn, j, nn)), f"B tile ({kk},{j}) slot {t}")
                        self.emit(MatMul(c, a_t, b_t, accumulate=kk > 0))
                        ab += 1
                    e_i = 0
                    for fn, idx, swapped in epi:
                        if idx is None:
                            self.emit(Vec(fn, c, [(c, FULL)]), "epilogue")
                            continue
                        other = op.inputs[idx]
                        mode = _mode(other, out)
                        e_t = Tile("sram", E[s][e_i].addr, *_tile_dims(mode, mm, nn))
                        self.emit(Load(e_t, self.operand_region(other, mode, b * M + i, mm, j, nn)),
                                  f"epilogue operand ({mode})")
                        srcs = [(e_t, mode), (c, FULL)] if swapped else [(c, FULL), (e_t, mode)]
                        self.emit(Vec(fn, c, srcs), "epilogue")
                        e_i += 1
                    o_t = Tile("sram", O[s].addr, mm, nn)
                    self.emit(Vec("copy", o_t, [(c, FULL)]), "writeback fp32 -> fp16")
                    self.emit(Store(self.region(out, b * M + i, mm, j, nn), o_t))
                    tile_no += 1

    def _row_tiles(self, rows: int, cols: int) -> tuple[int, int]:
        tc = min(cols, MAX_BUF)
        tr = max(1, min(rows, MAX_BUF // tc))
        return tr, tc

    def elementwise(self, op: ir.Op) -> None:
        out = op.result
        R, Cn = view2d(out.type.shape)
        tr, tc = self._row_tiles(R, Cn)
        modes = [_mode(v, out) for v in op.inputs]
        sram, _ = self.fresh()
        ins_bufs = [[sram.alloc(tr, tc) for _ in op.inputs] for _ in range(self.slots)]
        out_bufs = [sram.alloc(tr, tc) for _ in range(self.slots)]
        n = 0
        for r in range(0, R, tr):
            nr = min(tr, R - r)
            for c in range(0, Cn, tc):
                nc = min(tc, Cn - c)
                s = n % self.slots
                srcs = []
                for v, mode, buf in zip(op.inputs, modes, ins_bufs[s]):
                    t = Tile("sram", buf.addr, *_tile_dims(mode, nr, nc))
                    self.emit(Load(t, self.operand_region(v, mode, r, nr, c, nc)))
                    srcs.append((t, mode))
                o = Tile("sram", out_bufs[s].addr, nr, nc)
                self.emit(Vec(op.kind, o, srcs))
                self.emit(Store(self.region(out, r, nr, c, nc), o))
                n += 1

    def rowop(self, op: ir.Op) -> None:
        x, out = op.inputs[0], op.result
        R, Cn = view2d(out.type.shape)
        tr = max(1, min(R, MAX_BUF // Cn))
        sram, _ = self.fresh()
        extra = []
        for v in op.inputs[1:]:      # layernorm gamma / beta: loaded once, broadcast per row
            t = sram.alloc(1, Cn)
            self.emit(Load(t, self.region(v, 0, 1, 0, Cn)), "row parameter")
            extra.append(t)
        X = [sram.alloc(tr, Cn) for _ in range(self.slots)]
        Y = [sram.alloc(tr, Cn) for _ in range(self.slots)]
        for n, r in enumerate(range(0, R, tr)):
            nr = min(tr, R - r)
            s = n % self.slots
            xt = Tile("sram", X[s].addr, nr, Cn)
            yt = Tile("sram", Y[s].addr, nr, Cn)
            self.emit(Load(xt, self.region(x, r, nr, 0, Cn)))
            self.emit(RowOp(op.kind, yt, xt, extra, op.attrs.get("eps", 0.0)))
            self.emit(Store(self.region(out, r, nr, 0, Cn), yt))


def _mode(v: ir.Value, out: ir.Value) -> str:
    m = bcast_mode(v.type.shape, out.type.shape)
    assert m is not None, "check() should have rejected this op"
    return m


def _tile_dims(mode: str, nr: int, nc: int) -> tuple[int, int]:
    return {FULL: (nr, nc), ROW: (1, nc), SCALAR: (1, 1)}[mode]


def format_program(c: Compiled, limit: Optional[int] = None) -> str:
    lines = [f"; {n}" for n in c.notes]
    prog = c.program if limit is None else c.program[:limit]
    for i, ins in enumerate(prog):
        text = f"{i:5d}  {ins}"
        lines.append(f"{text:<88} ; {ins.comment}" if ins.comment else text)
    if limit is not None and len(c.program) > limit:
        lines.append(f"  ... {len(c.program) - limit:,} more instructions")
    return "\n".join(lines)
