"""Code generation for MNPU-1: IR subgraph -> instruction stream.

For each op this module:
  1. picks tile sizes that fit on-chip memory (a small cost model: minimize DRAM traffic),
  2. plans SRAM / accumulator buffers (bank-aligned bump allocation),
  3. emits DMA loads, compute, and stores, alternating between two buffer
     "slots" (double buffering) so loads of tile i+1 overlap compute on tile i.

Data layout: every tensor lives in DRAM as flat fp16 (int8 for quantized
weights), viewed as 2D [prod(leading dims), last dim]. `reshape` is free: the
output aliases the input's storage with a different 2D view. Layout changes
the hardware can do while copying (row permutations, 2D transposes, strided
gathers) are done by the DMA engine, not by compute.

Kernels:
  matmul     tiled GEMM with fused epilogue; int8 weights dequantized by the DMA
  conv2d     implicit GEMM: one matmul per kernel tap (i, j), accumulated; the
             DMA gathers the shifted/strided input rows, weights are rearranged
             to [KH, KW, O, C] at compile time so each tap is a contiguous block
  maxpool2d  running max over strided loads of each window tap
  elementwise / where / broadcast   vector unit, with full/row/col/scalar broadcast
  softmax, layernorm, row reductions   row-wise vector kernels (last axis)
  transpose  2D swaps via the transposing DMA; perms that keep the last axis
             last via strided row copies
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ... import ir
from ...ops import BINARY, UNARY
from .machine import (COL, FULL, ROW, SCALAR, Config, DramRegion, Instr, Load, MatMul, Program, RowOp, Store, Tile,
                      Vec)

MAX_BUF = 32 * 1024   # elements per vector buffer (4 SRAM banks)


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
    if operand and result and operand[-1] == result[-1] and all(d == 1 for d in operand[:-1]):
        return ROW
    lead = len(result) - len(operand)
    if (operand and result and operand[-1] == 1 and lead >= 0
            and tuple(operand[:-1]) == tuple(result[lead:-1]) and all(d == 1 for d in result[:lead])):
        return COL    # one value per row, e.g. x - max(x, axis=-1, keepdims=true)
    return None


def conv_channel_mode(operand: tuple[int, ...], out: tuple[int, ...]) -> Optional[str]:
    """Broadcast of a conv epilogue operand onto [N, O, OH, OW] tiles whose rows are channels."""
    if operand == out:
        return FULL
    if int(np.prod(operand, dtype=np.int64)) == 1:
        return SCALAR
    o = out[1]
    padded = (1,) * (4 - len(operand)) + tuple(operand)
    if len(operand) <= 4 and padded == (1, o, 1, 1):
        return COL
    return None


def transpose_kind(perm: tuple[int, ...]) -> Optional[str]:
    n = len(perm)
    if n >= 2 and perm[:-2] == tuple(range(n - 2)) and perm[-2:] == (n - 1, n - 2):
        return "swap"       # [..., R, C] -> [..., C, R]: transposing DMA
    if n >= 1 and perm[-1] == n - 1:
        return "rows"       # last axis stays last: rows move as whole units (strided row copies)
    return None


def decompose_transpose(perm: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Write any permutation as at most three the DMA engine can do: rows, then swap, then rows.

    Step 1 (rows) moves the axis that must end up last to position n-2, keeping the
    old last axis last; step 2 swaps the last two axes; step 3 (rows) puts the
    remaining axes in order. transpose(transpose(transpose(x, r1), s), r2) == transpose(x, perm).
    """
    n = len(perm)
    if transpose_kind(perm) is not None:
        return [perm]
    a = perm[-1]
    r1 = tuple([i for i in range(n) if i not in (a, n - 1)] + [a, n - 1])
    swap = tuple(range(n - 2)) + (n - 1, n - 2)
    cur = [r1[swap[i]] for i in range(n)]          # which original axis sits at each position now
    r2 = tuple(cur.index(perm[i]) for i in range(n))
    identity = tuple(range(n))
    return [p for p in (r1, swap, r2) if p != identity]


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
    if k == "conv2d":
        c = op.inputs[0].type.shape[1]
        if c > 512:
            return "conv2d: more than 512 input channels per tap not implemented"
        for fn, idx, _ in op.attrs.get("epilogue", ()):
            if idx is not None and conv_channel_mode(op.inputs[idx].type.shape, out) is None:
                return f"conv2d epilogue '{fn}': operand must be per-channel, full-size, or scalar"
        return None
    if k == "maxpool2d":
        return None
    if k in UNARY or k in BINARY or k in ("where", "broadcast"):
        for v in op.inputs:
            if bcast_mode(v.type.shape, out) is None:
                return f"{k}: broadcast of {v.type} to {op.result.type} needs a gather MNPU-1 lacks"
        return None
    if k == "reshape":
        return None
    if k == "transpose":
        return None     # any permutation decomposes into DMA-friendly steps
    if k in ("softmax", "layernorm") or k.startswith("reduce_"):
        x = op.inputs[0].type
        last = x.rank - 1
        if k == "softmax" and op.attrs["axis"] % max(x.rank, 1) != last:
            return "softmax: the vector unit only reduces along the last axis"
        if k.startswith("reduce_") and (x.rank == 0 or tuple(a % x.rank for a in op.attrs["axes"]) != (last,)):
            return f"{k}: the vector unit only reduces along the last axis"
        if view2d(x.shape)[1] > MAX_BUF:
            return f"{k}: a row of {view2d(x.shape)[1]} elements doesn't fit a vector buffer"
        return None
    reasons = {
        "concat": "concat: not implemented",
        "slice": "slice: not implemented",
        "pad": "pad: not implemented",
        "cast": "cast: MNPU-1 is fp16-only",
    }
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

    def free_elems(self) -> int:
        return self.capacity - self.top


def tile_candidates(d: int) -> list[int]:
    return sorted({min(d, 1 << p) for p in range(5, 13)})


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


def _tile_dims(mode: str, nr: int, nc: int) -> tuple[int, int]:
    return {FULL: (nr, nc), ROW: (1, nc), COL: (nr, 1), SCALAR: (1, 1)}[mode]


# ------------------------------------------------------------------ code generation

@dataclass
class Compiled:
    program: Program
    storage: dict[ir.Value, tuple[str, tuple[int, int]]]   # value -> (storage name, 2D view)
    sizes: dict[str, tuple[int, type]]                      # storage name -> (elements, dtype)
    consts: dict[str, np.ndarray]                           # storage preloaded at load time (weights)
    notes: list[str] = field(default_factory=list)


class CodeGen:
    def __init__(self, cfg: Config, double_buffer: bool = True):
        self.cfg = cfg
        self.slots = 2 if double_buffer else 1
        self.prog: Program = []
        self.storage: dict[ir.Value, tuple[str, tuple[int, int]]] = {}
        self.sizes: dict[str, tuple[int, type]] = {}
        self.consts: dict[str, np.ndarray] = {}
        self.quant: dict[ir.Value, tuple[str, int]] = {}    # dequantized value -> (scale storage, axis)
        self.notes: list[str] = []

    # ----- DRAM

    def place(self, v: ir.Value, alias_of: Optional[ir.Value] = None) -> None:
        if alias_of is not None:
            self.storage[v] = (self.storage[alias_of][0], view2d(v.type.shape))
            if alias_of in self.quant:
                self.quant[v] = self.quant[alias_of]
            return
        name = f"t{v.id}" if v.name is None else f"in_{v.name}"
        self.storage[v] = (name, view2d(v.type.shape))
        self.sizes[name] = (v.type.numel, np.float16)

    def scratch(self, name: str, elems: int) -> str:
        self.sizes[name] = (elems, np.float16)
        return name

    def region(self, v: ir.Value, r0: int, nr: int, c0: int, nc: int, row_stride: int = 1,
               col_stride: int = 1) -> DramRegion:
        name, (_, cols) = self.storage[v]
        return DramRegion(name, cols, r0, nr, c0, nc, row_stride, col_stride)

    def emit(self, ins: Instr, comment: str = "") -> None:
        ins.comment = comment
        self.prog.append(ins)

    def fresh(self) -> tuple[Allocator, Allocator]:
        c = self.cfg
        return (Allocator("sram", c.sram_elems, c.sram_bank_elems),
                Allocator("acc", c.acc_elems, c.acc_bank_elems))

    def operand_region(self, v: ir.Value, mode: str, r0: int, nr: int, c0: int, nc: int,
                       row_stride: int = 1) -> DramRegion:
        if mode == FULL:
            return self.region(v, r0, nr, c0, nc, row_stride)
        if mode == ROW:
            return self.region(v, 0, 1, c0, nc)
        if mode == COL:
            return DramRegion(self.storage[v][0], 1, r0, nr, 0, 1)
        return self.region(v, 0, 1, 0, 1)

    def load(self, dst: Tile, v: ir.Value, src: DramRegion, scale_rows: Optional[tuple[int, int]] = None,
             scale_cols: Optional[tuple[int, int]] = None, comment: str = "") -> None:
        """Load a region of v; if v is an int8 weight, dequantize it on the way in."""
        scale = None
        axis = 1
        if v in self.quant:
            sname, _ = self.quant[v]
            if scale_cols is not None:
                scale, axis = DramRegion(sname, 1, scale_cols[0], scale_cols[1], 0, 1), 1
            elif scale_rows is not None:
                scale, axis = DramRegion(sname, 1, scale_rows[0], scale_rows[1], 0, 1), 0
        self.emit(Load(dst, src, scale=scale, scale_axis=axis), comment)

    # ----- graph

    def compile(self, g: ir.Graph) -> Compiled:
        for v in g.inputs:
            self.place(v)
        for op in g.ops:
            if op.is_const:
                self.place(op.result)
                self.consts[self.storage[op.result][0]] = op.attrs["value"].astype(np.float16).ravel()
                continue
            if op.kind == "dequantize":
                self.place_quantized(op)
                continue
            if op.kind == "reshape":
                self.place(op.result, alias_of=op.inputs[0])
                continue
            self.place(op.result)
            if op.kind == "matmul":
                self.matmul(op)
            elif op.kind == "conv2d":
                self.conv2d(op)
            elif op.kind == "maxpool2d":
                self.maxpool2d(op)
            elif op.kind == "transpose":
                self.transpose(op)
            elif op.kind in ("softmax", "layernorm") or op.kind.startswith("reduce_"):
                self.rowop(op)
            else:
                self.elementwise(op)
        return Compiled(self.prog, self.storage, self.sizes, self.consts, self.notes)

    def place_quantized(self, op: ir.Op) -> None:
        """int8 weight: stored compressed in DRAM, expanded to fp16 by the DMA engine while loading."""
        v = op.result
        name = f"q{v.id}"
        q = op.attrs["q"]
        self.storage[v] = (name, view2d(v.type.shape))
        self.sizes[name] = (q.size, np.int8)
        self.consts[name] = q.ravel()
        self.sizes[name + "_scale"] = (op.attrs["scale"].size, np.float16)
        self.consts[name + "_scale"] = op.attrs["scale"].astype(np.float16)
        self.quant[v] = (name + "_scale", op.attrs["axis"])

    # ----- matmul

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
        qnote = " int8 weights" if w in self.quant else ""
        self.notes.append(f"matmul {out!r}: M={M} K={K} N={N} x{batches} -> tiles tm={tm} tk={tk} tn={tn}{qnote}"
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
                        self.load(a_t, x, self.region(x, b * M + i, mm, kk, kn), comment=f"A tile ({i},{kk}) slot {t}")
                        w_row = (b * K if w.type.rank > 2 else 0) + kk
                        self.load(b_t, w, self.region(w, w_row, kn, j, nn), scale_cols=(j, nn),
                                  comment=f"B tile ({kk},{j}) slot {t}")
                        self.emit(MatMul(c, a_t, b_t, accumulate=kk > 0))
                        ab += 1
                    self.epilogue(op, c, E[s], lambda v, mode: self.operand_region(v, mode, b * M + i, mm, j, nn),
                                  lambda v: bcast_mode_of(op, v))
                    o_t = Tile("sram", O[s].addr, mm, nn)
                    self.emit(Vec("copy", o_t, [(c, FULL)]), "writeback fp32 -> fp16")
                    self.emit(Store(self.region(out, b * M + i, mm, j, nn), o_t))
                    tile_no += 1

    def epilogue(self, op: ir.Op, c: Tile, bufs: list[Tile], region_for, mode_for) -> None:
        e_i = 0
        for fn, idx, swapped in op.attrs.get("epilogue", ()):
            if idx is None:
                self.emit(Vec(fn, c, [(c, FULL)]), "epilogue")
                continue
            other = op.inputs[idx]
            mode = mode_for(other)
            e_t = Tile("sram", bufs[e_i].addr, *_tile_dims(mode, c.rows, c.cols))
            self.load(e_t, other, region_for(other, mode), comment=f"epilogue operand ({mode})")
            srcs = [(e_t, mode), (c, FULL)] if swapped else [(c, FULL), (e_t, mode)]
            self.emit(Vec(fn, c, srcs), "epilogue")
            e_i += 1

    # ----- conv2d as implicit GEMM

    def conv2d(self, op: ir.Op) -> None:
        x, w = op.inputs[0], op.inputs[1]
        out = op.result
        n_, c_, h_, w_ = x.type.shape
        o_, _, kh, kw = w.type.shape
        _, _, oh_, ow_ = out.type.shape
        (sh, sw), (ph, pw) = op.attrs["stride"], op.attrs["padding"]

        # 1. padding: copy x into the interior of a zeroed DRAM buffer, by DMA
        if ph or pw:
            hp, wp = h_ + 2 * ph, w_ + 2 * pw
            pad_name = self.scratch(f"t{out.id}_pad", n_ * c_ * hp * wp)
            sram, _ = self.fresh()
            rows_per = max(1, min(h_, MAX_BUF // w_))
            bufs = [sram.alloc(rows_per, w_) for _ in range(self.slots)]
            k = 0
            for plane in range(n_ * c_):
                for r in range(0, h_, rows_per):
                    nr = min(rows_per, h_ - r)
                    t = Tile("sram", bufs[k % self.slots].addr, nr, w_)
                    self.emit(Load(t, self.region(x, plane * h_ + r, nr, 0, w_)), "pad: copy in")
                    self.emit(Store(DramRegion(pad_name, wp, plane * hp + ph + r, nr, pw, w_), t), "pad: copy out")
                    k += 1
            src_name, src_h, src_w = pad_name, hp, wp
        else:
            src_name, src_h, src_w = self.storage[x][0], h_, w_

        # 2. weights, rearranged at compile time to [KH, KW, O, C] so each tap (i, j) is a contiguous [O, C] block
        wp_ = w.producer
        wname = f"t{out.id}_w"
        if wp_ is not None and wp_.kind == "dequantize":
            q = np.ascontiguousarray(wp_.attrs["q"].transpose(2, 3, 0, 1))
            self.sizes[wname] = (q.size, np.int8)
            self.consts[wname] = q.ravel()
            scale_name = self.quant[w][0]
        elif wp_ is not None and wp_.is_const:
            wr = np.ascontiguousarray(wp_.attrs["value"].astype(np.float16).transpose(2, 3, 0, 1))
            self.sizes[wname] = (wr.size, np.float16)
            self.consts[wname] = wr.ravel()
            scale_name = None
        else:
            raise NotImplementedError("conv2d with runtime weights")   # check() keeps these off the NPU

        # 3. tiles: rows = output channels, cols = output pixels of one row, K = input channels
        to, tw, tc = min(o_, 128), min(ow_, 512), min(c_, 512)
        sram, acc = self.fresh()
        resident = kh * kw * o_ * c_ <= sram.free_elems() // 2 and tc == c_
        n_epi = sum(1 for _, idx, _ in op.attrs.get("epilogue", ()) if idx is not None)
        if resident:     # weight-stationary: load all weights once, reuse for every output row
            W = sram.alloc(kh * kw * o_, c_)
            if scale_name is None:
                self.emit(Load(W, DramRegion(wname, c_, 0, kh * kw * o_, 0, c_)), "all weights, loaded once")
            else:            # int8: dequantize each tap with the per-output-channel scales
                for tap in range(kh * kw):
                    self.emit(Load(Tile("sram", W.addr + tap * o_ * c_, o_, c_),
                                   DramRegion(wname, c_, tap * o_, o_, 0, c_),
                                   scale=DramRegion(scale_name, 1, 0, o_, 0, 1), scale_axis=0),
                              f"weights tap {tap}, loaded once")
            A = []
        else:
            A = [sram.alloc(to, tc) for _ in range(self.slots)]
        B = [sram.alloc(tc, tw) for _ in range(self.slots)]
        E = [[sram.alloc(to, tw) for _ in range(n_epi)] for _ in range(self.slots)]
        O = [sram.alloc(to, tw) for _ in range(self.slots)]
        C = [acc.alloc(to, tw) for _ in range(self.slots)]
        self.notes.append(f"conv2d {out!r}: implicit GEMM, {kh}x{kw} taps, tiles o={to} ow={tw} c={tc}"
                          + (", weights resident in SRAM" if resident else "") + (", int8" if scale_name else ""))

        ab = tile_no = 0
        for n in range(n_):
            for oh in range(oh_):
                for o0 in range(0, o_, to):
                    on = min(to, o_ - o0)
                    for w0 in range(0, ow_, tw):
                        wn = min(tw, ow_ - w0)
                        s = tile_no % self.slots
                        c = Tile("acc", C[s].addr, on, wn)
                        first = True
                        for i in range(kh):
                            for j in range(kw):
                                for c0 in range(0, c_, tc):
                                    cn = min(tc, c_ - c0)
                                    t = ab % self.slots
                                    tap = i * kw + j
                                    if resident:
                                        a_t = Tile("sram", W.addr + (tap * o_ + o0) * c_, on, c_)
                                    else:
                                        a_t = Tile("sram", A[t].addr, on, cn)
                                        self.emit(Load(a_t, DramRegion(wname, c_, tap * o_ + o0, on, c0, cn),
                                                       scale=DramRegion(scale_name, 1, o0, on, 0, 1) if scale_name
                                                       else None, scale_axis=0), f"weights tap ({i},{j})")
                                    b_t = Tile("sram", B[t].addr, cn, wn)
                                    row0 = (n * c_ + c0) * src_h + oh * sh + i
                                    self.emit(Load(b_t, DramRegion(src_name, src_w, row0, cn, w0 * sw + j, wn,
                                                                   row_stride=src_h, col_stride=sw)),
                                              f"input rows for tap ({i},{j})")
                                    self.emit(MatMul(c, a_t, b_t, accumulate=not first))
                                    first = False
                                    ab += 1
                        out_row = (n * o_ + o0) * oh_ + oh
                        self.epilogue(
                            op, c, E[s],
                            lambda v, mode: (self.region(v, out_row, on, w0, wn, row_stride=oh_) if mode == FULL
                                             else self.operand_region(v, mode, o0, on, 0, 1)),
                            lambda v: conv_channel_mode(v.type.shape, out.type.shape))
                        o_t = Tile("sram", O[s].addr, on, wn)
                        self.emit(Vec("copy", o_t, [(c, FULL)]), "writeback fp32 -> fp16")
                        self.emit(Store(self.region(out, out_row, on, w0, wn, row_stride=oh_), o_t))
                        tile_no += 1

    # ----- maxpool2d

    def maxpool2d(self, op: ir.Op) -> None:
        x, out = op.inputs[0], op.result
        n_, c_, h_, w_ = x.type.shape
        _, _, oh_, ow_ = out.type.shape
        k, s_ = op.attrs["size"], op.attrs["stride"]
        planes = n_ * c_
        tw = min(ow_, 512)
        tp = max(1, min(planes, MAX_BUF // tw))
        sram, _ = self.fresh()
        ACC = [sram.alloc(tp, tw) for _ in range(self.slots)]
        TAP = [sram.alloc(tp, tw) for _ in range(self.slots)]
        n = 0
        for oh in range(oh_):
            for p0 in range(0, planes, tp):
                pn = min(tp, planes - p0)
                for w0 in range(0, ow_, tw):
                    wn = min(tw, ow_ - w0)
                    sl = n % self.slots
                    a_t = Tile("sram", ACC[sl].addr, pn, wn)
                    for i in range(k):
                        for j in range(k):
                            src = self.region(x, p0 * h_ + oh * s_ + i, pn, w0 * s_ + j, wn, row_stride=h_,
                                              col_stride=s_)
                            if i == 0 and j == 0:
                                self.emit(Load(a_t, src), "first window tap")
                            else:
                                t_t = Tile("sram", TAP[sl].addr, pn, wn)
                                self.emit(Load(t_t, src))
                                self.emit(Vec("maximum", a_t, [(a_t, FULL), (t_t, FULL)]))
                    self.emit(Store(self.region(out, p0 * oh_ + oh, pn, w0, wn, row_stride=oh_), a_t))
                    n += 1

    # ----- transpose

    def transpose(self, op: ir.Op) -> None:
        steps = decompose_transpose(tuple(op.attrs["perm"]))
        src = op.inputs[0]
        for i, perm in enumerate(steps):
            if i == len(steps) - 1:
                dst = op.result
            else:              # intermediate result in a DRAM scratch buffer
                shape = tuple(src.type.shape[p] for p in perm)
                dst = ir.Value(src.type.__class__(src.type.dtype, shape))
                self.storage[dst] = (self.scratch(f"t{op.result.id}_step{i}", dst.type.numel), view2d(shape))
            self.transpose_to(src, dst, perm)
            src = dst
        if len(steps) > 1:
            steps_text = " then ".join(map(str, steps))
            self.notes.append(f"transpose {op.result!r}: perm {tuple(op.attrs['perm'])} = {steps_text}")

    def transpose_to(self, x: ir.Value, out: ir.Value, perm: tuple[int, ...]) -> None:
        sram, _ = self.fresh()
        if transpose_kind(perm) == "swap":
            *lead, R, Cn = x.type.shape
            batches = int(np.prod(lead, dtype=np.int64)) if lead else 1
            tr, tc = min(Cn, 128), min(R, 128)
            bufs = [sram.alloc(tr, tc) for _ in range(self.slots)]
            n = 0
            for b in range(batches):
                for c0 in range(0, Cn, tr):
                    cn = min(tr, Cn - c0)
                    for r0 in range(0, R, tc):
                        rn = min(tc, R - r0)
                        t = Tile("sram", bufs[n % self.slots].addr, cn, rn)
                        self.emit(Load(t, self.region(x, b * R + r0, rn, c0, cn), transpose=True))
                        self.emit(Store(self.region(out, b * Cn + c0, cn, r0, rn), t))
                        n += 1
            return
        # last axis stays last: every output row is some input row -> strided row copies
        lead_in = x.type.shape[:-1]
        D = x.type.shape[-1] if x.type.shape else 1
        src_rows = np.arange(int(np.prod(lead_in, dtype=np.int64))).reshape(lead_in).transpose(perm[:-1]).ravel()
        tr = max(1, min(len(src_rows), MAX_BUF // D))
        bufs = [sram.alloc(tr, D) for _ in range(self.slots)]
        start = n = 0
        while start < len(src_rows):
            end = start + 1                  # grow a run of rows with a constant source stride
            stride = int(src_rows[start + 1] - src_rows[start]) if start + 1 < len(src_rows) else 1
            while end < len(src_rows) and end - start < tr and src_rows[end] - src_rows[end - 1] == stride:
                end += 1
            if stride <= 0:
                stride, end = 1, start + 1
            t = Tile("sram", bufs[n % self.slots].addr, end - start, D)
            self.emit(Load(t, self.region(x, int(src_rows[start]), end - start, 0, D, row_stride=stride)))
            self.emit(Store(self.region(out, start, end - start, 0, D), t))
            start, n = end, n + 1

    # ----- elementwise (incl. where, broadcast)

    def _row_tiles(self, rows: int, cols: int) -> tuple[int, int]:
        tc = min(cols, MAX_BUF)
        tr = max(1, min(rows, MAX_BUF // tc))
        return tr, tc

    def elementwise(self, op: ir.Op) -> None:
        out = op.result
        fn = "copy" if op.kind == "broadcast" else op.kind
        R, Cn = view2d(out.type.shape)
        tr, tc = self._row_tiles(R, Cn)
        modes = [bcast_mode(v.type.shape, out.type.shape) for v in op.inputs]
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
                self.emit(Vec(fn, o, srcs))
                self.emit(Store(self.region(out, r, nr, c, nc), o))
                n += 1

    # ----- row-wise kernels

    def rowop(self, op: ir.Op) -> None:
        x, out = op.inputs[0], op.result
        R, Cn = view2d(x.type.shape)
        tr = max(1, min(R, MAX_BUF // Cn))
        sram, _ = self.fresh()
        extra = []
        for v in op.inputs[1:]:      # layernorm gamma / beta: loaded once, broadcast per row
            t = sram.alloc(1, Cn)
            self.emit(Load(t, self.region(v, 0, 1, 0, Cn)), "row parameter")
            extra.append(t)
        reduce = op.kind.startswith("reduce_")
        fn = {"reduce_sum": "rsum", "reduce_mean": "rmean", "reduce_max": "rmax"}.get(op.kind, op.kind)
        out_cols = 1 if reduce else Cn
        X = [sram.alloc(tr, Cn) for _ in range(self.slots)]
        Y = [sram.alloc(tr, out_cols) for _ in range(self.slots)]
        for n, r in enumerate(range(0, R, tr)):
            nr = min(tr, R - r)
            s = n % self.slots
            xt = Tile("sram", X[s].addr, nr, Cn)
            yt = Tile("sram", Y[s].addr, nr, out_cols)
            self.emit(Load(xt, self.region(x, r, nr, 0, Cn)))
            self.emit(RowOp(fn, yt, xt, extra, op.attrs.get("eps", 0.0)))
            dst = DramRegion(self.storage[out][0], 1, r, nr, 0, 1) if reduce else self.region(out, r, nr, 0, Cn)
            self.emit(Store(dst, yt))


def bcast_mode_of(op: ir.Op, v: ir.Value) -> str:
    m = bcast_mode(v.type.shape, op.result.type.shape)
    assert m is not None, "check() should have rejected this op"
    return m


def format_program(c: Compiled, limit: Optional[int] = None) -> str:
    lines = [f"; {n}" for n in c.notes]
    prog = c.program if limit is None else c.program[:limit]
    for i, ins in enumerate(prog):
        text = f"{i:5d}  {ins}"
        lines.append(f"{text:<88} ; {ins.comment}" if ins.comment else text)
    if limit is not None and len(c.program) > limit:
        lines.append(f"  ... {len(c.program) - limit:,} more instructions")
    return "\n".join(lines)
