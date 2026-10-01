"""Weight quantization: store large weights as int8 with per-channel scales.

Symmetric, per-output-channel int8 ("weight-only" quantization, the scheme most
NPU and LLM deployments use): for each output channel c,

    scale[c] = max |w[..., c]| / 127        q = round(w / scale)  in [-127, 127]

and the weight is rebuilt as q * scale where it's used. That quarters the
bytes of an f32 weight (halves an f16 one), which matters because NPUs are
usually limited by how fast weights stream in from DRAM, not by arithmetic.
Activations stay in fp16, so accuracy loss is small.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import ir

MIN_ELEMS = 1024   # small tensors (biases, norms) aren't worth quantizing


def quantize_array(w: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    reduce = tuple(i for i in range(w.ndim) if i != axis)
    amax = np.abs(w.astype(np.float32)).max(axis=reduce)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    shape = [1] * w.ndim
    shape[axis] = -1
    q = np.clip(np.round(w.astype(np.float32) / scale.reshape(shape)), -127, 127).astype(np.int8)
    return q, scale


def _axis_for(v: ir.Value, users: list[ir.Op]) -> Optional[int]:
    """Output-channel axis if every use is as a matmul/conv weight, else None."""
    if not users:
        return None
    if v.type.rank == 2 and all(u.kind == "matmul" and u.inputs[1] is v and u.inputs[0] is not v for u in users):
        return 1       # [K, N]: one scale per output column
    if v.type.rank == 4 and all(u.kind == "conv2d" and u.inputs[1] is v for u in users):
        return 0       # [O, C, KH, KW]: one scale per output channel
    return None


def quantize_weights(g: ir.Graph, mode: str = "int8", log: Optional[list] = None) -> tuple[int, int]:
    if mode != "int8":
        raise ValueError(f"unknown quantization mode '{mode}' (supported: int8)")
    before = after = 0
    users = g.users()
    for op in g.ops:
        for _, sub in op.subgraphs():
            b, a = quantize_weights(sub, mode)
            before, after = before + b, after + a
        if not op.is_const or op.result.type.numel < MIN_ELEMS:
            continue
        axis = _axis_for(op.result, users.get(op.result, []))
        if axis is None:
            continue
        w = op.attrs["value"]
        q, scale = quantize_array(w, axis)
        before += w.nbytes
        after += q.nbytes + scale.nbytes
        op.kind = "dequantize"
        op.attrs = {"q": q, "scale": scale, "axis": axis, "dtype": op.result.type.dtype}
    g.verify()
    if log is not None and before:
        log.append(("quantize-int8", f"// weights: {before / 2**20:.2f} MiB -> {after / 2**20:.2f} MiB\n{g}"))
    return before, after
