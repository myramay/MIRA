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


# ------------------------------------------------------------------ W8A8: int8 weights *and* int8 math

def _calibrate(g: ir.Graph, feeds: list[dict], watch: set) -> dict:
    """Run g on the CPU for each calibration input; return the largest |value| seen for each watched value."""
    from . import ops as O
    peak: dict = {v: 0.0 for v in watch}
    for feed in feeds:
        env = {}
        for v in g.inputs:
            if v.name not in feed:
                raise ValueError(f"calibration input is missing '{v.name}'")
            env[v] = np.asarray(feed[v.name]).astype(v.type.np_dtype)
        for op in g.ops:
            if op.is_control:
                raise ValueError("W8A8 quantization doesn't support programs with data-dependent control flow yet")
            env[op.result] = O.evaluate(op.kind, [env[x] for x in op.inputs], op.attrs)
            if op.result in peak:
                peak[op.result] = max(peak[op.result], float(np.abs(env[op.result].astype(np.float32)).max()))
        for v in g.inputs:
            if v in peak:
                peak[v] = max(peak[v], float(np.abs(env[v].astype(np.float32)).max()))
    return peak


def _w8a8_weight(op: ir.Op) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """(int8 weights, per-output-channel scales) if `op` can run in int8, else None."""
    w = op.inputs[1].producer
    if w is None or not w.is_constant_like:
        return None
    if op.kind == "matmul":
        if op.inputs[1].type.rank != 2 or op.inputs[0].type.rank < 2:
            return None
        axis = 1
    elif op.kind == "conv2d":
        if op.attrs.get("groups", 1) != 1:     # depthwise / grouped convs stay in floating point
            return None
        axis = 0
    else:
        return None
    if w.kind == "dequantize" and w.attrs["axis"] == axis:
        return w.attrs["q"], w.attrs["scale"]
    value = w.attrs["value"] if w.is_const else None
    if value is None:
        return None
    return quantize_array(value, axis)


def quantize_w8a8(g: ir.Graph, calibration: list[dict], log: Optional[list] = None) -> int:
    """Rewrite matmuls and (ungrouped) convolutions with constant weights to run in int8.

    Activation scales come from calibration: the largest |x| each op saw over the calibration inputs.
    Calibration data should look like real inputs; outliers it never saw get clipped to +-127.
    """
    candidates = [op for op in g.ops if op.kind in ("matmul", "conv2d") and _w8a8_weight(op) is not None]
    peak = _calibrate(g, calibration, {op.inputs[0] for op in candidates})
    for op in candidates:
        q, w_scale = _w8a8_weight(op)
        x_scale = peak[op.inputs[0]] / 127.0 or 1.0
        epi = tuple((fn, None if idx is None else idx - 1, sw) for fn, idx, sw in op.attrs.get("epilogue", ()))
        attrs = {k: v for k, v in op.attrs.items() if k != "epilogue"}
        attrs.update({"q": q, "w_scale": np.asarray(w_scale, np.float32), "x_scale": float(x_scale),
                      "dtype": op.result.type.dtype})
        if epi:
            attrs["epilogue"] = epi
        op.kind = "qmatmul" if op.kind == "matmul" else "qconv2d"
        op.inputs = [op.inputs[0]] + op.inputs[2:]
        op.attrs = attrs
    from .passes import dce
    dce(g)
    g.verify()
    if log is not None:
        log.append(("quantize-w8a8", f"// {len(candidates)} ops now compute in int8\n{g}"))
    return len(candidates)
