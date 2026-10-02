"""The op registry: the single source of truth for what each op means.

For every op we define
  * `infer(input_types, attrs) -> TensorType`  - shape/dtype rule (raises OpTypeError)
  * `ref(inputs, attrs) -> np.ndarray`          - reference semantics in NumPy

The elaborator uses `infer` to type-check programs, the optimizer uses `ref` to
constant-fold, the CPU backend uses `ref` to execute, and every other backend
is tested against `ref`. Reference math runs in float32 and rounds to the op's
output dtype at the end, which matches how NPUs accumulate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .types import NP_DTYPES, TensorType, type_of_array


class OpTypeError(Exception):
    """An op was applied to operands with incompatible shapes or dtypes."""


@dataclass(frozen=True)
class OpDef:
    name: str
    category: str   # "const" | "unary" | "binary" | "matmul" | "conv" | "pool" | "reduce" | "shape" | "norm" | "misc"
    infer: Callable[[list[TensorType], dict], TensorType]
    ref: Callable[[list[np.ndarray], dict], np.ndarray]


REGISTRY: dict[str, OpDef] = {}


def register(name: str, category: str, infer, ref) -> None:
    REGISTRY[name] = OpDef(name, category, infer, ref)


# ---------------------------------------------------------------- helpers

def f32(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32, copy=False)


def same_dtype(types: list[TensorType], op: str) -> str:
    dts = {t.dtype for t in types}
    if len(dts) != 1:
        raise OpTypeError(f"{op}: operands have different dtypes ({', '.join(str(t) for t in types)}); use cast()")
    return dts.pop()


def broadcast_shapes(a: tuple[int, ...], b: tuple[int, ...], op: str) -> tuple[int, ...]:
    out = []
    for i in range(max(len(a), len(b))):
        da = a[-1 - i] if i < len(a) else 1
        db = b[-1 - i] if i < len(b) else 1
        if da == db or db == 1:
            out.append(da)
        elif da == 1:
            out.append(db)
        else:
            raise OpTypeError(
                f"{op}: shapes [{', '.join(map(str, a))}] and [{', '.join(map(str, b))}] "
                f"can't be broadcast (dimension {da} vs {db})")
    return tuple(reversed(out))


def need_float(types: list[TensorType], op: str) -> None:
    for t in types:
        if not t.is_float:
            raise OpTypeError(f"{op}: needs floating-point operands, got {t}; use cast(x, f32)")


def need_int(t: TensorType, op: str, what: str = "indices") -> None:
    if t.dtype != "i32":
        raise OpTypeError(f"{op}: {what} must be i32, got {t}")


def norm_axis(axis: int, rank: int, op: str) -> int:
    if not -rank <= axis < rank:
        raise OpTypeError(f"{op}: axis {axis} is out of range for a rank-{rank} tensor")
    return axis % rank


# ---------------------------------------------------------------- const

register("const", "const", lambda ts, a: type_of_array(a["value"]), lambda xs, a: a["value"])


# ---------------------------------------------------------------- elementwise

def _gelu(x):
    # tanh approximation (the one NPUs and Core ML implement natively)
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


UNARY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": lambda x: np.maximum(x, 0),
    "gelu": _gelu,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "neg": np.negative,
    "sign": np.sign,
    "erf": None,          # filled in below
}


def _erf(x):
    # Abramowitz & Stegun 7.1.26: |error| < 1.5e-7, plenty for fp32/fp16
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t \
        * np.exp(-a * a)
    return s * y


UNARY["erf"] = _erf
INT_UNARY = {"relu", "neg", "abs", "sign"}            # unary ops that also work on i32

BINARY: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "pow": np.power,
    "maximum": np.maximum,
    "minimum": np.minimum,
    # comparisons return 1.0 / 0.0 in the operands' dtype (masks you can multiply with)
    "greater": np.greater,
    "greater_equal": np.greater_equal,
    "equal": np.equal,
}
COMPARISONS = {"greater", "greater_equal", "equal"}
FLOAT_ONLY_BINARY = {"div", "pow"}


def _unary_infer_named(name):
    def infer(ts, a):
        if name not in INT_UNARY:
            need_float(ts, name)
        return ts[0]
    return infer


def _binary_infer_named(name):
    def infer(ts, a):
        dt = same_dtype(ts, name)
        if name in FLOAT_ONLY_BINARY:
            need_float(ts, name)
        return TensorType(dt, broadcast_shapes(ts[0].shape, ts[1].shape, name))
    return infer


def _native_or_f32(x: np.ndarray) -> np.ndarray:
    """Floats compute in fp32 (like NPU accumulators); integers stay exact."""
    return x if x.dtype.kind == "i" else f32(x)


for _n, _f in UNARY.items():
    register(_n, "unary", _unary_infer_named(_n),
             (lambda f: lambda xs, a: f(_native_or_f32(xs[0])).astype(xs[0].dtype))(_f))

for _n, _f in BINARY.items():
    register(_n, "binary", _binary_infer_named(_n),
             (lambda f: lambda xs, a: f(_native_or_f32(xs[0]), _native_or_f32(xs[1])).astype(xs[0].dtype))(_f))


# ---------------------------------------------------------------- epilogues
#
# After fusion, a matmul/conv may carry an "epilogue": a list of elementwise
# steps applied to its result before it is written out, e.g.
#     [("add", 2, False), ("relu", None, False)]
# meaning  out = relu(core_result + inputs[2]).  `swapped` means the extra
# operand is on the left, i.e. f(other, acc) - it matters for sub/div/pow.

def epilogue_infer(core: TensorType, ts: list[TensorType], epilogue, op: str) -> TensorType:
    for fn, idx, _swapped in epilogue:
        if idx is not None:
            other = ts[idx]
            if other.dtype != core.dtype or broadcast_shapes(core.shape, other.shape, op) != core.shape:
                raise OpTypeError(f"{op}: epilogue operand {other} doesn't fit result {core}")
    return core


def epilogue_ref(acc: np.ndarray, xs: list[np.ndarray], epilogue) -> np.ndarray:
    for fn, idx, swapped in epilogue:
        if idx is None:
            acc = UNARY[fn](acc)
        else:
            other = f32(xs[idx])
            acc = BINARY[fn](other, acc) if swapped else BINARY[fn](acc, other)
    return acc


# ---------------------------------------------------------------- matmul

def _matmul_infer(ts, a):
    x, w = ts[0], ts[1]
    dt = same_dtype([x, w], "matmul")
    need_float([x], "matmul")
    if x.rank < 2 or w.rank < 2:
        raise OpTypeError(f"matmul: both operands need rank >= 2, got {x} @ {w}")
    if x.shape[-1] != w.shape[-2]:
        raise OpTypeError(f"matmul: inner dimensions don't match: {x} @ {w} ({x.shape[-1]} != {w.shape[-2]})")
    batch = broadcast_shapes(x.shape[:-2], w.shape[:-2], "matmul")
    core = TensorType(dt, batch + (x.shape[-2], w.shape[-1]))
    return epilogue_infer(core, ts, a.get("epilogue", ()), "matmul")


def _matmul_ref(xs, a):
    acc = np.matmul(f32(xs[0]), f32(xs[1]))
    return epilogue_ref(acc, xs, a.get("epilogue", ())).astype(xs[0].dtype)


register("matmul", "matmul", _matmul_infer, _matmul_ref)


# ---------------------------------------------------------------- conv2d (NCHW input, OIHW weight)

def _conv_out(size, k, stride, pad, dilation=1):
    return (size + 2 * pad - dilation * (k - 1) - 1) // stride + 1


def conv_params(a: dict):
    """(stride, padding, dilation, groups) of a conv2d-family op, with defaults for older graphs."""
    return tuple(a["stride"]), tuple(a["padding"]), tuple(a.get("dilation", (1, 1))), a.get("groups", 1)


def _conv2d_infer(ts, a):
    x, w = ts[0], ts[1]
    dt = same_dtype([x, w], "conv2d")
    need_float([x], "conv2d")
    if x.rank != 4 or w.rank != 4:
        raise OpTypeError(f"conv2d: expects input [N, C, H, W] and weight [O, C/groups, KH, KW], got {x} and {w}")
    n, c, h, wd = x.shape
    o, ci, kh, kw = w.shape
    (sh, sw), (ph, pw), (dh, dw), g = conv_params(a)
    if g < 1 or c % g or o % g:
        raise OpTypeError(f"conv2d: groups={g} must divide both the {c} input and {o} output channels")
    if c != ci * g:
        raise OpTypeError(f"conv2d: input has {c} channels but weight expects {ci} per group x {g} groups")
    oh, ow = _conv_out(h, kh, sh, ph, dh), _conv_out(wd, kw, sw, pw, dw)
    if oh <= 0 or ow <= 0:
        raise OpTypeError(f"conv2d: kernel {kh}x{kw} (dilation {dh}x{dw}) is larger than padded input {h}x{wd}")
    core = TensorType(dt, (n, o, oh, ow))
    return epilogue_infer(core, ts, a.get("epilogue", ()), "conv2d")


def _taps(x: np.ndarray, kh: int, kw: int, a: dict, oh: int, ow: int):
    """Yield (i, j, x_tap) where x_tap[n, c, y, x] is the input pixel that kernel tap (i, j) sees."""
    (sh, sw), (ph, pw), (dh, dw), _ = conv_params(a)
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    for i in range(kh):
        for j in range(kw):
            yield i, j, xp[:, :, i * dh:i * dh + sh * (oh - 1) + 1:sh, j * dw:j * dw + sw * (ow - 1) + 1:sw]


def _conv2d_ref(xs, a):
    x, w = f32(xs[0]), f32(xs[1])
    g = a.get("groups", 1)
    n, c = x.shape[:2]
    o, cg, kh, kw = w.shape
    out_t = _conv2d_infer([TensorType("f32", x.shape), TensorType("f32", w.shape)], {**a, "epilogue": ()})
    oh, ow = out_t.shape[2:]
    acc = np.zeros((n, g, o // g, oh, ow), np.float32)
    wg = w.reshape(g, o // g, cg, kh, kw)
    for i, j, xt in _taps(x, kh, kw, a, oh, ow):   # one small (grouped) matmul per kernel tap
        acc += np.einsum("ngchw,goc->ngohw", xt.reshape(n, g, cg, oh, ow), wg[:, :, :, i, j], optimize=True)
    return epilogue_ref(acc.reshape(n, o, oh, ow), xs, a.get("epilogue", ())).astype(xs[0].dtype)


register("conv2d", "conv", _conv2d_infer, _conv2d_ref)


def _maxpool_infer(ts, a):
    x = ts[0]
    if x.rank != 4:
        raise OpTypeError(f"maxpool2d: expects [N, C, H, W], got {x}")
    k, s = a["size"], a["stride"]
    n, c, h, w = x.shape
    if h < k or w < k:
        raise OpTypeError(f"maxpool2d: window {k} is larger than input {h}x{w}")
    return TensorType(x.dtype, (n, c, (h - k) // s + 1, (w - k) // s + 1))


def _maxpool_ref(xs, a):
    k, s = a["size"], a["stride"]
    win = np.lib.stride_tricks.sliding_window_view(xs[0], (k, k), axis=(2, 3))[:, :, ::s, ::s]
    return win.max(axis=(-2, -1)).astype(xs[0].dtype)


register("maxpool2d", "pool", _maxpool_infer, _maxpool_ref)


# ---------------------------------------------------------------- softmax / reductions / norms

def _softmax_infer(ts, a):
    need_float(ts, "softmax")
    norm_axis(a["axis"], ts[0].rank, "softmax")
    return ts[0]


def _softmax_ref(xs, a):
    x = f32(xs[0])
    e = np.exp(x - x.max(axis=a["axis"], keepdims=True))
    return (e / e.sum(axis=a["axis"], keepdims=True)).astype(xs[0].dtype)


register("softmax", "reduce", _softmax_infer, _softmax_ref)

REDUCE = {"reduce_sum": np.sum, "reduce_mean": np.mean, "reduce_max": np.max}


def _reduce_infer_named(name):
    def infer(ts, a):
        x = ts[0]
        if name == "reduce_mean":
            need_float(ts, name)
        axes = sorted(norm_axis(ax, x.rank, name) for ax in a["axes"])
        if a["keepdims"]:
            shape = tuple(1 if i in axes else d for i, d in enumerate(x.shape))
        else:
            shape = tuple(d for i, d in enumerate(x.shape) if i not in axes)
        return TensorType(x.dtype, shape)
    return infer


def _reduce_ref_named(f):
    def ref(xs, a):
        out = f(_native_or_f32(xs[0]), axis=tuple(a["axes"]), keepdims=a["keepdims"])
        return np.asarray(out).astype(xs[0].dtype)
    return ref


for _n, _f in REDUCE.items():
    register(_n, "reduce", _reduce_infer_named(_n), _reduce_ref_named(_f))


def _layernorm_infer(ts, a):
    x, g, b = ts
    same_dtype(ts, "layernorm")
    need_float([x], "layernorm")
    d = x.shape[-1]
    if g.shape != (d,) or b.shape != (d,):
        raise OpTypeError(f"layernorm: gamma and beta must be [{d}] to match the last axis of {x}, got {g} and {b}")
    return x


def _layernorm_ref(xs, a):
    x, g, b = (f32(v) for v in xs)
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + a["eps"]) * g + b).astype(xs[0].dtype)


register("layernorm", "norm", _layernorm_infer, _layernorm_ref)


# ---------------------------------------------------------------- shape ops

def _transpose_infer(ts, a):
    x, perm = ts[0], tuple(a["perm"])
    if sorted(perm) != list(range(x.rank)):
        raise OpTypeError(f"transpose: perm {list(perm)} is not a permutation of the {x.rank} axes of {x}")
    return TensorType(x.dtype, tuple(x.shape[p] for p in perm))


register("transpose", "shape", _transpose_infer, lambda xs, a: np.ascontiguousarray(np.transpose(xs[0], a["perm"])))


def _reshape_infer(ts, a):
    x, shape = ts[0], tuple(a["shape"])
    if int(np.prod(shape, dtype=np.int64)) != x.numel:
        raise OpTypeError(f"reshape: can't reshape {x} ({x.numel} elements) to [{', '.join(map(str, shape))}]")
    return TensorType(x.dtype, shape)


register("reshape", "shape", _reshape_infer, lambda xs, a: xs[0].reshape(a["shape"]))


def _concat_infer(ts, a):
    dt = same_dtype(ts, "concat")
    ranks = {t.rank for t in ts}
    if len(ranks) != 1:
        raise OpTypeError("concat: all operands must have the same rank")
    rank = ranks.pop()
    ax = norm_axis(a["axis"], rank, "concat")
    for t in ts[1:]:
        if any(d0 != d1 for i, (d0, d1) in enumerate(zip(ts[0].shape, t.shape)) if i != ax):
            raise OpTypeError(f"concat: {ts[0]} and {t} differ outside axis {a['axis']}")
    shape = list(ts[0].shape)
    shape[ax] = sum(t.shape[ax] for t in ts)
    return TensorType(dt, tuple(shape))


register("concat", "shape", _concat_infer, lambda xs, a: np.concatenate(xs, axis=a["axis"]))

register("cast", "misc", lambda ts, a: ts[0].with_dtype(a["dtype"]),
         lambda xs, a: xs[0].astype(NP_DTYPES[a["dtype"]]))


# ---------------------------------------------------------------- ops no NPU backend supports
# (they exist to exercise CPU fallback)

def _axis_infer_named(name):
    def infer(ts, a):
        norm_axis(a["axis"], ts[0].rank, name)
        return ts[0]
    return infer


register("sort", "misc", _axis_infer_named("sort"), lambda xs, a: np.sort(xs[0], axis=a["axis"]))
register("cumprod", "misc", _axis_infer_named("cumprod"),
         lambda xs, a: np.cumprod(f32(xs[0]), axis=a["axis"]).astype(xs[0].dtype))


# ---------------------------------------------------------------- selection and data movement

def _where_infer(ts, a):
    dt = same_dtype(ts[1:], "where")      # the condition may be any dtype (nonzero = true)
    shape = broadcast_shapes(broadcast_shapes(ts[0].shape, ts[1].shape, "where"), ts[2].shape, "where")
    return TensorType(dt, shape)


register("where", "misc", _where_infer,
         lambda xs, a: np.where(xs[0] != 0, xs[1], xs[2]).astype(xs[1].dtype))


def _broadcast_infer(ts, a):
    shape = tuple(a["shape"])
    if broadcast_shapes(ts[0].shape, shape, "broadcast") != shape:
        raise OpTypeError(f"broadcast: {ts[0]} can't be broadcast to [{', '.join(map(str, shape))}]")
    return TensorType(ts[0].dtype, shape)


register("broadcast", "shape", _broadcast_infer, lambda xs, a: np.broadcast_to(xs[0], a["shape"]).copy())


def _slice_infer(ts, a):
    x = ts[0]
    begin, size = tuple(a["begin"]), tuple(a["size"])
    if len(begin) != x.rank or len(size) != x.rank:
        raise OpTypeError(f"slice: begin and size need {x.rank} entries for {x}")
    for b, s, d in zip(begin, size, x.shape):
        if b < 0 or s < 0 or b + s > d:
            raise OpTypeError(f"slice: [{b}, {b + s}) is out of bounds for a dimension of size {d}")
    return TensorType(x.dtype, size)


register("slice", "shape", _slice_infer,
         lambda xs, a: xs[0][tuple(slice(b, b + s) for b, s in zip(a["begin"], a["size"]))].copy())


def _pad_infer(ts, a):
    x = ts[0]
    pads = tuple(a["pads"])
    if len(pads) != x.rank or any(lo < 0 or hi < 0 for lo, hi in pads):
        raise OpTypeError(f"pad: need {x.rank} non-negative (before, after) pairs for {x}")
    return TensorType(x.dtype, tuple(d + lo + hi for d, (lo, hi) in zip(x.shape, pads)))


register("pad", "shape", _pad_infer,
         lambda xs, a: np.pad(xs[0], a["pads"], constant_values=np.asarray(a.get("value", 0)).astype(xs[0].dtype)))


# ---------------------------------------------------------------- integer indexing

def _gather_infer(ts, a):
    table, idx = ts
    need_int(idx, "gather")
    ax = norm_axis(a["axis"], table.rank, "gather")
    return TensorType(table.dtype, table.shape[:ax] + idx.shape + table.shape[ax + 1:])


def _gather_ref(xs, a):
    table, idx = xs
    n = table.shape[a["axis"]]
    if idx.size and (idx.min() < -n or idx.max() >= n):
        raise IndexError(f"gather: index out of range for an axis of size {n}")
    return np.take(table, idx, axis=a["axis"])


register("gather", "index", _gather_infer, _gather_ref)


def _scatter_add_infer(ts, a):
    base, idx, upd = ts
    need_int(idx, "scatter_add")
    ax = norm_axis(a["axis"], base.rank, "scatter_add")
    want = base.shape[:ax] + idx.shape + base.shape[ax + 1:]
    if upd.shape != want or upd.dtype != base.dtype:
        raise OpTypeError(f"scatter_add: updates must be {TensorType(base.dtype, want)}, got {upd}")
    return base


def _scatter_add_ref(xs, a):
    """out = base; out[..., idx[k], ...] += updates[..., k, ...] (repeated indices accumulate)."""
    base, idx, upd = xs
    ax = a["axis"] % base.ndim
    out = _native_or_f32(base).copy()
    moved = np.moveaxis(out, ax, 0)                                 # a view with the axis first
    u = np.moveaxis(_native_or_f32(upd).reshape(base.shape[:ax] + (idx.size,) + base.shape[ax + 1:]), ax, 0)
    np.add.at(moved, idx.ravel(), u)
    return out.astype(base.dtype)


register("scatter_add", "index", _scatter_add_infer, _scatter_add_ref)


def _arg_infer_named(name):
    def infer(ts, a):
        x = ts[0]
        ax = norm_axis(a["axis"], x.rank, name)
        shape = tuple(1 if i == ax else d for i, d in enumerate(x.shape)) if a["keepdims"] else \
            tuple(d for i, d in enumerate(x.shape) if i != ax)
        return TensorType("i32", shape)
    return infer


register("argmax", "reduce", _arg_infer_named("argmax"),
         lambda xs, a: np.asarray(np.argmax(xs[0], axis=a["axis"], keepdims=a["keepdims"])).astype(np.int32))
register("argmin", "reduce", _arg_infer_named("argmin"),
         lambda xs, a: np.asarray(np.argmin(xs[0], axis=a["axis"], keepdims=a["keepdims"])).astype(np.int32))


# ---------------------------------------------------------------- slices at runtime positions
# Like XLA: the start position is a runtime i32 vector (one entry per axis), clamped so the
# slice stays in bounds. This is how fixed-size buffers (KV caches, token buffers) are read
# and written at a position only known while the program runs.

def _clamped_starts(start: np.ndarray, shape, size) -> list[int]:
    return [int(min(max(int(s), 0), d - n)) for s, d, n in zip(start.ravel(), shape, size)]


def _dyn_slice_infer(ts, a):
    x, start = ts
    need_int(start, "dynamic_slice", "start")
    size = tuple(a["size"])
    if start.shape != (x.rank,) or len(size) != x.rank or any(n > d or n < 0 for n, d in zip(size, x.shape)):
        raise OpTypeError(f"dynamic_slice: need start i32[{x.rank}] and a size that fits in {x}")
    return TensorType(x.dtype, size)


def _dyn_slice_ref(xs, a):
    x, start = xs
    st = _clamped_starts(start, x.shape, a["size"])
    return x[tuple(slice(s, s + n) for s, n in zip(st, a["size"]))].copy()


def _dyn_update_infer(ts, a):
    x, upd, start = ts
    need_int(start, "dynamic_update_slice", "start")
    if upd.dtype != x.dtype or upd.rank != x.rank or any(n > d for n, d in zip(upd.shape, x.shape)):
        raise OpTypeError(f"dynamic_update_slice: update {upd} doesn't fit in {x}")
    if start.shape != (x.rank,):
        raise OpTypeError(f"dynamic_update_slice: start must be i32[{x.rank}], got {start}")
    return x


def _dyn_update_ref(xs, a):
    x, upd, start = xs
    st = _clamped_starts(start, x.shape, upd.shape)
    out = x.copy()
    out[tuple(slice(s, s + n) for s, n in zip(st, upd.shape))] = upd
    return out


register("dynamic_slice", "index", _dyn_slice_infer, _dyn_slice_ref)
register("dynamic_update_slice", "index", _dyn_update_infer, _dyn_update_ref)


# ---------------------------------------------------------------- quantized constants

def _dequant_ref(xs, a):
    q, scale, axis = a["q"], a["scale"], a["axis"]
    shape = [1] * q.ndim
    shape[axis] = -1
    return (q.astype(np.float32) * scale.reshape(shape)).astype(NP_DTYPES[a["dtype"]])


# A compressed constant: int8 values with one scale per channel along `axis` (no runtime inputs).
register("dequantize", "const", lambda ts, a: TensorType(a["dtype"], tuple(a["q"].shape)), _dequant_ref)


# ---------------------------------------------------------------- int8 compute (W8A8)
#
# qmatmul / qconv2d multiply in 8-bit integers. The weight is stored as int8 with one scale per output
# channel (attrs q, w_scale); the activation x is quantized on the fly with a fixed per-tensor scale found
# by calibration (attr x_scale):
#     xq  = clip(round(x / x_scale), -127, 127)            int8
#     acc = xq @ q                                         exact int32 sums
#     y   = acc * x_scale * w_scale[out channel]           back to floating point, then the epilogue
# inputs = [x, epilogue operands...]; epilogue operand indices refer to this list.

def quantize_activation(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.round(f32(x) / np.float32(scale)), -127, 127)


def _qmatmul_infer(ts, a):
    x = ts[0]
    need_float([x], "qmatmul")
    k, n = a["q"].shape
    if x.rank < 2 or x.shape[-1] != k:
        raise OpTypeError(f"qmatmul: {x} doesn't match int8 weights of shape [{k}, {n}]")
    core = TensorType(a["dtype"], x.shape[:-1] + (n,))
    return epilogue_infer(core, ts, a.get("epilogue", ()), "qmatmul")


def _qmatmul_ref(xs, a):
    xq = quantize_activation(xs[0], a["x_scale"]).astype(np.int64)
    acc = np.matmul(xq, a["q"].astype(np.int64)).astype(np.float64)
    y = (acc * (np.float64(a["x_scale"]) * a["w_scale"].astype(np.float64))).astype(np.float32)
    return epilogue_ref(y, xs, a.get("epilogue", ())).astype(NP_DTYPES[a["dtype"]])


def _qconv2d_infer(ts, a):
    x = ts[0]
    need_float([x], "qconv2d")
    core = _conv2d_infer([x, TensorType(x.dtype, tuple(a["q"].shape))], {**a, "epilogue": ()})
    return epilogue_infer(core.with_dtype(a["dtype"]), ts, a.get("epilogue", ()), "qconv2d")


def _qconv2d_ref(xs, a):
    xq = quantize_activation(xs[0], a["x_scale"]).astype(np.float64)       # integer values, exact in float64
    acc = _conv2d_ref([xq, a["q"].astype(np.float64)], {**a, "epilogue": ()}).astype(np.float64)
    y = (acc * (np.float64(a["x_scale"]) * a["w_scale"].astype(np.float64)).reshape(1, -1, 1, 1)).astype(np.float32)
    return epilogue_ref(y, xs, a.get("epilogue", ())).astype(NP_DTYPES[a["dtype"]])


register("qmatmul", "matmul", _qmatmul_infer, _qmatmul_ref)
register("qconv2d", "conv", _qconv2d_infer, _qconv2d_ref)


# ---------------------------------------------------------------- gradient-only ops (created by autodiff)

def _conv2d_grad_input_infer(ts, a):
    return TensorType(ts[0].dtype, tuple(a["in_shape"]))


def _conv2d_grad_input_ref(xs, a):
    """dL/dx for y = conv2d(x, w): scatter each output gradient back through the kernel."""
    g, w = f32(xs[0]), f32(xs[1])
    (sh, sw), (ph, pw), (dh, dw), groups = conv_params(a)
    n, c, h, wd = a["in_shape"]
    o, cg, kh, kw = w.shape
    oh, ow = g.shape[2:]
    gg = g.reshape(n, groups, o // groups, oh, ow)
    wg = w.reshape(groups, o // groups, cg, kh, kw)
    dx = np.zeros((n, c, h + 2 * ph, wd + 2 * pw), np.float32)
    for i in range(kh):
        for j in range(kw):
            contrib = np.einsum("ngohw,goc->ngchw", gg, wg[:, :, :, i, j], optimize=True).reshape(n, c, oh, ow)
            dx[:, :, i * dh:i * dh + sh * (oh - 1) + 1:sh, j * dw:j * dw + sw * (ow - 1) + 1:sw] += contrib
    return dx[:, :, ph:ph + h, pw:pw + wd].astype(xs[0].dtype)


def _conv2d_grad_weight_infer(ts, a):
    return TensorType(ts[0].dtype, tuple(a["w_shape"]))


def _conv2d_grad_weight_ref(xs, a):
    """dL/dw for y = conv2d(x, w): correlate the input windows with the output gradient."""
    x, g = f32(xs[0]), f32(xs[1])
    groups = a.get("groups", 1)
    o, cg, kh, kw = a["w_shape"]
    n = x.shape[0]
    oh, ow = g.shape[2:]
    gg = g.reshape(n, groups, o // groups, oh, ow)
    dw = np.zeros((groups, o // groups, cg, kh, kw), np.float32)
    for i, j, xt in _taps(x, kh, kw, a, oh, ow):
        dw[:, :, :, i, j] = np.einsum("ngohw,ngchw->goc", gg, xt.reshape(n, groups, cg, oh, ow), optimize=True)
    return dw.reshape(o, cg, kh, kw).astype(xs[0].dtype)


def _maxpool2d_grad_ref(xs, a):
    """dL/dx for y = maxpool2d(x): route each gradient to the (first) max in its window."""
    x, g = f32(xs[0]), f32(xs[1])
    k, s = a["size"], a["stride"]
    oh, ow = g.shape[2:]
    y = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s].max(axis=(-2, -1))
    dx = np.zeros_like(x)
    taken = np.zeros(y.shape, bool)
    for i in range(k):
        for j in range(k):
            sl = (slice(None), slice(None), slice(i, i + s * oh, s), slice(j, j + s * ow, s))
            hit = (x[sl] == y) & ~taken
            dx[sl] += np.where(hit, g, 0)
            taken |= hit
    return dx.astype(xs[0].dtype)


register("conv2d_grad_input", "grad", _conv2d_grad_input_infer, _conv2d_grad_input_ref)
register("conv2d_grad_weight", "grad", _conv2d_grad_weight_infer, _conv2d_grad_weight_ref)
register("maxpool2d_grad", "grad", lambda ts, a: ts[0], _maxpool2d_grad_ref)


def infer(kind: str, types: list[TensorType], attrs: dict) -> TensorType:
    return REGISTRY[kind].infer(types, attrs)


def evaluate(kind: str, inputs: list[np.ndarray], attrs: dict) -> np.ndarray:
    return REGISTRY[kind].ref(inputs, attrs)
