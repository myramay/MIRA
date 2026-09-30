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

from .types import NP_DTYPES, TensorType


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


def norm_axis(axis: int, rank: int, op: str) -> int:
    if not -rank <= axis < rank:
        raise OpTypeError(f"{op}: axis {axis} is out of range for a rank-{rank} tensor")
    return axis % rank


# ---------------------------------------------------------------- const

register("const", "const",
         lambda ts, a: TensorType({np.dtype(np.float16): "f16", np.dtype(np.float32): "f32"}[a["value"].dtype],
                                  tuple(a["value"].shape)),
         lambda xs, a: a["value"])


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
}

BINARY: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "pow": np.power,
    "maximum": np.maximum,
    "minimum": np.minimum,
}


def _unary_infer(ts, a):
    return ts[0]


def _binary_infer_named(name):
    def infer(ts, a):
        dt = same_dtype(ts, name)
        return TensorType(dt, broadcast_shapes(ts[0].shape, ts[1].shape, name))
    return infer


for _n, _f in UNARY.items():
    register(_n, "unary", _unary_infer,
             (lambda f: lambda xs, a: f(f32(xs[0])).astype(xs[0].dtype))(_f))

for _n, _f in BINARY.items():
    register(_n, "binary", _binary_infer_named(_n),
             (lambda f: lambda xs, a: f(f32(xs[0]), f32(xs[1])).astype(xs[0].dtype))(_f))


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

def _conv_out(size, k, stride, pad):
    return (size + 2 * pad - k) // stride + 1


def _conv2d_infer(ts, a):
    x, w = ts[0], ts[1]
    dt = same_dtype([x, w], "conv2d")
    if x.rank != 4 or w.rank != 4:
        raise OpTypeError(f"conv2d: expects input [N, C, H, W] and weight [O, C, KH, KW], got {x} and {w}")
    n, c, h, wd = x.shape
    o, ci, kh, kw = w.shape
    if c != ci:
        raise OpTypeError(f"conv2d: input has {c} channels but weight expects {ci}")
    (sh, sw), (ph, pw) = a["stride"], a["padding"]
    oh, ow = _conv_out(h, kh, sh, ph), _conv_out(wd, kw, sw, pw)
    if oh <= 0 or ow <= 0:
        raise OpTypeError(f"conv2d: kernel {kh}x{kw} is larger than padded input {h}x{wd}")
    core = TensorType(dt, (n, o, oh, ow))
    return epilogue_infer(core, ts, a.get("epilogue", ()), "conv2d")


def _conv2d_ref(xs, a):
    x, w = f32(xs[0]), f32(xs[1])
    (sh, sw), (ph, pw) = a["stride"], a["padding"]
    x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    kh, kw = w.shape[2:]
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]
    acc = np.einsum("nchwij,ocij->nohw", win, w, optimize=True)
    return epilogue_ref(acc, xs, a.get("epilogue", ())).astype(xs[0].dtype)


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
        axes = sorted(norm_axis(ax, x.rank, name) for ax in a["axes"])
        if a["keepdims"]:
            shape = tuple(1 if i in axes else d for i, d in enumerate(x.shape))
        else:
            shape = tuple(d for i, d in enumerate(x.shape) if i not in axes)
        return TensorType(x.dtype, shape)
    return infer


for _n, _f in REDUCE.items():
    register(_n, "reduce", _reduce_infer_named(_n),
             (lambda f: lambda xs, a: np.asarray(f(f32(xs[0]), axis=tuple(a["axes"]), keepdims=a["keepdims"]))
              .astype(xs[0].dtype))(_f))


def _layernorm_infer(ts, a):
    x, g, b = ts
    same_dtype(ts, "layernorm")
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


def infer(kind: str, types: list[TensorType], attrs: dict) -> TensorType:
    return REGISTRY[kind].infer(types, attrs)


def evaluate(kind: str, inputs: list[np.ndarray], attrs: dict) -> np.ndarray:
    return REGISTRY[kind].ref(inputs, attrs)
