"""ONNX importer: run existing models (exported from PyTorch, TensorFlow, ...) through Mira.

ONNX is a graph of named tensors and nodes. We walk the nodes in order and keep,
for every tensor name, either

  * a NumPy array, when its value is known at import time (weights, constants,
    and anything computed only from them, including tensor *shapes*), or
  * an IR value, when it depends on a runtime input.

Nodes whose inputs are all known are evaluated right here with NumPy. That folds
away the shape arithmetic exporters emit (Shape -> Gather -> Concat -> Reshape),
so the graph that reaches the backends has static shapes, which NPUs need.
Every other node is translated into Mira ops, after which the normal pipeline
(optimizer, partitioner, backends) takes over.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .. import ir
from .. import ops as O
from ..errors import MiraError
from ..types import TensorType

Val = Union[np.ndarray, ir.Value]

# ONNX TensorProto element types -> Mira dtypes (64-bit ints and bools become i32)
ONNX_DTYPES = {1: "f32", 10: "f16", 11: "f32", 6: "i32", 7: "i32", 9: "i32", 2: "i32", 3: "i32", 4: "i32", 5: "i32",
               12: "i32", 13: "i32"}
NP_OF = {1: np.float32, 10: np.float16, 11: np.float64, 6: np.int32, 7: np.int64, 9: np.bool_, 2: np.uint8,
         3: np.int8, 4: np.uint16, 5: np.int16, 12: np.uint32, 13: np.uint64}


def _to_mira_array(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind == "f":
        return a.astype(np.float16 if a.dtype == np.float16 else np.float32)
    return a.astype(np.int32)


class ImportError_(MiraError):
    pass


class Importer:
    def __init__(self, model, input_shapes: Optional[dict[str, tuple[int, ...]]] = None, name: str = "main"):
        import onnx
        from onnx import helper, numpy_helper
        self.helper, self.numpy_helper = helper, numpy_helper
        self.model = model if not isinstance(model, (str, bytes)) else onnx.load(model)
        self.g = ir.Graph(name, [], [], [])
        self.env: dict[str, Val] = {}
        self.input_shapes = input_shapes or {}
        self.opset = max((o.version for o in self.model.opset_import if o.domain in ("", "ai.onnx")), default=13)

    # ----- helpers

    def attrs(self, node) -> dict:
        return {a.name: self.helper.get_attribute_value(a) for a in node.attribute}

    def known(self, name: str) -> bool:
        return isinstance(self.env.get(name), np.ndarray)

    def K(self, name: str, what: str = "") -> np.ndarray:
        v = self.env.get(name)
        if not isinstance(v, np.ndarray):
            raise ImportError_(f"{what or name} must be known at import time (a constant), but depends on an input")
        return v

    def T(self, name: str, like: Optional[ir.Value] = None) -> ir.Value:
        """The tensor `name` as an IR value (constants become const ops, in `like`'s dtype if given)."""
        v = self.env[name]
        if isinstance(v, ir.Value):
            return v
        a = _to_mira_array(v)
        return self.g.const(a.astype(like.type.np_dtype) if like is not None else a)

    def op(self, kind: str, inputs: list[ir.Value], **attrs) -> ir.Value:
        try:
            return self.g.add(kind, inputs, attrs)
        except O.OpTypeError as e:
            raise ImportError_(str(e)) from None

    def scalar(self, x: float, like: ir.Value) -> ir.Value:
        return self.g.const(np.array(x, dtype=like.type.np_dtype))

    def binary(self, kind: str, a: str, b: str) -> ir.Value:
        va, vb = self.env[a], self.env[b]
        anchor = va if isinstance(va, ir.Value) else vb
        x, y = self.T(a, anchor), self.T(b, anchor)
        if x.type.dtype != y.type.dtype:
            y = self.op("cast", [y], dtype=x.type.dtype)
        return self.op(kind, [x, y])

    def reshape(self, x: ir.Value, shape) -> ir.Value:
        shape = tuple(int(d) for d in shape)
        return x if shape == x.type.shape else self.op("reshape", [x], shape=shape)

    def mask_not(self, m: ir.Value) -> ir.Value:
        return self.op("sub", [self.scalar(1, m), m])

    # ----- driver

    def run(self) -> ir.Graph:
        graph = self.model.graph
        for init in graph.initializer:
            self.env[init.name] = self.numpy_helper.to_array(init)
        for inp in graph.input:
            if inp.name in self.env:
                continue
            tt = inp.type.tensor_type
            dims = []
            for i, d in enumerate(tt.shape.dim):
                dims.append(d.dim_value if d.HasField("dim_value") else None)
            given = self.input_shapes.get(inp.name)
            if given is not None:
                if len(given) != len(dims) or any(a is not None and a != b for a, b in zip(dims, given)):
                    raise ImportError_(f"input '{inp.name}' has shape {dims} in the model; got {list(given)}")
                dims = list(given)
            if any(d is None for d in dims):
                raise ImportError_(f"input '{inp.name}' has unknown dimensions {dims}; give input_shapes")
            dtype = ONNX_DTYPES.get(tt.elem_type)
            if dtype is None:
                raise ImportError_(f"input '{inp.name}' has an unsupported element type ({tt.elem_type})")
            v = ir.Value(TensorType(dtype, tuple(dims)), name=inp.name)
            self.g.inputs.append(v)
            self.env[inp.name] = v
        for node in graph.node:
            self.node(node)
        for out in graph.output:
            self.g.outputs.append(self.T(out.name))
        self.g.verify()
        return self.g

    def node(self, node) -> None:
        ins = [n for n in node.input]
        a = self.attrs(node)
        op = node.op_type
        if node.domain not in ("", "ai.onnx"):
            raise ImportError_(f"custom op '{node.domain}.{op}' isn't supported")
        present = [n for n in ins if n]
        if op == "Constant":
            outs = [self.constant(a)]
        elif op in NUMPY and all(self.known(n) for n in present):
            outs = NUMPY[op](self, [self.env[n] if n else None for n in ins], a)   # import-time evaluation
            outs = outs if isinstance(outs, (list, tuple)) else [outs]
        elif op in HANDLERS:
            outs = HANDLERS[op](self, ins, a)
            outs = outs if isinstance(outs, (list, tuple)) else [outs]
        else:
            raise ImportError_(f"ONNX op '{op}' isn't supported yet (node '{node.name}')")
        for name, v in zip(node.output, outs):
            if name:
                self.env[name] = v

    def constant(self, a: dict) -> np.ndarray:
        if "value" in a:
            return self.numpy_helper.to_array(a["value"])
        for k in ("value_float", "value_int"):
            if k in a:
                return np.array(a[k])
        for k in ("value_floats", "value_ints"):
            if k in a:
                return np.array(list(a[k]))
        raise ImportError_(f"unsupported Constant attributes {list(a)}")

    def axes_of(self, ins: list[str], a: dict, idx: int = 1) -> Optional[list[int]]:
        """Reduction/squeeze axes: an attribute in older opsets, an input in newer ones."""
        if "axes" in a:
            return [int(x) for x in a["axes"]]
        if len(ins) > idx and ins[idx]:
            return [int(x) for x in self.K(ins[idx], "axes").ravel()]
        return None


# ==================================================================== import-time (NumPy) evaluation

def _np_cast(imp, xs, a):
    return xs[0].astype(NP_OF.get(a["to"], np.float32))


def _np_slice(imp, xs, a):
    data, starts, ends = xs[0], xs[1], xs[2]
    axes = xs[3] if len(xs) > 3 and xs[3] is not None else np.arange(len(starts))
    steps = xs[4] if len(xs) > 4 and xs[4] is not None else np.ones(len(starts), np.int64)
    idx = [slice(None)] * data.ndim
    for s, e, ax, st in zip(starts, ends, axes, steps):
        idx[int(ax)] = slice(int(s), int(e), int(st))
    return data[tuple(idx)]


def _np_unsqueeze(imp, xs, a):
    axes = a.get("axes", xs[1] if len(xs) > 1 else None)
    out = xs[0]
    rank = out.ndim + len(axes)
    for ax in sorted(int(x) % rank for x in axes):
        out = np.expand_dims(out, ax)
    return out


def _np_squeeze(imp, xs, a):
    axes = a.get("axes", xs[1] if len(xs) > 1 and xs[1] is not None else None)
    return np.squeeze(xs[0], axis=None if axes is None else tuple(int(x) for x in axes))


def _np_reshape(imp, xs, a):
    shape = [int(d) if d != 0 else xs[0].shape[i] for i, d in enumerate(xs[1])]
    return xs[0].reshape(shape)


NUMPY = {
    "Identity": lambda imp, xs, a: xs[0],
    "Shape": lambda imp, xs, a: np.array(xs[0].shape[a.get("start", 0):a.get("end", None)], np.int64),
    "Size": lambda imp, xs, a: np.array(xs[0].size, np.int64),
    "Gather": lambda imp, xs, a: np.take(xs[0], xs[1], axis=a.get("axis", 0)),
    "Unsqueeze": _np_unsqueeze,
    "Squeeze": _np_squeeze,
    "Concat": lambda imp, xs, a: np.concatenate([np.atleast_1d(x) for x in xs if x is not None], axis=a["axis"]),
    "Cast": _np_cast,
    "ConstantOfShape": lambda imp, xs, a: np.full(tuple(int(d) for d in xs[0]),
                                                  imp.numpy_helper.to_array(a["value"]).ravel()[0]
                                                  if "value" in a else 0.0),
    "Range": lambda imp, xs, a: np.arange(xs[0], xs[1], xs[2]),
    "Slice": _np_slice,
    "Reshape": _np_reshape,
    "Transpose": lambda imp, xs, a: np.transpose(xs[0], a.get("perm")),
    "Expand": lambda imp, xs, a: xs[0] * np.ones(tuple(int(d) for d in xs[1]), xs[0].dtype),
    "Add": lambda imp, xs, a: xs[0] + xs[1], "Sub": lambda imp, xs, a: xs[0] - xs[1],
    "Mul": lambda imp, xs, a: xs[0] * xs[1],
    "Div": lambda imp, xs, a: (xs[0] // xs[1]) if xs[0].dtype.kind in "iu" else xs[0] / xs[1],
    "Mod": lambda imp, xs, a: np.fmod(xs[0], xs[1]) if a.get("fmod") else np.mod(xs[0], xs[1]),
    "Pow": lambda imp, xs, a: np.power(xs[0], xs[1]), "Neg": lambda imp, xs, a: -xs[0],
    "Sqrt": lambda imp, xs, a: np.sqrt(xs[0]), "Abs": lambda imp, xs, a: np.abs(xs[0]),
    "Floor": lambda imp, xs, a: np.floor(xs[0]), "Ceil": lambda imp, xs, a: np.ceil(xs[0]),
    "Equal": lambda imp, xs, a: xs[0] == xs[1], "Less": lambda imp, xs, a: xs[0] < xs[1],
    "Greater": lambda imp, xs, a: xs[0] > xs[1], "LessOrEqual": lambda imp, xs, a: xs[0] <= xs[1],
    "GreaterOrEqual": lambda imp, xs, a: xs[0] >= xs[1], "Not": lambda imp, xs, a: ~xs[0],
    "And": lambda imp, xs, a: xs[0] & xs[1], "Or": lambda imp, xs, a: xs[0] | xs[1],
    "Where": lambda imp, xs, a: np.where(xs[0], xs[1], xs[2]),
    "Min": lambda imp, xs, a: np.minimum.reduce(xs), "Max": lambda imp, xs, a: np.maximum.reduce(xs),
    "Trilu": lambda imp, xs, a: (np.triu if a.get("upper", 1) else np.tril)(xs[0], int(xs[1]) if len(xs) > 1 and
                                                                               xs[1] is not None else 0),
    "ReduceProd": lambda imp, xs, a: np.prod(xs[0], axis=None if not a.get("axes") else tuple(a["axes"]),
                                             keepdims=bool(a.get("keepdims", 1))),
}


# ==================================================================== runtime ops -> Mira IR

def _unary(kind):
    return lambda imp, ins, a: imp.op(kind, [imp.T(ins[0])])


def _binary(kind):
    return lambda imp, ins, a: imp.binary(kind, ins[0], ins[1])


def _compare(kind, swap=False):
    def h(imp, ins, a):
        x, y = (ins[1], ins[0]) if swap else (ins[0], ins[1])
        return imp.binary(kind, x, y)
    return h


def _variadic(kind):
    def h(imp, ins, a):
        acc = imp.T(ins[0])
        for n in ins[1:]:
            y = imp.T(n, acc)
            acc = imp.op(kind, [acc, y])
        return acc
    return h


def _matmul(imp, ins, a):
    x, w = imp.T(ins[0]), imp.T(ins[1])
    xs, ws = x.type.shape, w.type.shape
    x2 = imp.reshape(x, (1,) + xs) if len(xs) == 1 else x
    w2 = imp.reshape(w, ws + (1,)) if len(ws) == 1 else w
    out = imp.op("matmul", [x2, w2])
    shape = out.type.shape
    if len(xs) == 1:
        shape = shape[:-2] + shape[-1:]
    if len(ws) == 1:
        shape = shape[:-1]
    return imp.reshape(out, shape)


def _gemm(imp, ins, a):
    x = imp.T(ins[0])
    if a.get("transA", 0):
        x = imp.op("transpose", [x], perm=(1, 0))
    if imp.known(ins[1]):
        w = imp.env[ins[1]]
        w = imp.g.const(_to_mira_array(w.T if a.get("transB", 0) else w).astype(x.type.np_dtype))
    else:
        w = imp.T(ins[1])
        if a.get("transB", 0):
            w = imp.op("transpose", [w], perm=(1, 0))
    y = imp.op("matmul", [x, w])
    if a.get("alpha", 1.0) != 1.0:
        y = imp.op("mul", [y, imp.scalar(a["alpha"], y)])
    if len(ins) > 2 and ins[2]:
        c = imp.T(ins[2], y)
        if a.get("beta", 1.0) != 1.0:
            c = imp.op("mul", [c, imp.scalar(a["beta"], c)])
        y = imp.op("add", [y, c])
    return y


def _conv_pads(imp, x: ir.Value, a: dict, kernel) -> tuple[ir.Value, tuple[int, int]]:
    """Resolve ONNX padding (incl. auto_pad and asymmetric pads) to symmetric (ph, pw), padding x if needed."""
    strides = a.get("strides", [1, 1])
    auto = a.get("auto_pad", b"NOTSET")
    auto = auto.decode() if isinstance(auto, bytes) else auto
    h, w = x.type.shape[2:]
    if auto in ("SAME_UPPER", "SAME_LOWER"):
        pads = []
        for size, k, s in zip((h, w), kernel, strides):
            total = max((int(np.ceil(size / s)) - 1) * s + k - size, 0)
            lo = total // 2 if auto == "SAME_UPPER" else total - total // 2
            pads.append((lo, total - lo))
        top, left, bottom, right = pads[0][0], pads[1][0], pads[0][1], pads[1][1]
    elif auto == "VALID":
        top = left = bottom = right = 0
    else:
        top, left, bottom, right = (list(a.get("pads", [0, 0, 0, 0])) + [0, 0, 0, 0])[:4]
    if top == bottom and left == right:
        return x, (top, left)
    x = imp.op("pad", [x], pads=((0, 0), (0, 0), (top, bottom), (left, right)))
    return x, (0, 0)


def _conv(imp, ins, a):
    x = imp.T(ins[0])
    w = imp.T(ins[1], x)
    if a.get("group", 1) != 1:
        raise ImportError_("grouped / depthwise Conv (group > 1) isn't supported yet")
    if any(d != 1 for d in a.get("dilations", [1, 1])):
        raise ImportError_("dilated Conv isn't supported yet")
    if x.type.rank != 4:
        raise ImportError_("only 2-D convolutions (NCHW) are supported")
    x, pads = _conv_pads(imp, x, a, w.type.shape[2:])
    s = a.get("strides", [1, 1])
    y = imp.op("conv2d", [x, w], stride=(s[0], s[1]), padding=pads)
    if len(ins) > 2 and ins[2]:
        b = imp.T(ins[2], y)
        y = imp.op("add", [y, imp.reshape(b, (b.type.numel, 1, 1))])
    return y


def _maxpool(imp, ins, a):
    x = imp.T(ins[0])
    k = a["kernel_shape"]
    s = a.get("strides", [1, 1])
    if k[0] != k[1] or s[0] != s[1]:
        raise ImportError_("MaxPool: only square kernels and equal strides are supported")
    if any(a.get("pads", [0, 0, 0, 0])) or a.get("ceil_mode", 0) or any(d != 1 for d in a.get("dilations", [1, 1])):
        raise ImportError_("MaxPool: padding, ceil_mode and dilation aren't supported")
    return imp.op("maxpool2d", [x], size=k[0], stride=s[0])


def _avgpool(imp, ins, a):
    x = imp.T(ins[0])
    k, s = a["kernel_shape"], a.get("strides", a["kernel_shape"])
    n, c, h, w = x.type.shape
    if list(k) != list(s) or h % k[0] or w % k[1] or any(a.get("pads", [0, 0, 0, 0])):
        raise ImportError_("AveragePool: only non-overlapping windows (stride == kernel) without padding")
    r = imp.reshape(x, (n, c, h // k[0], k[0], w // k[1], k[1]))
    return imp.op("reduce_mean", [r], axes=(3, 5), keepdims=False)


def _global_avgpool(imp, ins, a):
    x = imp.T(ins[0])
    return imp.op("reduce_mean", [x], axes=tuple(range(2, x.type.rank)), keepdims=True)


def _batchnorm(imp, ins, a):
    x = imp.T(ins[0])
    gamma, beta, mean, var = (imp.K(n, "BatchNormalization parameters").astype(np.float64) for n in ins[1:5])
    scale = gamma / np.sqrt(var + a.get("epsilon", 1e-5))
    shift = beta - mean * scale
    shape = (x.type.shape[1],) + (1,) * (x.type.rank - 2)
    s = imp.g.const(scale.reshape(shape).astype(x.type.np_dtype))
    t = imp.g.const(shift.reshape(shape).astype(x.type.np_dtype))
    return imp.op("add", [imp.op("mul", [x, s]), t])     # folds into a preceding conv's epilogue


def _layernorm(imp, ins, a):
    x = imp.T(ins[0])
    axis = a.get("axis", -1) % x.type.rank
    eps = a.get("epsilon", 1e-5)
    norm_shape = x.type.shape[axis:]
    gamma = imp.T(ins[1], x)
    beta = imp.T(ins[2], x) if len(ins) > 2 and ins[2] else imp.g.const(np.zeros(norm_shape, x.type.np_dtype))
    if axis == x.type.rank - 1:
        return imp.op("layernorm", [x, imp.reshape(gamma, (norm_shape[0],)), imp.reshape(beta, (norm_shape[0],))],
                      eps=eps)
    axes = tuple(range(axis, x.type.rank))   # normalize over several trailing axes: decompose
    mu = imp.op("reduce_mean", [x], axes=axes, keepdims=True)
    xc = imp.op("sub", [x, mu])
    var = imp.op("reduce_mean", [imp.op("mul", [xc, xc])], axes=axes, keepdims=True)
    inv = imp.op("div", [imp.scalar(1, x), imp.op("sqrt", [imp.op("add", [var, imp.scalar(eps, x)])])])
    return imp.op("add", [imp.op("mul", [imp.op("mul", [xc, inv]), gamma]), beta])


def _softmax(imp, ins, a):
    x = imp.T(ins[0])
    axis = a.get("axis", -1 if imp.opset >= 13 else 1)
    if imp.opset < 13 and x.type.rank != 2 and axis % x.type.rank != x.type.rank - 1:
        raise ImportError_("Softmax before opset 13 flattens its input; only last-axis softmax is supported")
    return imp.op("softmax", [x], axis=axis)


def _log_softmax(imp, ins, a):
    x = imp.T(ins[0])
    ax = (a.get("axis", -1) % x.type.rank,)
    z = imp.op("sub", [x, imp.op("reduce_max", [x], axes=ax, keepdims=True)])
    lse = imp.op("log", [imp.op("reduce_sum", [imp.op("exp", [z])], axes=ax, keepdims=True)])
    return imp.op("sub", [z, lse])


def _gelu(imp, ins, a):
    x = imp.T(ins[0])
    approx = a.get("approximate", b"none")
    if (approx.decode() if isinstance(approx, bytes) else approx) == "tanh":
        return imp.op("gelu", [x])
    inner = imp.op("erf", [imp.op("mul", [x, imp.scalar(1 / np.sqrt(2), x)])])
    return imp.op("mul", [imp.op("mul", [x, imp.scalar(0.5, x)]), imp.op("add", [inner, imp.scalar(1, x)])])


def _clip(imp, ins, a):
    x = imp.T(ins[0])
    lo = a.get("min", None) if imp.opset < 11 else (imp.K(ins[1]).item() if len(ins) > 1 and ins[1] else None)
    hi = a.get("max", None) if imp.opset < 11 else (imp.K(ins[2]).item() if len(ins) > 2 and ins[2] else None)
    if lo is not None:
        x = imp.op("maximum", [x, imp.scalar(lo, x)])
    if hi is not None:
        x = imp.op("minimum", [x, imp.scalar(hi, x)])
    return x


def _leaky_relu(imp, ins, a):
    x = imp.T(ins[0])
    alpha = a.get("alpha", 0.01)
    return imp.op("where", [imp.op("greater", [x, imp.scalar(0, x)]), x, imp.op("mul", [x, imp.scalar(alpha, x)])])


def _hard_sigmoid(imp, x: ir.Value, alpha: float, beta: float) -> ir.Value:
    y = imp.op("add", [imp.op("mul", [x, imp.scalar(alpha, x)]), imp.scalar(beta, x)])
    return imp.op("minimum", [imp.op("maximum", [y, imp.scalar(0, x)]), imp.scalar(1, x)])


def _reshape(imp, ins, a):
    x = imp.T(ins[0])
    target = [int(d) for d in imp.K(ins[1], "Reshape shape")]
    shape = [x.type.shape[i] if d == 0 and not a.get("allowzero", 0) else d for i, d in enumerate(target)]
    if -1 in shape:
        known = int(np.prod([d for d in shape if d != -1]))
        shape[shape.index(-1)] = x.type.numel // known
    return imp.reshape(x, shape)


def _flatten(imp, ins, a):
    x = imp.T(ins[0])
    ax = a.get("axis", 1) % (x.type.rank + 1)
    lead = int(np.prod(x.type.shape[:ax]))
    return imp.reshape(x, (lead, x.type.numel // max(lead, 1)))


def _squeeze(imp, ins, a):
    x = imp.T(ins[0])
    axes = imp.axes_of(ins, a)
    axes = {ax % x.type.rank for ax in axes} if axes is not None else {i for i, d in enumerate(x.type.shape) if d == 1}
    return imp.reshape(x, [d for i, d in enumerate(x.type.shape) if i not in axes])


def _unsqueeze(imp, ins, a):
    x = imp.T(ins[0])
    axes = imp.axes_of(ins, a)
    rank = x.type.rank + len(axes)
    shape = list(x.type.shape)
    for ax in sorted(ax % rank for ax in axes):
        shape.insert(ax, 1)
    return imp.reshape(x, shape)


def _transpose(imp, ins, a):
    x = imp.T(ins[0])
    perm = tuple(a.get("perm", range(x.type.rank - 1, -1, -1)))
    return x if perm == tuple(range(x.type.rank)) else imp.op("transpose", [x], perm=perm)


def _concat(imp, ins, a):
    vals = [imp.env[n] for n in ins]
    anchor = next(v for v in vals if isinstance(v, ir.Value))
    xs = [imp.T(n, anchor) for n in ins]
    xs = [x if x.type.dtype == anchor.type.dtype else imp.op("cast", [x], dtype=anchor.type.dtype) for x in xs]
    return imp.op("concat", xs, axis=a["axis"])


def _split(imp, ins, a):
    x = imp.T(ins[0])
    ax = a.get("axis", 0) % x.type.rank
    if "split" in a:
        sizes = list(a["split"])
    elif len(ins) > 1 and ins[1]:
        sizes = [int(s) for s in imp.K(ins[1], "Split sizes")]
    else:
        n = a.get("num_outputs")
        d = x.type.shape[ax]
        sizes = [d // n] * n if n else None
    outs, start = [], 0
    for s in sizes:
        begin = tuple(start if i == ax else 0 for i in range(x.type.rank))
        size = tuple(s if i == ax else d for i, d in enumerate(x.type.shape))
        outs.append(imp.op("slice", [x], begin=begin, size=size))
        start += s
    return outs


def _slice(imp, ins, a):
    x = imp.T(ins[0])
    if imp.opset < 10:
        starts, ends, axes, steps = a["starts"], a["ends"], a.get("axes"), None
    else:
        starts, ends = imp.K(ins[1], "Slice starts"), imp.K(ins[2], "Slice ends")
        axes = imp.K(ins[3]) if len(ins) > 3 and ins[3] else None
        steps = imp.K(ins[4]) if len(ins) > 4 and ins[4] else None
    axes = list(range(len(starts))) if axes is None else [int(ax) for ax in axes]
    if steps is not None and any(int(s) != 1 for s in steps):
        raise ImportError_("Slice with steps other than 1 isn't supported")
    begin, size = [0] * x.type.rank, list(x.type.shape)
    for s, e, ax in zip(starts, ends, axes):
        ax %= x.type.rank
        d = x.type.shape[ax]
        s, e = int(s), int(e)
        s = max(0, min(d, s + d if s < 0 else s))
        e = max(0, min(d, e + d if e < 0 else e))
        begin[ax], size[ax] = s, max(e - s, 0)
    if tuple(size) == x.type.shape:
        return x
    return imp.op("slice", [x], begin=tuple(begin), size=tuple(size))


def _gather(imp, ins, a):
    ax = a.get("axis", 0)
    x = imp.T(ins[0])
    if imp.known(ins[1]):
        idx = imp.env[ins[1]].astype(np.int64)
        idx = np.where(idx < 0, idx + x.type.shape[ax % x.type.rank], idx)
        idx = imp.g.const(idx.astype(np.int32))
    else:
        idx = imp.T(ins[1])
        if idx.type.dtype != "i32":
            idx = imp.op("cast", [idx], dtype="i32")
    return imp.op("gather", [x, idx], axis=ax)


def _expand(imp, ins, a):
    x = imp.T(ins[0])
    shape = O.broadcast_shapes(x.type.shape, tuple(int(d) for d in imp.K(ins[1], "Expand shape")), "Expand")
    return x if shape == x.type.shape else imp.op("broadcast", [x], shape=shape)


def _where(imp, ins, a):
    c = imp.T(ins[0])
    va, vb = imp.env[ins[1]], imp.env[ins[2]]
    anchor = va if isinstance(va, ir.Value) else vb if isinstance(vb, ir.Value) else None
    x = imp.T(ins[1], anchor)
    y = imp.T(ins[2], x)
    return imp.op("where", [c, x, y])


def _cast(imp, ins, a):
    x = imp.T(ins[0])
    to = a["to"]
    if to == 9:   # bool: a 1/0 mask in the input's dtype
        return imp.mask_not(imp.op("equal", [x, imp.scalar(0, x)]))
    target = ONNX_DTYPES.get(to)
    if target is None:
        raise ImportError_(f"Cast to element type {to} isn't supported")
    return x if x.type.dtype == target else imp.op("cast", [x], dtype=target)


def _reduce(kind):
    def h(imp, ins, a):
        x = imp.T(ins[0])
        axes = imp.axes_of(ins, a)
        if axes is None or len(axes) == 0:
            if a.get("noop_with_empty_axes", 0):
                return x
            axes = list(range(x.type.rank))
        keep = bool(a.get("keepdims", 1))
        axes = tuple(ax % x.type.rank for ax in axes)
        if kind == "min":
            neg = imp.op("neg", [x])
            return imp.op("neg", [imp.op("reduce_max", [neg], axes=axes, keepdims=keep)])
        if kind == "l2":
            return imp.op("sqrt", [imp.op("reduce_sum", [imp.op("mul", [x, x])], axes=axes, keepdims=keep)])
        return imp.op(f"reduce_{kind}", [x], axes=axes, keepdims=keep)
    return h


def _arg(kind):
    def h(imp, ins, a):
        if a.get("select_last_index", 0):
            raise ImportError_(f"{kind} with select_last_index isn't supported")
        return imp.op(kind, [imp.T(ins[0])], axis=a.get("axis", 0), keepdims=bool(a.get("keepdims", 1)))
    return h


def _pad(imp, ins, a):
    x = imp.T(ins[0])
    mode = a.get("mode", b"constant")
    if (mode.decode() if isinstance(mode, bytes) else mode) != "constant":
        raise ImportError_("Pad: only constant mode is supported")
    pads = a["pads"] if "pads" in a else [int(p) for p in imp.K(ins[1], "Pad pads")]
    value = a.get("value", 0.0) if imp.opset < 11 else (imp.K(ins[2]).item() if len(ins) > 2 and ins[2] else 0.0)
    if value != 0:
        raise ImportError_("Pad: only zero padding is supported")
    r = x.type.rank
    if len(ins) > 3 and ins[3]:
        axes = [int(ax) % r for ax in imp.K(ins[3])]
        full = [0] * (2 * r)
        for i, ax in enumerate(axes):
            full[ax], full[ax + r] = pads[i], pads[i + len(axes)]
        pads = full
    return imp.op("pad", [x], pads=tuple((int(pads[i]), int(pads[i + r])) for i in range(r)))


def _reciprocal(imp, ins, a):
    x = imp.T(ins[0])
    return imp.op("div", [imp.scalar(1, x), x])


def _not(imp, ins, a):
    return imp.mask_not(imp.T(ins[0]))


HANDLERS = {
    # shapes are static, so these give import-time constants even for runtime tensors
    "Shape": lambda imp, ins, a: np.array(imp.env[ins[0]].type.shape[a.get("start", 0):a.get("end", None)], np.int64),
    "Size": lambda imp, ins, a: np.array(imp.env[ins[0]].type.numel, np.int64),
    "Identity": lambda imp, ins, a: imp.T(ins[0]),
    "Dropout": lambda imp, ins, a: imp.T(ins[0]),      # inference: identity
    **{k: _unary(v) for k, v in {"Relu": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh", "Exp": "exp", "Log": "log",
                                 "Sqrt": "sqrt", "Abs": "abs", "Neg": "neg", "Erf": "erf", "Sign": "sign"}.items()},
    **{k: _binary(v) for k, v in {"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div", "Pow": "pow"}.items()},
    "Equal": _compare("equal"), "Greater": _compare("greater"), "GreaterOrEqual": _compare("greater_equal"),
    "Less": _compare("greater", swap=True), "LessOrEqual": _compare("greater_equal", swap=True),
    "And": _binary("mul"), "Or": _binary("maximum"), "Not": _not,
    "Max": _variadic("maximum"), "Min": _variadic("minimum"),
    "MatMul": _matmul, "Gemm": _gemm, "Conv": _conv, "MaxPool": _maxpool, "AveragePool": _avgpool,
    "GlobalAveragePool": _global_avgpool, "BatchNormalization": _batchnorm, "LayerNormalization": _layernorm,
    "Softmax": _softmax, "LogSoftmax": _log_softmax, "Gelu": _gelu, "Clip": _clip, "LeakyRelu": _leaky_relu,
    "HardSigmoid": lambda imp, ins, a: _hard_sigmoid(imp, imp.T(ins[0]), a.get("alpha", 0.2), a.get("beta", 0.5)),
    "HardSwish": lambda imp, ins, a: imp.op("mul", [imp.T(ins[0]), _hard_sigmoid(imp, imp.T(ins[0]), 1 / 6, 0.5)]),
    "Reciprocal": _reciprocal,
    "Reshape": _reshape, "Flatten": _flatten, "Squeeze": _squeeze, "Unsqueeze": _unsqueeze, "Transpose": _transpose,
    "Concat": _concat, "Split": _split, "Slice": _slice, "Gather": _gather, "Expand": _expand, "Where": _where,
    "Cast": _cast, "Pad": _pad,
    "ReduceMean": _reduce("mean"), "ReduceSum": _reduce("sum"), "ReduceMax": _reduce("max"),
    "ReduceMin": _reduce("min"), "ReduceL2": _reduce("l2"),
    "ArgMax": _arg("argmax"), "ArgMin": _arg("argmin"),
}


def import_onnx(model, input_shapes: Optional[dict[str, tuple[int, ...]]] = None) -> ir.Graph:
    """Translate an ONNX model (path, bytes, or ModelProto) into a Mira IR graph."""
    return Importer(model, input_shapes).run()


def onnx_inputs(model) -> list[tuple[str, list[Optional[int]], str]]:
    """(name, dims with None for symbolic ones, dtype) for each runtime input of the model."""
    import onnx
    m = model if not isinstance(model, (str, bytes)) else onnx.load(model)
    inits = {i.name for i in m.graph.initializer}
    out = []
    for inp in m.graph.input:
        if inp.name in inits:
            continue
        tt = inp.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim]
        out.append((inp.name, dims, ONNX_DTYPES.get(tt.elem_type, "?")))
    return out
