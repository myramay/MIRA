"""Reverse-mode automatic differentiation over the IR.

`gradients(graph, y, xs)` appends the backward pass to `graph` and returns
dy/dx for each x. It's the same algorithm as PyTorch's autograd, done at
compile time on the graph:

  1. Find the ops that lie on a path from some x to y.
  2. Walk them in reverse order, carrying the adjoint (dy/d value) of each
     value. Each op's rule (its vector-Jacobian product, VJP) turns the
     adjoint of its result into adjoints for its inputs; adjoints of values
     used several times are summed.

The backward ops are ordinary IR ops, so everything downstream (optimizer,
partitioner, NPU backends) handles training exactly like inference.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from . import ir

GELU_C = 0.7978845608028654
GELU_K = 0.044715


class GradError(Exception):
    pass


class Builder:
    """Small helper for emitting backward ops with readable code."""

    def __init__(self, g: ir.Graph, loc):
        self.g, self.loc = g, loc

    def op(self, kind: str, *inputs: ir.Value, **attrs) -> ir.Value:
        return self.g.add(kind, list(inputs), attrs, self.loc)

    def scalar(self, x: float, like: ir.Value) -> ir.Value:
        return self.g.const(np.array(x, dtype=like.type.np_dtype), self.loc)

    def add(self, a, b):
        return self.op("add", a, b)

    def sub(self, a, b):
        return self.op("sub", a, b)

    def mul(self, a, b):
        return self.op("mul", a, b)

    def div(self, a, b):
        return self.op("div", a, b)

    def neg(self, a):
        return self.op("neg", a)

    def swap_last(self, a):
        perm = list(range(a.type.rank))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        return self.op("transpose", a, perm=tuple(perm))

    def unbroadcast(self, grad: ir.Value, shape: tuple[int, ...]) -> ir.Value:
        """Sum `grad` over the axes that broadcasting expanded, so it matches `shape`."""
        if grad.type.shape == shape:
            return grad
        lead = grad.type.rank - len(shape)
        axes = list(range(lead)) + [lead + i for i, d in enumerate(shape) if d == 1 and grad.type.shape[lead + i] != 1]
        if axes:
            grad = self.op("reduce_sum", grad, axes=tuple(axes), keepdims=True)
        return self.op("reshape", grad, shape=tuple(shape)) if grad.type.shape != shape else grad

    def expand(self, g: ir.Value, x_shape: tuple[int, ...], axes: tuple[int, ...], keepdims: bool) -> ir.Value:
        """Undo a reduction: reshape to keepdims form, then broadcast to x's shape."""
        kept = tuple(1 if i in axes else d for i, d in enumerate(x_shape))
        if g.type.shape != kept:
            g = self.op("reshape", g, shape=kept)
        return self.op("broadcast", g, shape=tuple(x_shape))


# ------------------------------------------------------------------ VJP rules
# Each rule gets (builder, op, adjoint of op.result, which inputs need grads)
# and returns one adjoint (or None) per input.

Rule = Callable[[Builder, ir.Op, ir.Value, list[bool]], list[Optional[ir.Value]]]
RULES: dict[str, Rule] = {}


def rule(*kinds: str):
    def deco(fn: Rule) -> Rule:
        for k in kinds:
            RULES[k] = fn
        return fn
    return deco


@rule("add")
def _add(b, op, g, need):
    x, y = op.inputs
    return [b.unbroadcast(g, x.type.shape), b.unbroadcast(g, y.type.shape)]


@rule("sub")
def _sub(b, op, g, need):
    x, y = op.inputs
    return [b.unbroadcast(g, x.type.shape), b.unbroadcast(b.neg(g), y.type.shape) if need[1] else None]


@rule("mul")
def _mul(b, op, g, need):
    x, y = op.inputs
    return [b.unbroadcast(b.mul(g, y), x.type.shape) if need[0] else None,
            b.unbroadcast(b.mul(g, x), y.type.shape) if need[1] else None]


@rule("div")
def _div(b, op, g, need):
    x, y = op.inputs
    return [b.unbroadcast(b.div(g, y), x.type.shape) if need[0] else None,
            b.unbroadcast(b.neg(b.div(b.mul(g, op.result), y)), y.type.shape) if need[1] else None]


@rule("pow")
def _pow(b, op, g, need):
    x, y = op.inputs
    gx = gy = None
    if need[0]:   # y * x ** (y - 1)
        gx = b.unbroadcast(b.mul(g, b.mul(y, b.op("pow", x, b.sub(y, b.scalar(1, y))))), x.type.shape)
    if need[1]:   # out * log(x)
        gy = b.unbroadcast(b.mul(g, b.mul(op.result, b.op("log", x))), y.type.shape)
    return [gx, gy]


@rule("maximum", "minimum")
def _minmax(b, op, g, need):
    x, y = op.inputs
    cmp = "greater_equal"
    mask = b.op(cmp, x, y) if op.kind == "maximum" else b.op(cmp, y, x)   # 1 where x wins
    other = b.sub(b.scalar(1, mask), mask)
    return [b.unbroadcast(b.mul(g, mask), x.type.shape) if need[0] else None,
            b.unbroadcast(b.mul(g, other), y.type.shape) if need[1] else None]


@rule("neg")
def _neg(b, op, g, need):
    return [b.neg(g)]


@rule("relu")
def _relu(b, op, g, need):
    x = op.inputs[0]
    return [b.mul(g, b.op("greater", x, b.scalar(0, x)))]


@rule("sigmoid")
def _sigmoid(b, op, g, need):
    s = op.result
    return [b.mul(g, b.mul(s, b.sub(b.scalar(1, s), s)))]


@rule("tanh")
def _tanh(b, op, g, need):
    t = op.result
    return [b.mul(g, b.sub(b.scalar(1, t), b.mul(t, t)))]


@rule("exp")
def _exp(b, op, g, need):
    return [b.mul(g, op.result)]


@rule("log")
def _log(b, op, g, need):
    return [b.div(g, op.inputs[0])]


@rule("sqrt")
def _sqrt(b, op, g, need):
    return [b.div(g, b.mul(b.scalar(2, op.result), op.result))]


@rule("abs")
def _abs(b, op, g, need):
    return [b.mul(g, b.op("sign", op.inputs[0]))]


@rule("gelu")
def _gelu(b, op, g, need):
    # d/dx 0.5 x (1 + tanh(u)),  u = c (x + k x^3)
    #   = 0.5 (1 + t) + 0.5 x (1 - t^2) c (1 + 3 k x^2)
    x = op.inputs[0]
    x2 = b.mul(x, x)
    u = b.mul(b.scalar(GELU_C, x), b.add(x, b.mul(b.scalar(GELU_K, x), b.mul(x2, x))))
    t = b.op("tanh", u)
    left = b.mul(b.scalar(0.5, x), b.add(b.scalar(1, x), t))
    du = b.mul(b.scalar(GELU_C, x), b.add(b.scalar(1, x), b.mul(b.scalar(3 * GELU_K, x), x2)))
    right = b.mul(b.mul(b.scalar(0.5, x), x), b.mul(b.sub(b.scalar(1, t), b.mul(t, t)), du))
    return [b.mul(g, b.add(left, right))]


@rule("sign", "greater", "greater_equal", "equal")
def _zero(b, op, g, need):
    return [None] * len(op.inputs)   # piecewise constant: zero gradient


@rule("where")
def _where(b, op, g, need):
    c, x, y = op.inputs
    zero = b.scalar(0, g)
    return [None,
            b.unbroadcast(b.op("where", c, g, zero), x.type.shape) if need[1] else None,
            b.unbroadcast(b.op("where", c, zero, g), y.type.shape) if need[2] else None]


@rule("matmul")
def _matmul(b, op, g, need):
    if op.attrs.get("epilogue"):
        raise GradError("can't differentiate a fused matmul (grad runs before fusion)")
    x, w = op.inputs
    return [b.unbroadcast(b.op("matmul", g, b.swap_last(w)), x.type.shape) if need[0] else None,
            b.unbroadcast(b.op("matmul", b.swap_last(x), g), w.type.shape) if need[1] else None]


@rule("conv2d")
def _conv2d(b, op, g, need):
    x, w = op.inputs
    a = dict(stride=op.attrs["stride"], padding=op.attrs["padding"])
    return [b.op("conv2d_grad_input", g, w, in_shape=x.type.shape, **a) if need[0] else None,
            b.op("conv2d_grad_weight", x, g, w_shape=w.type.shape, **a) if need[1] else None]


@rule("maxpool2d")
def _maxpool(b, op, g, need):
    return [b.op("maxpool2d_grad", op.inputs[0], g, size=op.attrs["size"], stride=op.attrs["stride"])]


@rule("softmax")
def _softmax(b, op, g, need):
    s = op.result
    dot = b.op("reduce_sum", b.mul(g, s), axes=(op.attrs["axis"],), keepdims=True)
    return [b.mul(s, b.sub(g, dot))]


@rule("reduce_sum")
def _reduce_sum(b, op, g, need):
    x = op.inputs[0]
    axes = tuple(ax % x.type.rank for ax in op.attrs["axes"])
    return [b.expand(g, x.type.shape, axes, op.attrs["keepdims"])]


@rule("reduce_mean")
def _reduce_mean(b, op, g, need):
    x = op.inputs[0]
    axes = tuple(ax % x.type.rank for ax in op.attrs["axes"])
    count = int(np.prod([x.type.shape[ax] for ax in axes]))
    return [b.div(b.expand(g, x.type.shape, axes, op.attrs["keepdims"]), b.scalar(count, g))]


@rule("reduce_max")
def _reduce_max(b, op, g, need):
    x = op.inputs[0]
    axes = tuple(ax % x.type.rank for ax in op.attrs["axes"])
    keep = op.attrs["keepdims"]
    mask = b.op("equal", x, b.expand(op.result, x.type.shape, axes, keep))
    count = b.expand(b.op("reduce_sum", mask, axes=axes, keepdims=True), x.type.shape, axes, True)
    return [b.div(b.mul(mask, b.expand(g, x.type.shape, axes, keep)), count)]   # ties share the gradient


@rule("layernorm")
def _layernorm(b, op, g, need):
    x, gamma, beta = op.inputs
    d = x.type.shape[-1]
    last = (x.type.rank - 1,)
    mu = b.op("reduce_mean", x, axes=last, keepdims=True)
    xc = b.sub(x, mu)
    var = b.op("reduce_mean", b.mul(xc, xc), axes=last, keepdims=True)
    inv = b.div(b.scalar(1, x), b.op("sqrt", b.add(var, b.scalar(op.attrs["eps"], x))))
    xhat = b.mul(xc, inv)
    gx = ggamma = gbeta = None
    if need[0]:
        dy = b.mul(g, gamma)
        m1 = b.op("reduce_mean", dy, axes=last, keepdims=True)
        m2 = b.op("reduce_mean", b.mul(dy, xhat), axes=last, keepdims=True)
        gx = b.mul(inv, b.sub(b.sub(dy, m1), b.mul(xhat, m2)))
    if need[1]:
        ggamma = b.unbroadcast(b.mul(g, xhat), (d,))
    if need[2]:
        gbeta = b.unbroadcast(g, (d,))
    return [gx, ggamma, gbeta]


@rule("transpose")
def _transpose(b, op, g, need):
    perm = op.attrs["perm"]
    inv = tuple(int(i) for i in np.argsort(perm))
    return [b.op("transpose", g, perm=inv)]


@rule("reshape")
def _reshape(b, op, g, need):
    return [b.op("reshape", g, shape=op.inputs[0].type.shape)]


@rule("broadcast")
def _broadcast(b, op, g, need):
    return [b.unbroadcast(g, op.inputs[0].type.shape)]


@rule("slice")
def _slice(b, op, g, need):
    x = op.inputs[0]
    pads = tuple((bg, d - bg - s) for bg, s, d in zip(op.attrs["begin"], op.attrs["size"], x.type.shape))
    return [b.op("pad", g, pads=pads)]


@rule("pad")
def _pad(b, op, g, need):
    x = op.inputs[0]
    return [b.op("slice", g, begin=tuple(lo for lo, _ in op.attrs["pads"]), size=x.type.shape)]


@rule("concat")
def _concat(b, op, g, need):
    ax = op.attrs["axis"] % g.type.rank
    out, start = [], 0
    for x, n in zip(op.inputs, need):
        size = x.type.shape
        begin = tuple(start if i == ax else 0 for i in range(g.type.rank))
        out.append(b.op("slice", g, begin=begin, size=size) if n else None)
        start += size[ax]
    return out


@rule("cast")
def _cast(b, op, g, need):
    return [b.op("cast", g, dtype=op.inputs[0].type.dtype)]


# ------------------------------------------------------------------ driver

def gradients(g: ir.Graph, y: ir.Value, xs: list[ir.Value], loc=None) -> list[ir.Value]:
    if y.type.numel != 1:
        raise GradError(f"grad needs a scalar (one-element) output to differentiate, got {y.type}")
    b = Builder(g, loc)

    # values that depend on any x (forward reachability)
    depends = set(xs)
    for op in g.ops:
        if any(v in depends for v in op.inputs):
            depends.add(op.result)

    adj: dict[ir.Value, ir.Value] = {y: g.const(np.ones(y.type.shape, dtype=y.type.np_dtype), loc)}
    stop = y.producer
    ops = g.ops[:g.ops.index(stop) + 1] if stop is not None else []
    for op in reversed(ops):
        gout = adj.get(op.result)
        if gout is None or op.is_const:
            continue
        need = [v in depends for v in op.inputs]
        if not any(need):
            continue
        fn = RULES.get(op.kind)
        if fn is None:
            raise GradError(f"'{op.kind}' is not differentiable")
        for v, n, gv in zip(op.inputs, need, fn(b, op, gout, need)):
            if not n or gv is None:
                continue
            if gv.type != v.type:
                raise AssertionError(f"grad of {op.kind} produced {gv.type} for input {v.type}")
            adj[v] = b.add(adj[v], gv) if v in adj else gv

    out = []
    for x in xs:
        if x in adj:
            out.append(adj[x])
        else:   # y doesn't depend on x
            out.append(g.const(np.zeros(x.type.shape, dtype=x.type.np_dtype), loc))
    return out
