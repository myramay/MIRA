"""Gradient checks: every differentiable op's grad() must match finite differences."""

import numpy as np
import pytest

import mira
from conftest import ACCEL



def shape_str(shape):
    return "f32[" + ", ".join(map(str, shape)) + "]"


def grad_check(expr: str, shapes: dict[str, tuple], positive: tuple = (), eps: float = 1e-2, tol: float = 2e-2):
    """loss = sum(tanh(expr)); compare grad(loss, each input) with central differences (float64 math)."""
    params = ", ".join(f"{n}: {shape_str(s)}" for n, s in shapes.items())
    names = list(shapes)
    rets = ", ".join(["f32[]"] + [shape_str(s) for s in shapes.values()])
    src = (f"fn main({params}) -> ({rets}) {{\n  let loss = sum(tanh({expr}))\n"
           f"  return loss, {', '.join(f'grad(loss, {n})' for n in names)}\n}}\n")
    prog = mira.compile_source(src)
    fwd = mira.compile_source(f"fn main({params}) -> f32[] {{\n  return sum(tanh({expr}))\n}}\n")
    rng = np.random.default_rng(len(expr))
    ins = {}
    for n, s in shapes.items():
        a = rng.standard_normal(s)
        a = a + 0.1 * np.sign(a)        # stay away from kinks at 0 (relu, abs), where finite differences lie
        ins[n] = (np.abs(a) + 0.5 if n in positive else a).astype(np.float32)
    _, *grads = prog.run(ins)
    for n, g in zip(names, grads):
        num = np.zeros(shapes[n], np.float64)
        for idx in np.ndindex(*shapes[n]):
            hi = {k: v.copy() for k, v in ins.items()}
            lo = {k: v.copy() for k, v in ins.items()}
            hi[n][idx] += eps
            lo[n][idx] -= eps
            num[idx] = (float(fwd.run(hi)) - float(fwd.run(lo))) / (2 * eps)
        err = np.abs(g - num).max() / (np.abs(num).max() + 1e-3)
        assert err < tol, f"d/d{n} of {expr}: relative error {err:.3e}\nanalytic\n{g}\nnumeric\n{num}"


S = {"a": (3, 4), "b": (3, 4)}

CASES = [
    ("a + b", S, ()), ("a - b", S, ()), ("a * b", S, ()), ("a / b", S, ("b",)), ("a ** b", S, ("a",)),
    ("maximum(a, b)", S, ()), ("minimum(a, b)", S, ()), ("-a", {"a": (3, 4)}, ()),
    ("relu(a)", {"a": (3, 4)}, ()), ("sigmoid(a)", {"a": (3, 4)}, ()), ("tanh(a * 2)", {"a": (3, 4)}, ()),
    ("exp(a)", {"a": (3, 4)}, ()), ("log(a)", {"a": (3, 4)}, ("a",)), ("sqrt(a)", {"a": (3, 4)}, ("a",)),
    ("abs(a)", {"a": (3, 4)}, ()), ("gelu(a)", {"a": (3, 4)}, ()),
    ("where(a > 0, a * b, b)", S, ()),
    ("a + b", {"a": (3, 4), "b": (4,)}, ()),                  # broadcasting
    ("a * b", {"a": (2, 3, 4), "b": (3, 1)}, ()),
    ("a @ b", {"a": (3, 5), "b": (5, 4)}, ()),
    ("a @ b", {"a": (2, 3, 5), "b": (5, 4)}, ()),             # batched x shared weights
    ("a @ b", {"a": (2, 3, 5), "b": (2, 5, 4)}, ()),          # batched x batched
    ("softmax(a, axis=1)", {"a": (3, 4)}, ()), ("softmax(a, axis=0) * 3", {"a": (3, 4)}, ()),
    ("sum(a, axis=1)", {"a": (3, 4)}, ()), ("mean(a, axis=[0, 1], keepdims=true)", {"a": (3, 4)}, ()),
    ("max(a, axis=1, keepdims=true)", {"a": (3, 4)}, ()),
    ("layernorm(a, g, b)", {"a": (3, 4), "g": (4,), "b": (4,)}, ()),
    ("transpose(a, perm=[2, 0, 1])", {"a": (2, 3, 4)}, ()), ("reshape(a, [4, 3])", {"a": (3, 4)}, ()),
    ("broadcast(a, [3, 4])", {"a": (1, 4)}, ()),
    ("slice(a, begin=[1, 0], size=[2, 3])", {"a": (3, 4)}, ()),
    ("pad(a, [[1, 0], [0, 2]])", {"a": (3, 4)}, ()),
    ("concat([a, b], axis=1)", {"a": (3, 2), "b": (3, 4)}, ()),
    ("cast(cast(a, f16), f32)", {"a": (3, 4)}, ()),
    ("conv2d(x, w, stride=1, padding=1)", {"x": (2, 3, 5, 5), "w": (4, 3, 3, 3)}, ()),
    ("conv2d(x, w, stride=2, padding=0)", {"x": (1, 2, 7, 6), "w": (3, 2, 3, 2)}, ()),
    ("maxpool2d(x, size=2)", {"x": (1, 2, 4, 6)}, ()),
]


@pytest.mark.parametrize("expr,shapes,positive", CASES, ids=[c[0] for c in CASES])
def test_gradient(expr, shapes, positive):
    tol = 5e-2 if "cast" in expr else 2e-2
    grad_check(expr, shapes, positive, tol=tol)


def test_grad_of_unused_input_is_zero():
    src = "fn main(a: f32[3], b: f32[3]) -> f32[3] {\n  return grad(sum(a * a), b)\n}\n"
    np.testing.assert_array_equal(mira.compile_source(src).run({"a": np.ones(3, np.float32),
                                                                "b": np.ones(3, np.float32)}), 0)


def test_grad_needs_scalar():
    with pytest.raises(mira.MiraError, match="scalar"):
        mira.compile_source("fn main(a: f32[3]) -> f32[3] {\n  return grad(a * 2, a)\n}\n")


def test_grad_not_differentiable():
    with pytest.raises(mira.MiraError, match="not differentiable"):
        mira.compile_source("fn main(a: f32[3]) -> f32[3] {\n  return grad(sum(sort(a)), a)\n}\n")


def test_second_order_derivative():
    # d2/dx2 of sum(x^3) = 6x
    src = "fn main(x: f32[4]) -> f32[4] {\n  return grad(sum(grad(sum(x * x * x), x)), x)\n}\n"
    x = np.array([1, 2, -1, 0.5], np.float32)
    np.testing.assert_allclose(mira.compile_source(src).run({"x": x}), 6 * x, rtol=1e-5)


def _train(target, steps=25):
    src = open("examples/train_mlp.mira").read()
    step = mira.compile_source(src, target)
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((10, 64)).astype(np.float32) * 1.5
    params = {"w1": (rng.standard_normal((64, 128)) / 8).astype(np.float32), "b1": np.zeros(128, np.float32),
              "w2": (rng.standard_normal((128, 10)) / 11).astype(np.float32), "b2": np.zeros(10, np.float32)}
    losses = []
    for _ in range(steps):
        y = rng.integers(0, 10, 256)
        x = (centers[y] + rng.standard_normal((256, 64))).astype(np.float32)
        loss, *new = step.run({"x": x, "labels": np.eye(10, dtype=np.float32)[y], **params})
        params = dict(zip(params, (np.asarray(p, np.float32) for p in new)))
        losses.append(float(loss))
    return losses


@pytest.mark.parametrize("target", ["cpu"] + ACCEL)
def test_training_reduces_loss(target):
    losses = _train(target)
    assert losses[-1] < 0.1 * losses[0], losses
