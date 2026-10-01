"""Integer tensors, indexing, gather/scatter, argmax, and runtime-position slices."""
import numpy as np
import pytest

import mira
from conftest import ACCEL
from test_autodiff import grad_check

PROGRAM = """
fn main(x: f32[6, 8], tok: i32[5], t: i32[], emb: f32[10, 8]) -> (f32[5, 8], i32[6], f32[6, 8], f32[2, 8], i32[]) {
  let e = emb[tok]
  let am = argmax(x @ transpose(emb[0:8]), axis=1)
  let y = x
  y[t] = x[t] * 3
  let d = dynamic_slice(x, [t, 0], [2, 8])
  return e, am, y, d, t + 1
}
"""


def feeds(t):
    rng = np.random.default_rng(t)
    return {"x": rng.standard_normal((6, 8)).astype(np.float32), "tok": np.array([1, 9, 1, 0, 4], np.int32),
            "t": np.array(t, np.int32), "emb": rng.standard_normal((10, 8)).astype(np.float32)}


def test_reference_semantics_match_numpy():
    f = feeds(2)
    e, am, y, d, t1 = mira.compile_source(PROGRAM).run(f)
    x, emb = f["x"], f["emb"]
    want_y = x.copy()
    want_y[2] = x[2] * 3
    np.testing.assert_allclose(e, emb[f["tok"]])
    np.testing.assert_array_equal(am, (x @ emb[:8].T).argmax(1))
    np.testing.assert_allclose(y, want_y)
    np.testing.assert_allclose(d, x[2:4])
    assert am.dtype == np.int32 and int(t1) == 3


@pytest.mark.parametrize("target", ACCEL)
@pytest.mark.parametrize("t", [0, 3, 5])     # 5: the [2, 8] slice clamps to start at row 4
def test_targets_match_reference(target, t):
    ref = mira.compile_source(PROGRAM).run(feeds(t))
    got = mira.compile_source(PROGRAM, target, cost_model=False).run(feeds(t))
    for g, r in zip(got, ref):
        assert g.dtype == r.dtype or (g.dtype.kind == "f" and r.dtype.kind == "f")
        tol = 0 if r.dtype.kind == "i" else 2e-2 * (np.abs(r).max() + 1)
        assert np.abs(g.astype(np.float64) - r).max() <= tol


def test_slicing_forms():
    src = """
fn main(x: f32[4, 6]) -> (f32[6], f32[4], f32[2, 3], f32[4, 2], f32[6]) {
  return x[-1], x[:, 2], x[1:3, :-1 + 4], x[:, -2:], x[2, :]
}"""
    x = np.arange(24, dtype=np.float32).reshape(4, 6)
    a, b, c, d, e = mira.compile_source(src).run({"x": x})
    np.testing.assert_array_equal(a, x[-1])
    np.testing.assert_array_equal(b, x[:, 2])
    np.testing.assert_array_equal(c, x[1:3, :3])
    np.testing.assert_array_equal(d, x[:, -2:])
    np.testing.assert_array_equal(e, x[2, :])


def test_scatter_add_accumulates_repeats():
    src = "fn main(i: i32[4], u: f32[4, 2]) -> f32[3, 2] {\n  return scatter_add(zeros([3, 2]), i, u)\n}\n"
    i = np.array([2, 0, 2, 2], np.int32)
    u = np.ones((4, 2), np.float32)
    np.testing.assert_array_equal(mira.compile_source(src).run({"i": i, "u": u}), [[1, 1], [0, 0], [3, 3]])


def test_integer_arithmetic_is_exact():
    src = "fn main(a: i32[3]) -> i32[3] {\n  return a * 3 + maximum(a, 5) - abs(-a)\n}\n"
    a = np.array([2 ** 29, -7, 6], np.int32)
    np.testing.assert_array_equal(mira.compile_source(src).run({"a": a}), a * 3 + np.maximum(a, 5) - np.abs(a))


def test_constructors():
    src = "fn main() -> (f32[2, 3], i32[4], i32[3], f32[2]) {\n" \
          "  return zeros([2, 3]) + 1, arange(4), arange(2, 5), full([2], 7.5)\n}\n"
    z, r, r2, f = mira.compile_source(src).run({})
    np.testing.assert_array_equal(z, np.ones((2, 3)))
    np.testing.assert_array_equal(r, [0, 1, 2, 3])
    np.testing.assert_array_equal(r2, [2, 3, 4])
    np.testing.assert_array_equal(f, [7.5, 7.5])


@pytest.mark.parametrize("src,msg", [
    ("fn main(t: i32[3]) -> i32[3] {\n  return t * 0.5\n}\n", "can't mix the number 0.5"),
    ("fn main(t: i32[3]) -> i32[3] {\n  return exp(t)\n}\n", "needs floating-point"),
    ("fn main(x: f32[3]) -> f32[3] {\n  return x[cast(x, f16)]\n}\n", "must be i32"),
    ("fn main(x: f32[3]) -> f32[2] {\n  return x[0, 1]\n}\n", "too many indices"),
    ("fn main(x: f32[3]) -> f32[3] {\n  return x[5]\n}\n", "out of range"),
    ("fn main(x: f32[3], i: i32[]) -> f32[3] {\n  return x[i:3]\n}\n", "compile-time"),
    ("fn main(x: f32[3]) -> f32[3] {\n  return grad(sum(x), cast(x, i32))\n}\n", "floating-point"),
])
def test_errors(src, msg):
    with pytest.raises(mira.MiraError, match=msg):
        mira.compile_source(src)


# gradients of the new ops (grad_check feeds float inputs; indices are compile-time constants here)
@pytest.mark.parametrize("expr,shapes", [
    ("gather(a, arange(3) * 2 - 1 + 1, axis=1)", {"a": (3, 6)}),
    ("a[arange(4) * 0 + 2] * b[1]", {"a": (3, 5), "b": (2, 5)}),
    ("scatter_add(a, arange(3), b, axis=0)", {"a": (4, 2), "b": (3, 2)}),
    ("dynamic_slice(a, [1, 2], [2, 3])", {"a": (4, 6)}),
    ("dynamic_update_slice(a, b, [2, 1])", {"a": (4, 5), "b": (2, 3)}),
    ("erf(a)", {"a": (3, 4)}),
])
def test_new_op_gradients(expr, shapes):
    grad_check(expr, shapes)


def test_embedding_training_step():
    """grad flows through an embedding lookup (gather -> scatter_add) into the table."""
    src = """
fn main(tok: i32[6], target: f32[6, 4], emb: f32[10, 4]) -> (f32[], f32[10, 4]) {
  let loss = mean((emb[tok] - target) ** 2)
  return loss, emb - 2.0 * grad(loss, emb)
}"""
    rng = np.random.default_rng(0)
    tok = np.array([1, 1, 3, 7, 3, 1], np.int32)
    target = rng.standard_normal((6, 4)).astype(np.float32)
    emb = np.zeros((10, 4), np.float32)
    p = mira.compile_source(src)
    losses = []
    for _ in range(30):
        loss, emb = p.run({"tok": tok, "target": target, "emb": emb})
        losses.append(float(loss))
    assert losses[-1] < 0.6 * losses[0]
    np.testing.assert_array_equal(emb[[0, 2, 4, 5, 6, 8, 9]], 0)     # rows never looked up never change
