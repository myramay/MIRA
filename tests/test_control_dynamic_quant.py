
import numpy as np
import pytest

import mira
from conftest import ACCEL

TARGETS = ["cpu"] + ACCEL


# ------------------------------------------------------------------ control flow

NEWTON = """
fn main(a: f32[8]) -> (f32[8], f32[1]) {
  let x = a
  let n = reshape(sum(a * 0), [1])
  while max(abs(x * x - a) / a) > 0.01 {
    x = 0.5 * (x + a / x)
    n = n + 1
  }
  return x, n
}
"""


@pytest.mark.parametrize("target", TARGETS)
def test_data_dependent_while(target):
    a = np.linspace(1, 50, 8).astype(np.float32)
    x, n = mira.compile_source(NEWTON, target).run({"a": a})
    np.testing.assert_allclose(x, np.sqrt(a), rtol=5e-3)
    assert 3 <= float(n[0]) <= 10


BRANCH = """
fn main(x: f32[16, 16], const w: f32[16, 16]) -> f32[16, 16] {
  let y = x
  if mean(x) > 0 {
    y = relu(x @ w)
  } else if mean(x) > -1 {
    y = x * 2
  } else {
    y = -x
  }
  return y
}
"""


@pytest.mark.parametrize("target", TARGETS)
def test_data_dependent_if(target):
    p = mira.compile_source(BRANCH, target)
    ref = mira.compile_source(BRANCH, "cpu")
    for shift in (1.0, -0.5, -3.0):
        x = (np.random.default_rng(0).standard_normal((16, 16)) * 0.1 + shift).astype(np.float32)
        np.testing.assert_allclose(p.run({"x": x}), ref.run({"x": x}), rtol=2e-2, atol=2e-2)


def test_loop_state_and_captures():
    # the body captures an outer tensor (w) and an outer constant; state has two tensors
    src = """
fn main(x: f32[4], w: f32[4]) -> (f32[4], f32[]) {
  let acc = x
  let total = sum(x * 0)
  let k = 3
  while total < 10 {
    acc = acc * w + k
    total = total + 1
  }
  return acc, total
}
"""
    x, w = np.ones(4, np.float32), np.full(4, 0.5, np.float32)
    acc, total = mira.compile_source(src).run({"x": x, "w": w})
    expect = x.copy()
    for _ in range(10):
        expect = expect * w + 3
    np.testing.assert_allclose(acc, expect, rtol=1e-6)
    assert float(total) == 10


def test_nested_control_flow():
    src = """
fn main(x: f32[3]) -> f32[3] {
  let y = x
  while sum(y) < 100 {
    if max(y) > 10 {
      y = y + 1
    } else {
      y = y * 2
    }
  }
  return y
}
"""
    y = np.array([1, 2, 3], np.float32)
    ref = y.copy()
    while ref.sum() < 100:
        ref = ref + 1 if ref.max() > 10 else ref * 2
    np.testing.assert_allclose(mira.compile_source(src).run({"x": y}), ref)


def test_compile_time_control_flow_is_resolved_statically():
    src = """
fn main(x: f32[4]) -> f32[4] {
  let y = x
  let n = 1
  while n < 20 {
    n = n * 2
  }
  if n == 32 {
    y = y * n
  } else {
    y = y - 1
  }
  return y
}
"""
    p = mira.compile_source(src)
    assert not any(op.is_control for op in p.graph.ops)
    np.testing.assert_allclose(p.run({"x": np.ones(4, np.float32)}), 32)


def test_comparisons_make_masks():
    src = "fn main(a: f32[4], b: f32[4]) -> f32[4] {\n  return (a < b) + (a == b) * 10 + (a != b) * 100\n}\n"
    a, b = np.array([1, 2, 3, 4], np.float32), np.array([2, 2, 1, 5], np.float32)
    np.testing.assert_allclose(mira.compile_source(src).run({"a": a, "b": b}), [101, 10, 100, 101])


def test_control_flow_errors():
    with pytest.raises(mira.MiraError, match="compile-time value"):
        mira.compile_source("fn main(x: f32[2]) -> f32[2] {\n  let n = 1\n  while sum(x) > 0 {\n"
                            "    n = n + 1\n  }\n  return x\n}\n")
    with pytest.raises(mira.MiraError, match="single value"):
        mira.compile_source("fn main(x: f32[2]) -> f32[2] {\n  if x > 0 {\n    x = x\n  }\n  return x\n}\n")
    with pytest.raises(mira.MiraError, match="last statement"):
        mira.compile_source("fn main(x: f32[2]) -> f32[2] {\n  if sum(x) > 0 {\n    return x\n  }\n  return x\n}\n")


# ------------------------------------------------------------------ dynamic shapes

DYN = "fn main(x: f32[B, 16], const w: f32[16, 8]) -> f32[B, 8] {\n  return relu(x @ w)\n}\n"


def test_dynamic_specializes_per_shape():
    d = mira.compile_source(DYN, "cpu")
    for n in (1, 5, 5, 9):
        assert d.run({"x": np.ones((n, 16), np.float32)}).shape == (n, 8)
    assert sorted(s["B"] for s in d.specializations) == [1, 5, 9]


@pytest.mark.parametrize("target", TARGETS)
def test_dynamic_buckets_pad_and_slice(target):
    d = mira.compile_source(DYN, target, buckets={"B": [8, 32]})
    ref = mira.compile_source(DYN, "cpu", dims={"B": 32})
    rng = np.random.default_rng(0)
    for n in (1, 7, 8, 20, 32):
        x = rng.standard_normal((n, 16)).astype(np.float32)
        expect = ref.run({"x": np.pad(x, ((0, 32 - n), (0, 0)))})[:n]
        np.testing.assert_allclose(d.run({"x": x}), expect, rtol=2e-2, atol=2e-2)
    assert sorted(s["B"] for s in d.specializations) == [8, 32]


def test_dynamic_const_needs_weights():
    with pytest.raises(mira.MiraError, match="no weight"):
        mira.compile_source("fn main(x: f32[B, N], const w: f32[N, 4]) -> f32[B, 4] {\n  return x @ w\n}\n")


def test_dynamic_weights_fix_shapes():
    src = "fn main(x: f32[B, N], const w: f32[N, 4]) -> f32[B, 4] {\n  return x @ w\n}\n"
    w = np.ones((6, 4), np.float32)
    d = mira.compile_source(src, weights={"w": w})
    np.testing.assert_allclose(d.run({"x": np.ones((3, 6), np.float32)}), 6)


# ------------------------------------------------------------------ int8 quantization

@pytest.mark.parametrize("target", TARGETS)
def test_int8_quantization_is_accurate_and_smaller(target):
    ref = mira.compile_file("examples/mlp.mira", "cpu")
    q = mira.compile_file("examples/mlp.mira", target, quantize="int8")
    kinds = [op.kind for op in q.graph.ops]
    assert kinds.count("dequantize") == 3
    x = np.random.default_rng(0).standard_normal((64, 784)).astype(np.float32)
    got, want = q.run({"x": x}), ref.run({"x": x})
    assert np.abs(got - want).max() < 0.05
    assert (got.argmax(1) == want.argmax(1)).mean() > 0.9
    log = dict(q.pass_log)["quantize-int8"]
    before, after = (float(t.split()[0]) for t in log.split("\n")[0].replace("// weights:", "").split("->"))
    assert after < before


def test_quantize_array_roundtrip():
    from mira.quantize import quantize_array
    w = np.random.default_rng(0).standard_normal((64, 32)).astype(np.float32)
    q, scale = quantize_array(w, axis=1)
    assert q.dtype == np.int8 and scale.shape == (32,)
    assert np.abs(q * scale - w).max() <= scale.max() / 2 + 1e-6


# ------------------------------------------------------------------ W8A8: int8 weights and int8 math

W8A8_CASES = [("examples/mlp.mira", "x", (64, 784), 0.9), ("examples/cnn.mira", "img", (16, 1, 28, 28), 0.9)]


@pytest.mark.parametrize("path,name,shape,agree", W8A8_CASES)
def test_w8a8_is_accurate(path, name, shape, agree):
    feeds = {name: np.random.default_rng(0).standard_normal(shape).astype(np.float32)}
    ref = mira.compile_file(path, "cpu").run(feeds)
    q = mira.compile_file(path, "cpu", quantize="w8a8")
    assert {"qmatmul", "qconv2d"} & {op.kind for op in q.graph.ops}
    got = q.run(feeds)
    assert (got.argmax(1) == ref.argmax(1)).mean() >= agree
    assert np.abs(got - ref).max() < 0.1 * np.abs(ref).max()


@pytest.mark.parametrize("target", ACCEL)
@pytest.mark.parametrize("path,name,shape,agree", W8A8_CASES)
def test_w8a8_targets_match_cpu(path, name, shape, agree, target):
    feeds = {name: np.random.default_rng(1).standard_normal(shape).astype(np.float32)}
    calib = [{name: np.random.default_rng(s).standard_normal(shape).astype(np.float32)} for s in (2, 3)]
    want = mira.compile_file(path, "cpu", quantize="w8a8", calibration=calib, precision="f16").run(feeds)
    p = mira.compile_file(path, target, quantize="w8a8", calibration=calib)
    got = p.run(feeds)
    assert np.abs(got - want).max() < 3e-2 * np.abs(want).max() + 1e-3, target
    assert (got.argmax(1) == want.argmax(1)).mean() >= 0.95


def test_w8a8_doubles_npu_sim_matrix_throughput():
    x = {"x": np.random.default_rng(0).standard_normal((64, 784)).astype(np.float32)}
    stats = []
    for q in (None, "w8a8"):
        p = mira.compile_file("examples/mlp.mira", "npu-sim", quantize=q)
        p.run(x)
        stats.append(next(s.executable.last_stats for s in p.segments if s.device == "npu-sim"))
    fp16, w8 = stats
    assert w8.int8_macs > 0 and w8.cycles < 0.8 * fp16.cycles and w8.dram_bytes < fp16.dram_bytes
