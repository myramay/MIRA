"""Differential tests: every target must agree with the CPU reference."""
import glob
import importlib.util
import platform

import numpy as np
import pytest

import mira

EXAMPLES = sorted(glob.glob("examples/*.mira"))
HAS_COREML = importlib.util.find_spec("coremltools") is not None and platform.system() == "Darwin"


def feeds_for(p, seed=1):
    rng = np.random.default_rng(seed)
    return {v.name: rng.standard_normal(v.type.shape).astype(v.type.np_dtype) for v in p.graph.inputs}


def rel_err(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


TARGETS = ["npu-sim"] + (["coreml"] if HAS_COREML else [])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("path", EXAMPLES)
def test_example_matches_cpu(path, target):
    ref = mira.compile_file(path, "cpu")
    feeds = feeds_for(ref)
    expected = ref.run(feeds)
    got = mira.compile_file(path, target).run(feeds)
    assert got.shape == expected.shape
    assert rel_err(got, expected) < 2e-2   # fp16 tolerance


def test_fallback_partitions_around_unsupported_ops():
    p = mira.compile_file("examples/fallback.mira", "npu-sim")
    devices = [s.device for s in p.segments]
    assert "cpu" in devices and "npu-sim" in devices
    sort_op = next(op for op in p.graph.compute_ops() if op.kind == "sort")
    assert p.placement.device[sort_op] == "cpu"
    assert "sort" in p.placement.reason[sort_op]


def test_cost_model_keeps_tiny_segments_on_cpu():
    src = "fn main(x: f32[4, 4]) -> f32[4, 4] {\n  return relu(x) * 2\n}\n"
    p = mira.compile_source(src, "npu-sim")
    assert [s.device for s in p.segments] == ["cpu"]
    p2 = mira.compile_source(src, "npu-sim", cost_model=False)
    assert "npu-sim" in [s.device for s in p2.segments]


def test_all_ops_run_on_npu_sim_or_fall_back():
    src = """
fn main(x: f32[2, 3, 8, 8], const k: f32[4, 3, 3, 3], const g: f32[16], const b: f32[16]) -> f32[2, 4, 16] {
  let c = conv2d(x, k, stride=1, padding=1)
  let p = maxpool2d(c, size=2)
  let r = reshape(p, [2, 4, 16])
  let n = layernorm(r, g, b)
  let s = softmax(n * 2, axis=-1)
  let m = mean(s, axis=1, keepdims=true) + max(s, axis=[1], keepdims=true) - sum(s, axis=1, keepdims=true)
  return concat([m, m, m, m], axis=1) + exp(-abs(n)) + minimum(n, 0.5) + maximum(n, -0.5) ** 2
}
"""
    ref = mira.compile_source(src, "cpu")
    feeds = feeds_for(ref)
    for target in TARGETS:
        got = mira.compile_source(src, target).run(feeds)
        assert rel_err(got, ref.run(feeds)) < 2e-2, target


@pytest.mark.skipif(not HAS_COREML, reason="needs macOS + coremltools")
@pytest.mark.parametrize("m,k,n", [(60, 48, 44), (64, 64, 64)])
def test_coreml_matmul_then_transpose(m, k, n):
    # Regression test for a Core ML bug the fuzzer found: const-weight fp16 matmul -> transpose
    src = f"fn main(x: f32[{m}, {k}], const w: f32[{k}, {n}]) -> f32[{n}, {m}] {{\n  return transpose(x @ w)\n}}\n"
    ref = mira.compile_source(src, "cpu")
    feeds = feeds_for(ref)
    assert rel_err(mira.compile_source(src, "coreml").run(feeds), ref.run(feeds)) < 2e-2


@pytest.mark.skipif(not HAS_COREML, reason="needs macOS + coremltools")
def test_coreml_reports_neural_engine_placement():
    p = mira.compile_file("examples/mlp.mira", "coreml")
    ex = p.segments[0].executable
    devices = {ex.device_of(op.result) for op in p.graph.compute_ops()}
    assert devices & {"ANE", "CPU", "GPU"}, devices
