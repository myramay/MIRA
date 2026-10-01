"""Differential tests: every target must agree with the CPU reference."""

import numpy as np
import pytest

import mira
from conftest import ACCEL, HAS_COREML, HAS_IREE, HAS_METAL, MAIN_EXAMPLES

EXAMPLES = MAIN_EXAMPLES


def feeds_for(p, seed=1):
    rng = np.random.default_rng(seed)
    return {v.name: rng.standard_normal(v.type.shape).astype(v.type.np_dtype) for v in p.graph.inputs}


def rel_err(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


TARGETS = ACCEL


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("path", EXAMPLES)
def test_example_matches_cpu(path, target):
    ref = mira.compile_file(path, "cpu")
    feeds = feeds_for(ref)
    expected = ref.run(feeds)
    got = mira.compile_file(path, target).run(feeds)
    expected = expected if isinstance(expected, tuple) else (expected,)
    got = got if isinstance(got, tuple) else (got,)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g.shape == e.shape
        assert rel_err(g, e) < 2e-2   # fp16 tolerance


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


@pytest.mark.skipif(not HAS_IREE, reason="needs iree-base-compiler")
def test_mlir_module_is_valid_upstream_mlir():
    import subprocess
    import sys
    from pathlib import Path
    from mira.backends.mlir import emit_module, supported
    iree_opt = Path(sys.executable).parent / "iree-opt"
    for path in EXAMPLES:
        g = mira.compile_file(path, "cpu").graph
        if any(supported(op) for op in g.compute_ops()):
            continue                   # e.g. sort: runs on the CPU, not in MLIR
        text = emit_module(g)
        r = subprocess.run([str(iree_opt), "-"], input=text, capture_output=True, text=True)
        assert r.returncode == 0, f"{path}: {r.stderr[:2000]}"


@pytest.mark.skipif(not HAS_IREE, reason="needs iree-base-compiler")
def test_mlir_compiles_control_flow_onto_the_device():
    src = ("fn main(x: f32[8]) -> f32[8] {\n  let y = x\n  while max(y) < 100 {\n    y = y * 2 + 1\n  }\n"
           "  return y\n}\n")
    p = mira.compile_source(src, "mlir", cost_model=False)
    assert [s.device for s in p.segments] == ["mlir"]
    x = np.linspace(1, 2, 8).astype(np.float32)
    np.testing.assert_allclose(p.run({"x": x}), mira.compile_source(src).run({"x": x}), rtol=2e-3)  # fp16


@pytest.mark.skipif(not HAS_METAL, reason="needs the Metal toolchain (xcodebuild -downloadComponent MetalToolchain)")
def test_mlir_metal_gpu():
    ref = mira.compile_file("examples/mlp.mira", "cpu")
    x = np.random.default_rng(0).standard_normal((64, 784)).astype(np.float32)
    got = mira.compile_file("examples/mlp.mira", "mlir", mlir_backend="metal").run({"x": x})
    assert rel_err(got, ref.run({"x": x})) < 2e-2


@pytest.mark.skipif(not HAS_COREML, reason="needs macOS + coremltools")
def test_coreml_survives_idle_cleanup():
    """Regression: Core ML frees the last prediction's NumPy inputs on a background thread when the
    model goes idle, which used to crash the interpreter a few seconds after a small prediction."""
    import subprocess
    import sys
    code = "\n".join([
        "import time, numpy as np, mira",
        "src = 'fn main(x: f32[8, 32], const w: f32[32, 32]) -> f32[8, 32] {\\n  return relu(x @ w)\\n}\\n'",
        "p = mira.compile_source(src, 'coreml')",
        "p.run({'x': np.ones((8, 32), np.float32)})",
        "time.sleep(3)",
        "print('survived')",
    ])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "survived" in r.stdout, r.stderr[-2000:]
