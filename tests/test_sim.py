import numpy as np
import pytest

import mira
from mira.backends.npusim.codegen import choose_matmul_tiles
from mira.backends.npusim.machine import Config


def sim_run(src, feeds, **kw):
    p = mira.compile_source(src, "npu-sim", cost_model=False, **kw)
    out = p.run(feeds)
    seg = next(s for s in p.segments if s.device == "npu-sim")
    return out, seg.executable


@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (33, 65, 17), (100, 300, 70), (256, 1000, 130)])
def test_matmul_odd_sizes(m, k, n):
    src = f"fn main(a: f16[{m}, {k}], b: f16[{k}, {n}]) -> f16[{m}, {n}] {{\n  return a @ b\n}}\n"
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k)).astype(np.float16)
    b = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(np.float16)
    out, _ = sim_run(src, {"a": a, "b": b})
    np.testing.assert_allclose(out.astype(np.float32), a.astype(np.float32) @ b.astype(np.float32),
                               rtol=2e-2, atol=2e-2)


def test_batched_matmul():
    src = "fn main(a: f16[3, 20, 40], b: f16[3, 40, 10]) -> f16[3, 20, 10] {\n  return a @ b\n}\n"
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 20, 40)).astype(np.float16)
    b = (rng.standard_normal((3, 40, 10)) / 6).astype(np.float16)
    out, _ = sim_run(src, {"a": a, "b": b})
    np.testing.assert_allclose(out.astype(np.float32), np.matmul(a.astype(np.float32), b.astype(np.float32)),
                               rtol=2e-2, atol=2e-2)


def test_tiles_fit_on_chip():
    cfg = Config()
    for M, K, N in [(4096, 4096, 4096), (1, 50000, 1), (64, 784, 512)]:
        tm, tk, tn = choose_matmul_tiles(M, K, N, n_epi=2, slots=2, cfg=cfg)
        assert 2 * (tm * tk + tk * tn + 3 * tm * tn) <= cfg.sram_elems
        assert 2 * tm * tn <= cfg.acc_elems


def test_double_buffering_is_faster():
    src = open("examples/resnet_loop.mira").read()
    x = np.random.default_rng(0).standard_normal((256, 512)).astype(np.float32)
    cycles = {}
    for db in (True, False):
        p = mira.compile_source(src, "npu-sim", double_buffer=db)
        p.run({"x": x})
        cycles[db] = sum(s.executable.last_stats.cycles for s in p.segments if s.device == "npu-sim")
    assert cycles[True] < cycles[False]


def test_fusion_reduces_dram_traffic():
    src = "fn main(x: f16[128, 256], const w: f16[256, 256], const b: f16[256]) -> f16[128, 256] {\n" \
          "  return gelu(x @ w + b)\n}\n"
    x = np.random.default_rng(0).standard_normal((128, 256)).astype(np.float16)
    traffic = {}
    for fuse in (True, False):
        _, ex = sim_run(src, {"x": x}, fuse=fuse)
        traffic[fuse] = ex.last_stats.dram_bytes
    assert traffic[True] < traffic[False]


def test_scoreboard_orders_dependent_kernels():
    # h feeds the second matmul through DRAM; wrong ordering would read garbage
    src = "fn main(x: f16[64, 64], const w: f16[64, 64]) -> f16[64, 64] {\n  return (x @ w) @ w\n}\n"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 64)).astype(np.float16)
    p = mira.compile_source(src, "npu-sim", seed=3)
    w = p.graph.ops[[op.is_const for op in p.graph.ops].index(True)].attrs["value"].astype(np.float32)
    out = p.run({"x": x})
    np.testing.assert_allclose(out.astype(np.float32), x.astype(np.float32) @ w @ w, rtol=3e-2, atol=3e-2)


# ------------------------------------------------------------------ kernels added with the simulator upgrade

def sim_vs_cpu(src, feeds, tol=2e-2, **kw):
    p = mira.compile_source(src, "npu-sim", cost_model=False, **kw)
    got = p.run(feeds)
    want = mira.compile_source(src, "cpu", **kw).run(feeds)
    err = np.abs(got.astype(np.float32) - want.astype(np.float32)).max() / (np.abs(want).max() + 1e-3)
    assert err < tol, err
    return p


def on_npu(p, kind):
    return all(p.placement.device[op] == "npu-sim" for op in p.graph.compute_ops() if op.kind == kind)


@pytest.mark.parametrize("n,c,h,w,o,k,stride,pad", [
    (2, 3, 9, 7, 5, 3, 1, 1), (1, 4, 8, 8, 6, 3, 2, 0), (2, 1, 5, 6, 3, 1, 1, 0), (1, 2, 7, 7, 4, 5, 2, 2)])
def test_conv2d_implicit_gemm(n, c, h, w, o, k, stride, pad):
    src = (f"fn main(x: f32[{n}, {c}, {h}, {w}], const k: f32[{o}, {c}, {k}, {k}], const b: f32[{o}, 1, 1]) "
           f"-> f32[{n}, {o}, {(h + 2 * pad - k) // stride + 1}, {(w + 2 * pad - k) // stride + 1}] {{\n"
           f"  return relu(conv2d(x, k, stride={stride}, padding={pad}) + b)\n}}\n")
    x = np.random.default_rng(0).standard_normal((n, c, h, w)).astype(np.float32)
    p = sim_vs_cpu(src, {"x": x})
    assert on_npu(p, "conv2d")


def test_conv2d_int8_weights():
    src = ("fn main(x: f32[2, 8, 10, 10], const k: f32[16, 8, 3, 3]) -> f32[2, 16, 10, 10] {\n"
           "  return conv2d(x, k, padding=1)\n}\n")
    x = np.random.default_rng(0).standard_normal((2, 8, 10, 10)).astype(np.float32)
    p = sim_vs_cpu(src, {"x": x}, tol=4e-2, quantize="int8")
    assert any(op.kind == "dequantize" for op in p.graph.ops) and on_npu(p, "conv2d")


def test_matmul_int8_weights_halve_weight_traffic():
    src = "fn main(x: f16[64, 512], const w: f16[512, 512]) -> f16[64, 512] {\n  return x @ w\n}\n"
    x = np.random.default_rng(0).standard_normal((64, 512)).astype(np.float16)
    traffic = {}
    for q in (None, "int8"):
        p = sim_vs_cpu(src, {"x": x}, tol=4e-2, quantize=q)
        traffic[q] = next(s for s in p.segments if s.device == "npu-sim").executable.last_stats.dram_bytes
    assert traffic["int8"] < 0.75 * traffic[None]


def test_maxpool_and_row_reductions():
    src = """
fn main(x: f32[2, 3, 8, 10]) -> f32[2, 3, 4, 1] {
  let p = maxpool2d(x, size=2)
  return sum(p, axis=3, keepdims=true) + max(p, axis=-1, keepdims=true) - mean(p, axis=3, keepdims=true)
}
"""
    p = sim_vs_cpu(src, {"x": np.random.default_rng(1).standard_normal((2, 3, 8, 10)).astype(np.float32)})
    assert on_npu(p, "maxpool2d") and on_npu(p, "reduce_sum") and on_npu(p, "reduce_max")


@pytest.mark.parametrize("perm", [(1, 0), (0, 2, 1), (1, 0, 2), (2, 0, 1), (1, 2, 0), (0, 2, 1, 3), (3, 1, 0, 2)])
def test_transposes(perm):
    shape = (3, 5, 4, 6)[:len(perm)]
    out = [shape[p] for p in perm]
    src = (f"fn main(x: f32[{', '.join(map(str, shape))}]) -> f32[{', '.join(map(str, out))}] {{\n"
           f"  return transpose(x * 2, perm={list(perm)})\n}}\n")
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    p = sim_vs_cpu(src, {"x": x})
    assert on_npu(p, "transpose")


def test_column_broadcast_where_and_broadcast():
    src = """
fn main(x: f32[6, 40]) -> f32[6, 40] {
  let z = x - max(x, axis=1, keepdims=true)
  return where(z > -1, exp(z), broadcast(mean(x, axis=1, keepdims=true), [6, 40]))
}
"""
    p = sim_vs_cpu(src, {"x": np.random.default_rng(2).standard_normal((6, 40)).astype(np.float32)})
    assert on_npu(p, "where") and on_npu(p, "broadcast") and on_npu(p, "sub")


def test_fast_mode_matches_functional_timing():
    src = open("examples/mlp.mira").read()
    x = np.random.default_rng(0).standard_normal((64, 784)).astype(np.float32)
    stats = []
    for fast in (False, True):
        p = mira.compile_source(src, "npu-sim", sim_fast=fast)
        out = p.run({"x": x})
        stats.append((out, next(s for s in p.segments if s.device == "npu-sim").executable.last_stats))
    (o1, s1), (o2, s2) = stats
    assert s1.cycles == s2.cycles and s1.dram_bytes == s2.dram_bytes
    assert np.abs(o1 - o2).max() < 2e-2


def test_hardware_config_changes_timing():
    src = open("examples/mlp.mira").read()
    x = np.random.default_rng(0).standard_normal((64, 784)).astype(np.float32)
    cycles = {}
    for dim in ("32", "64"):
        p = mira.compile_source(src, "npu-sim", sim_config={"mxu_dim": dim})
        p.run({"x": x})
        cycles[dim] = next(s for s in p.segments if s.device == "npu-sim").executable.last_stats.cycles
    assert cycles["64"] < cycles["32"]
