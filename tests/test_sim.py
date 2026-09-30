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
