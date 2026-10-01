import copy

import numpy as np

import mira
from mira import passes
from mira.backends.cpu import CPUExecutable
from mira.elaborate import elaborate
from mira.parser import parse


def graph(src: str, **kw):
    return elaborate(parse(src), seed=0, **kw)


def run(g, feeds):
    return CPUExecutable(g).run({v: feeds[v.name] for v in g.inputs})[g.outputs[0]]


def kinds(g):
    return [op.kind for op in g.compute_ops()]


def test_const_fold_folds_weight_only_math():
    g = graph("fn main(x: f32[2, 3], const w: f32[3, 3]) -> f32[2, 3] {\n"
              "  return x @ transpose(w * 2 + 1)\n}\n")
    passes.optimize(g, fuse=False)
    assert kinds(g) == ["matmul"]


def test_simplify_identities():
    g = graph("fn main(x: f32[4, 5]) -> f32[4, 5] {\n"
              "  return transpose(transpose(x)) * 1 + 0 - -(-0)\n}\n")
    passes.optimize(g)
    assert kinds(g) == []
    assert g.outputs[0] is g.inputs[0]


def test_cse_dedupes_identical_work():
    g = graph("fn main(x: f32[4, 4]) -> f32[4, 4] {\n  return relu(x @ x) + relu(x @ x)\n}\n")
    passes.optimize(g, fuse=False)
    assert kinds(g) == ["matmul", "relu", "add"]


def test_dce_removes_unused():
    g = graph("fn main(x: f32[4]) -> f32[4] {\n  let unused = exp(x)\n  return x * 3\n}\n")
    passes.optimize(g)
    assert kinds(g) == ["mul"]


def test_fusion_builds_epilogue():
    g = graph("fn main(x: f32[8, 16], const w: f32[16, 4], const b: f32[4]) -> f32[8, 4] {\n"
              "  return relu(x @ w + b)\n}\n")
    passes.optimize(g)
    (op,) = g.compute_ops()
    assert op.kind == "matmul"
    assert [s[0] for s in op.attrs["epilogue"]] == ["add", "relu"]


def test_fusion_respects_operand_order():
    g = graph("fn main(x: f32[8, 16], const w: f32[16, 4], const b: f32[8, 4]) -> f32[8, 4] {\n"
              "  return b - x @ w\n}\n")
    passes.optimize(g)
    (op,) = g.compute_ops()
    assert op.attrs["epilogue"][0][0] == "sub" and op.attrs["epilogue"][0][2] is True


def test_fusion_skips_multi_use_results():
    g = graph("fn main(x: f32[8, 16], const w: f32[16, 16]) -> f32[8, 16] {\n"
              "  let h = x @ w\n  return relu(h) + h\n}\n")
    passes.optimize(g)
    assert "epilogue" not in g.compute_ops()[0].attrs


def test_every_example_is_preserved_by_optimization(tmp_path):
    from conftest import MAIN_EXAMPLES
    for path in MAIN_EXAMPLES:
        src = open(path).read()
        g0 = elaborate(parse(src), seed=0)
        g1 = copy.deepcopy(g0)
        passes.optimize(g1)
        rng = np.random.default_rng(0)
        feeds = {v.name: rng.standard_normal(v.type.shape).astype(np.float32) for v in g0.inputs}
        np.testing.assert_allclose(run(g1, feeds), run(g0, feeds), rtol=1e-4, atol=1e-5, err_msg=path)


def test_to_f16_keeps_interface():
    p = mira.compile_source("fn main(x: f32[4, 4]) -> f32[4, 4] {\n  return x @ x\n}\n", precision="f16")
    assert p.graph.inputs[0].type.dtype == "f32" and p.graph.outputs[0].type.dtype == "f32"
    assert any(op.result.type.dtype == "f16" for op in p.graph.compute_ops())
