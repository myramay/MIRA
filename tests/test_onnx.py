"""ONNX import: PyTorch models exported to ONNX must match PyTorch on every target."""
import importlib.util

import numpy as np
import pytest

import mira
from conftest import ACCEL

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("onnx") is not None
HAS_DYNAMO_EXPORT = HAS_TORCH and importlib.util.find_spec("onnxscript") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="needs torch + onnx")

EXPORTERS = [False] + ([True] if HAS_DYNAMO_EXPORT else [])
MODELS = ["mlp", "cnn", "transformer"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    from onnx_models import export
    out = {}
    d = tmp_path_factory.mktemp("onnx")
    for name in MODELS:
        for dynamo in EXPORTERS:
            path = str(d / f"{name}_{dynamo}.onnx")
            args, ref, _ = export(name, path, dynamo)
            out[(name, dynamo)] = (path, args, ref)
    return out


def _run(path, target, args):
    p = mira.compile_onnx(path, target)
    got = p.run({v.name: a for v, a in zip(p.graph.inputs, args)})
    return p, (got if isinstance(got, tuple) else (got,))


@pytest.mark.parametrize("dynamo", EXPORTERS, ids=lambda d: "dynamo" if d else "torchscript")
@pytest.mark.parametrize("name", MODELS)
def test_cpu_matches_pytorch_exactly(exported, name, dynamo):
    path, args, ref = exported[(name, dynamo)]
    _, got = _run(path, "cpu", args)
    for g, r in zip(got, ref):
        assert g.shape == r.shape
        np.testing.assert_allclose(g, r, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("target", ACCEL)
@pytest.mark.parametrize("name", MODELS)
def test_accelerators_match_pytorch(exported, name, target):
    path, args, ref = exported[(name, EXPORTERS[-1])]
    p, got = _run(path, target, args)
    for g, r in zip(got, ref):
        if r.dtype.kind in "iu":            # argmax outputs: allow fp16 to flip a near-tie
            assert (g == r).mean() >= 0.9
        else:
            assert np.abs(g - r).max() / (np.abs(r).max() + 1e-9) < 2e-2
    assert target in {s.device for s in p.segments}


def test_batchnorm_folds_into_conv_epilogue(exported):
    path, _, _ = exported[("cnn", False)]
    g = mira.compile_onnx(path, "cpu").graph
    convs = [op for op in g.compute_ops() if op.kind == "conv2d"]
    assert convs and all(op.attrs.get("epilogue") for op in convs)


@pytest.mark.parametrize("dynamo", EXPORTERS, ids=lambda d: "dynamo" if d else "torchscript")
def test_dynamic_batch(tmp_path, dynamo):
    import torch
    from onnx_models import export
    path = str(tmp_path / "cnn_dyn.onnx")
    _, _, m = export("cnn", path, dynamo, dynamic_batch=True)
    p = mira.compile_onnx(path, "cpu")
    assert isinstance(p, mira.ShapeSpecialized)
    for b in (1, 4, 4, 2):
        x = torch.randn(b, 3, 16, 16)
        with torch.no_grad():
            want = m(x).numpy()
        np.testing.assert_allclose(p.run({p.input_names[0]: x.numpy()}), want, rtol=1e-4, atol=1e-5)
    assert len(p.specializations) == 3


def test_pinned_input_shapes(tmp_path):
    from onnx_models import export
    path = str(tmp_path / "cnn_dyn.onnx")
    export("cnn", path, False, dynamic_batch=True)
    p = mira.compile_onnx(path, "cpu", input_shapes={"x": (2, 3, 16, 16)})
    assert isinstance(p, mira.CompiledProgram) and p.graph.inputs[0].type.shape == (2, 3, 16, 16)


def test_unsupported_op_is_a_clear_error(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    node = helper.make_node("Hardmax", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = str(tmp_path / "bad.onnx")
    onnx.save(model, path)
    with pytest.raises(mira.MiraError, match="Hardmax"):
        mira.compile_onnx(path, "cpu")


def test_shape_arithmetic_is_folded_at_import(exported):
    """The transformer export computes reshape targets from Shape ops; none of that reaches the IR."""
    path, _, _ = exported[("transformer", False)]
    kinds = {op.kind for op in mira.compile_onnx(path, "cpu").graph.ops}
    assert not kinds & {"Shape", "Concat", "Gather"} and "matmul" in kinds
