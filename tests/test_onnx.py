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


# ------------------------------------------------------------------ vision models and newer ONNX features

HAS_TORCHVISION = HAS_TORCH and importlib.util.find_spec("torchvision") is not None


def _export(module, args, path, **kw):
    import contextlib
    import io
    import warnings
    import torch
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(module, args, path, dynamo=False, opset_version=17, **kw)
    with torch.no_grad():
        out = module(*args)
    return out.numpy() if hasattr(out, "numpy") else out


@pytest.fixture(scope="module")
def vision_models(tmp_path_factory):
    if not HAS_TORCHVISION:
        pytest.skip("needs torchvision")
    import torch
    import torchvision
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("vision")
    out = {}
    for name, ctor in [("resnet18", torchvision.models.resnet18), ("mobilenet_v2", torchvision.models.mobilenet_v2)]:
        m = ctor(weights=None).eval()
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.running_var.uniform_(0.5, 1.5)
        x = torch.randn(1, 3, 64, 64)
        path = str(d / f"{name}.onnx")
        out[name] = (path, x.numpy(), _export(m, (x,), path, input_names=["image"]))
    return out


@pytest.mark.parametrize("target", ["cpu"] + ACCEL)
@pytest.mark.parametrize("name", ["resnet18", "mobilenet_v2"])
def test_torchvision_models(vision_models, name, target):
    path, x, ref = vision_models[name]
    p = mira.compile_onnx(path, target)
    got = p.run({"image": x})
    tol = 1e-4 if target == "cpu" else 2e-2
    assert np.abs(got - ref).max() / np.abs(ref).max() < tol
    assert got.argmax() == ref.argmax()


@pytest.mark.parametrize("name,make,shape", [
    ("grouped+dilated conv", lambda nn: nn.Sequential(nn.Conv2d(8, 12, 3, padding=2, dilation=2, groups=4), nn.ReLU()),
     (2, 8, 9, 9)),
    ("depthwise conv", lambda nn: nn.Conv2d(6, 6, 5, stride=2, padding=2, groups=6), (1, 6, 11, 11)),
    ("maxpool pad + ceil", lambda nn: nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True), (1, 3, 10, 10)),
    ("avgpool no pad count", lambda nn: nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False), (1, 3, 9, 9)),
    ("avgpool ceil", lambda nn: nn.AvgPool2d(2, stride=2, ceil_mode=True), (1, 3, 9, 9)),
    ("reflect pad", lambda nn: nn.ReflectionPad2d((1, 2, 2, 1)), (1, 2, 5, 6)),
    ("replicate pad", lambda nn: nn.ReplicationPad2d(2), (1, 2, 4, 4)),
    ("constant pad 1.5", lambda nn: nn.ConstantPad2d((1, 0, 2, 1), 1.5), (1, 2, 4, 4)),
])
def test_onnx_layers_match_pytorch(tmp_path, name, make, shape):
    import torch
    torch.manual_seed(0)
    m = make(torch.nn).eval()
    x = torch.randn(*shape)
    path = str(tmp_path / "m.onnx")
    ref = _export(m, (x,), path, input_names=["x"])
    np.testing.assert_allclose(mira.compile_onnx(path, "cpu").run({"x": x.numpy()}), ref, rtol=1e-4, atol=1e-5)


def test_onnx_slice_with_steps(tmp_path):
    import torch

    class M(torch.nn.Module):
        def forward(self, x):
            return x[:, ::2, 1::3] * 2

    x = torch.randn(2, 7, 10)
    path = str(tmp_path / "m.onnx")
    ref = _export(M(), (x,), path, input_names=["x"])
    np.testing.assert_allclose(mira.compile_onnx(path, "cpu").run({"x": x.numpy()}), ref, rtol=1e-6)


def _model(nodes, inputs, outputs, inits=()):
    from onnx import helper
    g = helper.make_graph(nodes, "g", inputs, outputs, list(inits))
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


def test_onnx_if_runtime_condition(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    vi = helper.make_tensor_value_info
    then_g = helper.make_graph([helper.make_node("Mul", ["x", "x"], ["t"])], "then", [], [vi("t", TensorProto.FLOAT, [3])])
    else_g = helper.make_graph([helper.make_node("Neg", ["x"], ["e"])], "else", [], [vi("e", TensorProto.FLOAT, [3])])
    nodes = [helper.make_node("ReduceSum", ["x"], ["s"], keepdims=0),
             helper.make_node("Constant", [], ["zero"], value=helper.make_tensor("z", TensorProto.FLOAT, [], [0.0])),
             helper.make_node("Greater", ["s", "zero"], ["c"]),
             helper.make_node("If", ["c"], ["y"], then_branch=then_g, else_branch=else_g)]
    path = str(tmp_path / "if.onnx")
    onnx.save(_model(nodes, [vi("x", TensorProto.FLOAT, [3])], [vi("y", TensorProto.FLOAT, [3])]), path)
    p = mira.compile_onnx(path, "cpu")
    for x in (np.array([1, 2, 3], np.float32), np.array([-1, -2, 0.5], np.float32)):
        np.testing.assert_allclose(p.run({"x": x}), x * x if x.sum() > 0 else -x)


def test_onnx_loop_unrolled_with_scan_output(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    vi = helper.make_tensor_value_info
    body = helper.make_graph(
        [helper.make_node("Identity", ["cond_in"], ["cond_out"]),
         helper.make_node("Mul", ["acc_in", "x"], ["acc_out"]),
         helper.make_node("Identity", ["acc_out"], ["scan"])],
        "body", [vi("i", TensorProto.INT64, []), vi("cond_in", TensorProto.BOOL, []),
                 vi("acc_in", TensorProto.FLOAT, [2])],
        [vi("cond_out", TensorProto.BOOL, []), vi("acc_out", TensorProto.FLOAT, [2]), vi("scan", TensorProto.FLOAT, [2])])
    inits = [helper.make_tensor("M", TensorProto.INT64, [], [4]), helper.make_tensor("C", TensorProto.BOOL, [], [True]),
             helper.make_tensor("one", TensorProto.FLOAT, [2], [1.0, 1.0])]
    nodes = [helper.make_node("Loop", ["M", "C", "one"], ["final", "steps"], body=body)]
    path = str(tmp_path / "loop.onnx")
    onnx.save(_model(nodes, [vi("x", TensorProto.FLOAT, [2])],
                     [vi("final", TensorProto.FLOAT, [2]), vi("steps", TensorProto.FLOAT, [4, 2])], inits), path)
    x = np.array([2.0, -0.5], np.float32)
    final, steps = mira.compile_onnx(path, "cpu").run({"x": x})
    np.testing.assert_allclose(final, x ** 4)
    np.testing.assert_allclose(steps, np.stack([x ** k for k in range(1, 5)]))
