"""Small PyTorch models exported to ONNX, used to test the importer against PyTorch itself."""
import contextlib
import io
import warnings

import torch

torch.manual_seed(0)

class MLP(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.net = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.LayerNorm(32),
                                    torch.nn.Linear(32, 10), torch.nn.LogSoftmax(-1))
    def forward(s, x):
        return s.net(x)

class CNN(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = torch.nn.Conv2d(3, 8, 3, padding=1); s.bn = torch.nn.BatchNorm2d(8)
        s.c2 = torch.nn.Conv2d(8, 16, 3, stride=2); s.fc = torch.nn.Linear(16, 10)
    def forward(s, x):
        x = torch.relu(s.bn(s.c1(x))); x = torch.nn.functional.max_pool2d(x, 2); x = torch.relu(s.c2(x))
        return s.fc(torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1))

class TF(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = torch.nn.Embedding(50, 32); s.pos = torch.nn.Parameter(torch.randn(12, 32) * 0.1)
        s.att = torch.nn.MultiheadAttention(32, 4, batch_first=True)
        s.ln1 = torch.nn.LayerNorm(32); s.ln2 = torch.nn.LayerNorm(32)
        s.ff = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(64, 32))
        s.out = torch.nn.Linear(32, 50)
    def forward(s, tok):
        h = s.emb(tok) + s.pos
        a, _ = s.att(s.ln1(h), s.ln1(h), s.ln1(h), need_weights=False)
        h = h + a
        h = h + s.ff(s.ln2(h))
        return s.out(h).argmax(-1), s.out(h)

def bn_init(m):
    """Non-trivial BatchNorm statistics, so folding them is actually tested."""
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5); mod.running_var.uniform_(0.5, 2); mod.weight.data.uniform_(0.5, 1.5)
    return m

MODELS = {
    "mlp": (MLP, lambda: (torch.randn(8, 16),)),
    "cnn": (lambda: bn_init(CNN()), lambda: (torch.randn(2, 3, 16, 16),)),
    "transformer": (TF, lambda: (torch.randint(0, 50, (2, 12)),)),
}

def export(name, path, dynamo, dynamic_batch=False):
    """Export model `name` to `path`; return (input arrays, PyTorch outputs, the module)."""
    m = MODELS[name][0]().eval()
    args = MODELS[name][1]()
    kw = {}
    if dynamic_batch:
        if dynamo:
            kw["dynamic_shapes"] = ({0: torch.export.Dim("batch", min=1, max=64)},)
        else:
            kw["input_names"] = ["x"]; kw["dynamic_axes"] = {"x": {0: "batch"}}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(m, args, path, dynamo=dynamo, opset_version=18, **kw)
    with torch.no_grad():
        ref = m(*args)
    ref = [r.numpy() for r in (ref if isinstance(ref, tuple) else (ref,))]
    return [a.numpy() for a in args], ref, m
