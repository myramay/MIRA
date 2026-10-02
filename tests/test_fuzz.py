"""Fuzzing: generate random well-typed MIRA programs and check that the
optimizer and the NPU simulator agree with the unoptimized CPU reference.

When a seed fails, `print(generate(seed))` gives you the program to debug.
"""
import random

import numpy as np
import pytest

import mira
from conftest import HAS_COREML, HAS_IREE

UNARY = ["relu", "gelu", "sigmoid", "tanh", "abs"]
BINARY = ["+", "-", "*", "maximum", "minimum"]


def generate(seed: int) -> str:
    r = random.Random(seed)
    rows, cols = r.randint(1, 70), r.randint(1, 70)
    params = [f"x: f32[{rows}, {cols}]"]
    body = []
    live = [("x", (rows, cols))]
    n_w = 0

    def weight(shape):
        nonlocal n_w
        n_w += 1
        params.append(f"const w{n_w}: f32[{', '.join(map(str, shape))}]")
        return f"w{n_w}"

    for i in range(r.randint(2, 8)):
        name, shape = live[-1]
        choice = r.choice(["matmul", "matmul", "bias", "unary", "binary", "scalar", "softmax", "layernorm",
                           "reshape", "transpose", "rowmax", "where", "rowsum"])
        new = f"t{i}"
        if choice == "matmul":
            n = r.randint(1, 90)
            expr, shape = f"{name} @ {weight((shape[1], n))}", (shape[0], n)
            if r.random() < 0.5:
                expr = f"tanh({expr})"
        elif choice == "bias":
            expr = f"{name} + {weight((shape[1],))}"
        elif choice == "unary":
            expr = f"{r.choice(UNARY)}({name})"
        elif choice == "binary":
            same = [n for n, s in live if s == shape]
            other = r.choice(same)
            op = r.choice(BINARY)
            expr = f"{op}({name}, {other})" if op.isalpha() else f"{name} {op} {other}"
        elif choice == "scalar":
            expr = f"{name} * {r.choice([0.5, 2, -1])} + {r.choice([0, 1, 0.25])}"
        elif choice == "softmax":
            expr = f"softmax({name}, axis={r.choice([0, 1, -1])})"
        elif choice == "layernorm":
            expr = f"layernorm({name}, {weight((shape[1],))}, {weight((shape[1],))})"
        elif choice == "reshape":
            shape = (shape[1], shape[0])
            expr = f"reshape({name}, [{shape[0]}, {shape[1]}])"
        elif choice == "rowmax":
            expr = f"{name} - max({name}, axis=1, keepdims=true)"
        elif choice == "where":
            expr = f"where({name} > 0, {name}, {name} * 0.1)"
        elif choice == "rowsum":
            expr = f"{name} + mean({name}, axis=1, keepdims=true)"
        else:
            shape = (shape[1], shape[0])
            expr = f"transpose({name})"
        body.append(f"  let {new} = {expr}")
        live.append((new, shape))
    out, shape = live[-1]
    return (f"fn main({', '.join(params)}) -> f32[{shape[0]}, {shape[1]}] {{\n"
            + "\n".join(body) + f"\n  return {out}\n}}\n")


def check(seed: int, target: str, tol: float):
    src = generate(seed)
    ref = mira.compile_source(src, "cpu", optimize=False)
    rng = np.random.default_rng(seed)
    feeds = {v.name: rng.standard_normal(v.type.shape).astype(np.float32) for v in ref.graph.inputs}
    expected = ref.run(feeds)
    got = mira.compile_source(src, target, cost_model=False).run(feeds)
    err = np.abs(got.astype(np.float32) - expected).max() / (np.abs(expected).max() + 1e-3)
    assert err < tol, f"seed {seed} on {target}: relative error {err:.3e}\n{src}"


@pytest.mark.parametrize("seed", range(60))
def test_optimizer_preserves_semantics(seed):
    check(seed, "cpu", 1e-4)


@pytest.mark.parametrize("seed", range(60))
def test_npu_sim_matches_reference(seed):
    check(seed, "npu-sim", 3e-2)


@pytest.mark.skipif(not HAS_COREML, reason="needs macOS + coremltools")
@pytest.mark.parametrize("seed", range(20))
def test_coreml_matches_reference(seed):
    check(seed, "coreml", 3e-2)


@pytest.mark.skipif(not HAS_IREE, reason="needs iree-base-compiler")
@pytest.mark.parametrize("seed", range(20))
def test_mlir_matches_reference(seed):
    check(seed, "mlir", 3e-2)
