"""The tiny GPT: KV-cache decoding must agree with full recomputation, on every target, and it must learn."""
import subprocess
import sys

import numpy as np
import pytest

import mira
from conftest import ACCEL

SRC = open("examples/gpt.mira").read()
PROMPT = np.array([3, 1, 4, 1, 5], np.int32)


def test_kv_cache_decoding_matches_full_recompute():
    gen = mira.compile_source(SRC, "cpu", entry="generate", seed=7)
    fwd = mira.compile_source(SRC, "cpu", entry="forward", seed=7)       # same seed -> same random weights
    tokens = gen.run({"prompt": PROMPT})
    np.testing.assert_array_equal(tokens[:5], PROMPT)
    logits = fwd.run({"tok": tokens[None, :]})[0]
    for t in range(4, 15):     # every generated token is the argmax of the full model at the previous position
        assert tokens[t + 1] == logits[t].argmax(), t


@pytest.mark.parametrize("target", ACCEL)
def test_generation_on_accelerators(target):
    want = mira.compile_source(SRC, "cpu", entry="generate", seed=7).run({"prompt": PROMPT})
    got = mira.compile_source(SRC, target, entry="generate", seed=7).run({"prompt": PROMPT})
    assert (got == want).mean() >= 0.9          # fp16 may flip a near-tie; usually identical


def test_generation_loop_compiles_onto_device_with_mlir():
    pytest.importorskip("iree.compiler")
    gen = mira.compile_source(SRC, "mlir", entry="generate", seed=7)
    assert [s.device for s in gen.segments] == ["mlir"]
    assert any(op.kind == "while" for op in gen.graph.ops)


def test_training_learns_the_pattern_task():
    r = subprocess.run([sys.executable, "examples/gpt_train.py", "--steps", "400"], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("generation:"))
    correct, total = map(int, line.split()[1].split("/"))
    assert correct / total > 0.95, r.stdout
