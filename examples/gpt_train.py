"""Train the tiny GPT in examples/gpt.mira, then generate with its KV-cache decoding loop.

    python examples/gpt_train.py                    # train with the CPU reference backend
    python examples/gpt_train.py --target coreml    # gradients computed through Core ML
    python examples/gpt_train.py --target mlir      # ... or MLIR/IREE

The task needs attention: every sequence is a random 4-token pattern repeated
(e.g. 3 1 4 9 3 1 4 9 ...), so predicting the next token means looking back 4
positions. Given a 5-token prompt, a trained model continues the pattern.

Mira computes the loss and gradients (`grad` in gpt.mira); Adam runs here, on
the host, in NumPy, the way optimizers often run next to an accelerator.
"""
import argparse
import time

import numpy as np

import mira

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="cpu")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

V, D, F, T, B, PERIOD, PROMPT = 16, 32, 64, 16, 32, 4, 5
rng = np.random.default_rng(args.seed)


def batch(n):
    patterns = rng.integers(0, V, (n, PERIOD))
    return np.tile(patterns, (1, T // PERIOD)).astype(np.int32)


def init():
    def w(*shape, scale=None):
        return (rng.standard_normal(shape) * (scale or 1 / np.sqrt(shape[0]))).astype(np.float32)
    ones, zeros = np.ones(D, np.float32), np.zeros(D, np.float32)
    return {"emb": w(V, D, scale=0.5), "pos": w(T, D, scale=0.5), "wq": w(D, D), "wk": w(D, D), "wv": w(D, D),
            "wo": w(D, D), "w1": w(D, F), "w2": w(F, D), "g1": ones, "b1": zeros, "g2": ones.copy(),
            "b2": zeros.copy(), "wout": w(D, V)}


src = open("examples/gpt.mira").read()
t0 = time.perf_counter()
step = mira.compile_source(src, args.target, entry="train_step")
print(f"compiled train_step for {args.target} in {time.perf_counter() - t0:.2f}s")

params = init()
names = list(params)
m = {k: np.zeros_like(v) for k, v in params.items()}
v2 = {k: np.zeros_like(v) for k, v in params.items()}
b1, b2, eps = 0.9, 0.98, 1e-8
t0 = time.perf_counter()
for i in range(1, args.steps + 1):
    loss, *grads = step.run({"tok": batch(B), **params})
    for k, g in zip(names, grads):            # Adam
        g = np.asarray(g, np.float32)
        m[k] = b1 * m[k] + (1 - b1) * g
        v2[k] = b2 * v2[k] + (1 - b2) * g * g
        params[k] -= args.lr * (m[k] / (1 - b1 ** i)) / (np.sqrt(v2[k] / (1 - b2 ** i)) + eps)
    if i % 100 == 0 or i == 1:
        print(f"step {i:4d}  loss {float(loss):.4f}")
print(f"trained in {time.perf_counter() - t0:.1f}s")

gen = mira.compile_source(src, args.target, entry="generate", weights=params)
test = batch(50)
correct = total = 0
for seq in test:
    out = gen.run({"prompt": seq[:PROMPT]})
    correct += int((out[PROMPT:] == seq[PROMPT:]).sum())
    total += T - PROMPT
print(f"generation: {correct}/{total} continuation tokens correct ({correct / total:.1%})")
seq = test[0]
print("prompt   ", seq[:PROMPT].tolist())
print("generated", gen.run({"prompt": seq[:PROMPT]}).tolist())
print("expected ", seq.tolist())
