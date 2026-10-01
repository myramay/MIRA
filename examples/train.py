"""Train an MLP with a Mira-compiled training step.

    python examples/train.py --target coreml     # train on the Apple Neural Engine
    python examples/train.py --target npu-sim    # train on the simulated NPU
    python examples/train.py --target cpu

The data is synthetic: 10 Gaussian clusters in 64 dimensions.
"""
import argparse
import time

import numpy as np

import mira

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="coreml")
ap.add_argument("--steps", type=int, default=60)
args = ap.parse_args()

rng = np.random.default_rng(0)
B, D, H, C = 256, 64, 128, 10
centers = rng.standard_normal((C, D)).astype(np.float32) * 1.5


def batch(n=B):
    y = rng.integers(0, C, n)
    x = centers[y] + rng.standard_normal((n, D)).astype(np.float32)
    return x.astype(np.float32), np.eye(C, dtype=np.float32)[y], y


params = {
    "w1": (rng.standard_normal((D, H)) / np.sqrt(D)).astype(np.float32),
    "b1": np.zeros(H, np.float32),
    "w2": (rng.standard_normal((H, C)) / np.sqrt(H)).astype(np.float32),
    "b2": np.zeros(C, np.float32),
}

t0 = time.perf_counter()
step = mira.compile_file("examples/train_mlp.mira", args.target)
print(f"compiled training step for {args.target} in {time.perf_counter() - t0:.2f}s")
print(step.summary())

predict = mira.compile_source(
    open("examples/train_mlp.mira").read().split("fn main")[0] +
    "fn main(x: f32[1024, 64], w1: f32[64, 128], b1: f32[128], w2: f32[128, 10], b2: f32[10]) -> f32[1024, 10] {\n"
    "  return forward(x, w1, b1, w2, b2)\n}\n", "cpu")
xt, _, yt = batch(1024)

t0 = time.perf_counter()
for i in range(args.steps):
    x, onehot, _ = batch()
    loss, *new = step.run({"x": x, "labels": onehot, **params})
    params = dict(zip(params, (np.asarray(p, np.float32) for p in new)))
    if i % 10 == 0 or i == args.steps - 1:
        acc = (predict.run({"x": xt, **params}).argmax(1) == yt).mean()
        print(f"step {i:3d}  loss {float(loss):.4f}  test accuracy {acc:.1%}")
print(f"{args.steps} steps in {time.perf_counter() - t0:.2f}s")
