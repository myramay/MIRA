"""Train a character-level GPT on nursery rhymes, then let it write.

    python examples/gpt_text_train.py                  # train with the CPU reference backend
    python examples/gpt_text_train.py --target coreml  # gradients computed through Core ML

The model (examples/gpt_text.mira) reads one character at a time. MIRA computes the
loss and gradients; Adam runs here on the host, on fp32 master weights. Accelerator
targets compute in fp16, so the loss is scaled by 1024 before differentiating and the
gradients are divided back down here (mixed-precision training). After training,
generation runs as a single compiled program: a while loop with a KV cache per layer.
"""
import argparse
import time

import numpy as np

import mira

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="cpu")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

text = open("examples/data/rhymes.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], np.int32)
dims = {"V": len(chars), "D": 64, "F": 128, "L": 2, "T": 64, "B": 32}
V, D, F, L, T, B = (dims[k] for k in "VDFLTB")
rng = np.random.default_rng(args.seed)


def batch():
    starts = rng.integers(0, len(data) - T, B)
    return np.stack([data[s:s + T] for s in starts])


def w(*shape, fan_in=None):
    return (rng.standard_normal(shape) / np.sqrt(fan_in or shape[-2])).astype(np.float32)


params = {
    "emb": w(V, D, fan_in=4), "pos": w(T, D, fan_in=16),
    "wq": w(L, D, D), "wk": w(L, D, D), "wv": w(L, D, D), "wo": w(L, D, D) * 0.5,
    "w1": w(L, D, F), "w2": w(L, F, D) * 0.5,
    "g1": np.ones((L, D), np.float32), "b1": np.zeros((L, D), np.float32),
    "g2": np.ones((L, D), np.float32), "b2": np.zeros((L, D), np.float32),
    "gf": np.ones(D, np.float32), "bf": np.zeros(D, np.float32), "wout": w(D, V),
}

src = open("examples/gpt_text.mira").read()
t0 = time.perf_counter()
LOSS_SCALE = 1 if args.target == "cpu" else 1024          # fp16 targets need loss scaling
step = mira.compile_source(src, args.target, entry="train_step", dims={**dims, "loss_scale": LOSS_SCALE})
print(f"{len(text)} characters, vocabulary of {V}; compiled train_step for {args.target} "
      f"in {time.perf_counter() - t0:.1f}s")

names = list(params)
m = {k: np.zeros_like(v) for k, v in params.items()}
v2 = {k: np.zeros_like(v) for k, v in params.items()}
t0 = time.perf_counter()
for i in range(1, args.steps + 1):
    lr = args.lr * min(1.0, i / 100) * (0.1 + 0.9 * (1 - i / args.steps))      # warmup, then decay
    loss, *grads = step.run({"tok": batch(), **params})
    for k, g in zip(names, grads):         # Adam
        g = np.asarray(g, np.float32) / LOSS_SCALE
        m[k] = 0.9 * m[k] + 0.1 * g
        v2[k] = 0.99 * v2[k] + 0.01 * g * g
        params[k] -= lr * (m[k] / (1 - 0.9 ** i)) / (np.sqrt(v2[k] / (1 - 0.99 ** i)) + 1e-8)
    if i % 250 == 0 or i == 1:
        print(f"step {i:5d}  loss {float(loss):.3f}")
print(f"trained in {time.perf_counter() - t0:.0f}s\n")

P = 16
gen = mira.compile_source(src, args.target, entry="generate", dims={**dims, "P": P}, weights=params)
for prompt in ["twinkle, twinkle", "jack and jill we", "the cow jumped o"]:
    out = gen.run({"prompt": np.array([stoi[c] for c in prompt[:P]], np.int32)})
    print(repr("".join(chars[i] for i in out)))
