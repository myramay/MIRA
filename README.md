# Mira

A small tensor language and compiler that runs neural-network programs on three targets:

| target    | what it is                                                                 |
|-----------|----------------------------------------------------------------------------|
| `cpu`     | NumPy interpreter of the IR, used as the correctness reference ("golden model") |
| `npu-sim` | MNPU-1, a simulated NPU with SRAM, DMA, a 32×32 systolic array, and a cycle model |
| `coreml`  | Apple Neural Engine via Core ML (MIL), with per-op placement reporting       |

Ops a target can't run fall back to the CPU automatically.

```mira
fn dense(x: f32[B, N], w: f32[N, M], b: f32[M]) -> f32[B, M] {
  return x @ w + b
}

fn main(x: f32[64, 784], const w1: f32[784, 512], const b1: f32[512],
        const w2: f32[512, 10], const b2: f32[10]) -> f32[64, 10] {
  return softmax(dense(relu(dense(x, w1, b1)), w2, b2), axis=1)
}
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[coreml,dev]'
source .venv/bin/activate
```

## Use

```bash
mira run examples/mlp.mira -t coreml --check      # run on the Neural Engine, compare with CPU
mira run examples/mlp.mira -t sim --check         # run on the simulated NPU
mira emit examples/mlp.mira -s tokens|ast|ir|opt|passes|partition   # inspect any stage
mira emit examples/mlp.mira -t sim -s asm         # MNPU-1 assembly
mira emit examples/mlp.mira -t sim -s timeline    # engine timeline + utilization
mira bench examples/bench/big_mlp.mira --targets cpu,coreml:cpu,coreml:gpu,coreml:ne
python -m pytest                                  # 196 tests, including fuzzing
```

Weights: `const` parameters are baked into the compiled program. Pass `--weights w.npz` or omit it for
seeded random weights. Inputs come from `--inputs x.npz`, or are random if omitted.

From Python:

```python
import mira, numpy as np
prog = mira.compile_file("examples/mlp.mira", "coreml", weights={...})
y = prog.run({"x": np.random.randn(64, 784).astype(np.float32)})
print(prog.placement_report())
```

## Language

- Types: `f16[...]`, `f32[...]`. Dimensions are integers, shape symbols (`B`), or expressions (`H / 2`).
- Functions are generic over shape symbols and specialized per call. `int` parameters are compile-time integers.
- Statements: `let x = ...`, `x = ...` (same type), `for i in 0..N { ... }` (unrolled), `return ...`.
- Operators: `+ - * / ** @`, with NumPy broadcasting.
- Builtins: `relu gelu sigmoid tanh exp log sqrt abs maximum minimum matmul softmax sum mean max
  transpose reshape flatten conv2d maxpool2d layernorm concat cast size sort cumprod`.

## Layout

```
mira/
  lexer.py  parser.py  syntax.py     front end: text -> tokens -> AST
  elaborate.py                       type/shape checking, specialization, inlining, unrolling -> IR
  ir.py  types.py  ops.py            the IR, and every op's shape rule and reference semantics
  passes.py                          const-fold, simplify, CSE, DCE, epilogue fusion, fp16 lowering
  partition.py                       device placement, scheduling, cost model, segments
  compiler.py  cli.py                driver and command line
  backends/cpu.py                    golden model / fallback
  backends/coreml.py                 MIL emission, Neural Engine placement report
  backends/npusim/                   MNPU-1: codegen (tiling, memory planning, DMA) and machine
tests/                               front end, passes, backends, simulator, fuzzing
```
