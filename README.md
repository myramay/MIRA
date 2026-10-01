# Mira

[![tests](https://github.com/myramay/MIRA/actions/workflows/tests.yml/badge.svg)](https://github.com/myramay/MIRA/actions/workflows/tests.yml)

A small tensor language and compiler that runs neural-network programs, both inference and training, on four targets:

| target    | what it is                                                                              |
|-----------|-----------------------------------------------------------------------------------------|
| `cpu`     | NumPy interpreter of the IR, used as the correctness reference ("golden model")          |
| `npu-sim` | MNPU-1, a simulated NPU: banked SRAM, strided/transposing/int8 DMA, a 32×32 systolic array, a cycle model |
| `coreml`  | Apple Neural Engine via Core ML (MIL), with per-op placement reporting                    |
| `mlir`    | Standard MLIR (linalg/tensor/scf) compiled by IREE to native CPU code (or the Mac GPU via Metal) |

Ops a target can't run fall back to the CPU automatically.

```mira
fn dense(x: f32[B, N], w: f32[N, M], b: f32[M]) -> f32[B, M] {
  return x @ w + b
}

fn main(x: f32[256, 64], labels: f32[256, 10], w: f32[64, 10], b: f32[10])
    -> (f32[], f32[64, 10], f32[10]) {
  let z = dense(x, w, b)
  let loss = -sum(labels * (z - log(sum(exp(z), axis=1, keepdims=true)))) / 256
  let gw, gb = grad(loss, [w, b])
  return loss, w - 0.5 * gw, b - 0.5 * gb       # one SGD step, compiled for the NPU
}
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[coreml,mlir,onnx,dev]'
uv pip install --python .venv/bin/python -e '.[torch]'      # optional: to run the ONNX tests against PyTorch
source .venv/bin/activate
```

The `mlir` target's Metal GPU backend also needs Apple's Metal toolchain: `xcodebuild -downloadComponent MetalToolchain`.

## Use

```bash
mira run examples/mlp.mira -t coreml --check           # run on the Neural Engine, compare with CPU
mira run examples/cnn.mira -t sim --check --quantize int8
mira run examples/attention.mira -t mlir --check
mira emit examples/mlp.mira -s tokens|ast|ir|opt|passes|partition|mlir
mira emit examples/mlp.mira -t sim -s asm              # MNPU-1 assembly
mira emit examples/mlp.mira -t sim -s timeline         # engine timeline + utilization
mira bench examples/bench/big_mlp.mira --targets cpu,mlir,coreml:cpu,coreml:ne
mira bench examples/bench/big_mlp.mira --targets npu-sim --sim-fast --sim mxu_dim=64
python examples/train.py --target coreml               # train an MLP on the Neural Engine
python examples/gpt_train.py --target coreml           # train a tiny GPT, then generate with a KV cache
mira run model.onnx -t coreml --check                  # run an ONNX model (e.g. exported from PyTorch)
mira run model.onnx -t sim --shape x=4,3,224,224       # pin symbolic input dimensions
python -m pytest                                       # ~370 tests: fuzzing, gradient checks, PyTorch parity
```

From Python:

```python
import mira, numpy as np
prog = mira.compile_file("examples/mlp.mira", "coreml", weights={...}, quantize="int8")
y = prog.run({"x": np.random.randn(64, 784).astype(np.float32)})
print(prog.placement_report())

# models exported from PyTorch / TensorFlow via ONNX
resnet = mira.compile_onnx("model.onnx", "coreml")          # symbolic batch dims compile per shape

# shapes known only at run time: compiled per shape, or per bucket with padding
dyn = mira.compile_source("fn main(x: f32[B, 16], const w: f32[16, 8]) -> f32[B, 8] {\n  return x @ w\n}\n",
                          "coreml", buckets={"B": [8, 32, 128]})
dyn.run({"x": np.ones((5, 16), np.float32)})     # runs the B=8 specialization, returns 5 rows
```

## Language

- Types: `f16[...]`, `f32[...]`, `i32[...]`. Dimensions are integers, shape symbols (`B`), or expressions (`H / 2`).
- Indexing like NumPy: `x[i]`, `x[:, 1:3]`, `x[-1]`, `emb[tokens]` (an `i32` tensor index is a gather), and
  indexed assignment `buf[t] = v` (a functional update, also at a runtime position `t`).
- Functions are generic over shape symbols and specialized per call. `int` parameters are compile-time integers.
  Functions can return several values: `-> (f32[], f32[4, 4])`, and `let a, b = f(x)` unpacks them.
- Statements: `let`, reassignment, `for i in 0..N` (unrolled), `if`/`else if`/`else`, `while`, `return a, b`.
  `if` and `while` on compile-time values are resolved by the compiler. On one-element tensors they become
  real control flow. The CPU drives it on `npu-sim`/`coreml`; on `mlir` it compiles onto the device as `scf.if`/`scf.while`.
- Operators: `+ - * / ** @` (NumPy broadcasting), comparisons `< <= > >= == !=` (give 1/0 masks on tensors).
  A line ending in a comma or an operator continues on the next line.
- Builtins: `relu gelu sigmoid tanh exp log sqrt abs sign erf maximum minimum matmul softmax sum mean max
  argmax argmin transpose reshape flatten broadcast slice pad concat where gather scatter_add dynamic_slice
  dynamic_update_slice conv2d maxpool2d layernorm cast size sort cumprod zeros ones full arange`
  and `grad(y, x)` / `grad(y, [x1, x2])` (reverse-mode autodiff, including higher-order).

## What each limitation became

| limitation | now |
|---|---|
| inference only | `grad` differentiates any program at compile time. The backward pass is ordinary IR, so training runs on every target (`examples/train.py`) |
| no data-dependent control flow | `if`/`while` on tensors: structured ops with sub-graphs, host-driven or compiled onto the device (MLIR) |
| no runtime shapes | unbound shape symbols give a `DynamicProgram` that specializes per shape, with optional bucketing and padding |
| no quantization | `quantize="int8"`: per-channel weight quantization. Core ML keeps weights compressed (1.8× faster on the ANE for big models); MNPU-1's DMA dequantizes on the fly |
| simulator gaps | MNPU-1 now runs conv (implicit GEMM), pooling, any transpose, reductions, `where`, and int8 weights. It has a fast timing-only mode and configurable hardware (`--sim key=value`) |
| not MLIR | the `mlir` target emits upstream MLIR and compiles it with IREE. `mira emit -s mlir` prints it |
| Core ML placement is opaque | still decided by Core ML, but reported per op, and a warning is shown when Apple's ANE compiler rejects part of a model |
| can't load existing models | `mira.compile_onnx` / `mira run model.onnx`: ONNX import with import-time folding of shape arithmetic; PyTorch MLPs, CNNs and transformers match PyTorch on every target |
| no integers or indexing | `i32` tensors, NumPy-style indexing, `gather`/`scatter_add`/`argmax`, all differentiable where it makes sense |
| no growing sequences | fixed-size buffers written at runtime positions (`dynamic_update_slice`, `buf[t] = v`). `examples/gpt.mira` trains a tiny GPT and generates with a KV cache in a `while` loop |

Still true: the Neural Engine can only be reached through Core ML, which makes the final placement decisions, and MNPU-1's
timing is a model of hardware that doesn't exist.

## Layout

```
mira/
  lexer.py  parser.py  syntax.py     front end: text -> tokens -> AST
  elaborate.py                       type/shape checking, specialization, inlining, control flow -> IR
  autodiff.py                        reverse-mode differentiation (one VJP rule per op)
  frontends/onnx_import.py           ONNX -> Mira IR (NumPy evaluation of everything known at import time)
  ir.py  types.py  ops.py            the IR (with structured if/while), every op's shape rule and reference semantics
  passes.py  quantize.py             const-fold, simplify, CSE, DCE, epilogue fusion, fp16 lowering, int8 weights
  partition.py                       device placement, scheduling, cost model, segments
  compiler.py  cli.py                driver (incl. dynamic shapes) and command line
  backends/cpu.py                    golden model, fallback, host for control flow
  backends/coreml.py                 MIL emission, Neural Engine placement report
  backends/mlir.py                   MLIR emission (linalg/tensor/scf) + IREE compile & run
  backends/npusim/                   MNPU-1: codegen (tiling, memory planning, DMA) and machine
examples/                            MLP, CNN, transformer block, residual loop, fallback, training, tiny GPT
tests/                               front end, passes, autodiff, control flow, indexing, ONNX, GPT, backends,
                                     simulator, fuzzing
```
