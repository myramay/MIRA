"""Core ML backend: IR subgraph -> MIL program -> Apple Neural Engine.

Apple doesn't publish the Neural Engine's instruction set, so this is as low
as a public compiler can go: we emit MIL (Core ML's own tensor IR), Core ML
compiles it, and Core ML decides per op whether it runs on the Neural Engine,
GPU, or CPU. After compiling we ask Core ML for its compute plan, which says
where each op was actually placed, and map that back to our IR ops.
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

import numpy as np

from .. import ir

_UNARY = {"relu": "relu", "sigmoid": "sigmoid", "tanh": "tanh", "exp": "exp", "log": "log", "sqrt": "sqrt",
          "abs": "abs"}
_BINARY = {"add": "add", "sub": "sub", "mul": "mul", "div": "real_div", "pow": "pow",
           "maximum": "maximum", "minimum": "minimum"}
_REDUCE = {"reduce_sum": "reduce_sum", "reduce_mean": "reduce_mean", "reduce_max": "reduce_max"}
SUPPORTED = set(_UNARY) | set(_BINARY) | set(_REDUCE) | {
    "neg", "gelu", "matmul", "conv2d", "maxpool2d", "softmax", "transpose", "reshape", "concat", "layernorm", "cast"}

COMPUTE_UNITS = {"ne": "CPU_AND_NE", "all": "ALL", "gpu": "CPU_AND_GPU", "cpu": "CPU_ONLY"}


def _quiet():
    logging.getLogger("coremltools").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="coremltools")
    os.environ.setdefault("TQDM_DISABLE", "1")


class CoreMLExecutable:
    def __init__(self, g: ir.Graph, units: str):
        _quiet()
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import Function, Program
        from coremltools.converters.mil.mil import types as mtypes

        self.g = g
        self.units = units
        mdt = {"f16": mtypes.fp16, "f32": mtypes.fp32}
        specs, self.in_names = {}, {}
        for i, v in enumerate(g.inputs):
            name = f"in{i}"
            self.in_names[v] = name
            specs[name] = mb.TensorSpec(shape=v.type.shape or (1,), dtype=mdt[v.type.dtype])
        self.out_names = {v: f"out{i}" for i, v in enumerate(g.outputs)}

        with Function(specs, opset_version=ct.target.macOS15) as func:
            env: dict[ir.Value, object] = {v: func.inputs[self.in_names[v]] for v in g.inputs}
            for op in g.ops:
                if op.is_const:
                    env[op.result] = op.attrs["value"]
                    continue
                name = self.out_names.get(op.result, f"op{op.result.id}")
                if op.result in self.out_names and op.result.type.rank == 0:
                    tmp = self._lower(mb, op, env, f"op{op.result.id}")
                    env[op.result] = mb.reshape(x=tmp, shape=[1], name=name)
                else:
                    env[op.result] = self._lower(mb, op, env, name)
            func.set_outputs([env[v] for v in g.outputs])
        prog = Program()
        prog.add_function("main", func)

        precision = ct.precision.FLOAT16 if all(op.result.type.dtype == "f16" for op in g.compute_ops()) \
            else ct.precision.FLOAT32
        self.model = ct.convert(prog, convert_to="mlprogram",
                                compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[units]),
                                minimum_deployment_target=ct.target.macOS15,
                                compute_precision=precision)
        self.placement = self._compute_plan(ct)

    # ----- IR op -> MIL ops

    def _lower(self, mb, op: ir.Op, env: dict, name: str):
        x = [env[v] for v in op.inputs]
        k, a = op.kind, op.attrs
        np_dt = op.result.type.np_dtype
        if k in _UNARY:
            return getattr(mb, _UNARY[k])(x=x[0], name=name)
        if k == "neg":
            return mb.mul(x=x[0], y=np_dt(-1), name=name)
        if k == "gelu":
            return mb.gelu(x=x[0], mode="TANH_APPROXIMATION", name=name)
        if k in _BINARY:
            return getattr(mb, _BINARY[k])(x=x[0], y=x[1], name=name)
        if k in _REDUCE:
            return getattr(mb, _REDUCE[k])(x=x[0], axes=list(a["axes"]), keep_dims=a["keepdims"], name=name)
        if k == "softmax":
            return mb.softmax(x=x[0], axis=a["axis"], name=name)
        if k == "transpose":
            return mb.transpose(x=x[0], perm=list(a["perm"]), name=name)
        if k == "reshape":
            return mb.reshape(x=x[0], shape=list(a["shape"]), name=name)
        if k == "concat":
            return mb.concat(values=x, axis=a["axis"], name=name)
        if k == "cast":
            return mb.cast(x=x[0], dtype={"f16": "fp16", "f32": "fp32"}[a["dtype"]], name=name)
        if k == "layernorm":
            return mb.layer_norm(x=x[0], axes=[-1], gamma=x[1], beta=x[2], epsilon=np_dt(a["eps"]), name=name)
        if k == "maxpool2d":
            s = a["size"]
            return mb.max_pool(x=x[0], kernel_sizes=[s, s], strides=[a["stride"]] * 2, pad_type="valid", name=name)
        if k in ("matmul", "conv2d"):
            epi = a.get("epilogue", ())
            core_name = name if not epi else f"{name}_core"
            w_const = op.inputs[1].producer is not None and op.inputs[1].producer.is_const
            if k == "matmul" and w_const and op.inputs[1].type.rank == 2 and op.inputs[0].type.rank <= 3:
                # Emit `linear` (weight stored as [N, K]) rather than `matmul` with a constant y.
                # Workaround: Core ML (coremltools 9.0, macOS 26) computes a constant-weight fp16
                # matmul followed directly by a transpose incorrectly; `linear` is unaffected.
                # Found by tests/test_fuzz.py; see tests/test_backends.py::test_coreml_matmul_then_transpose.
                acc = mb.linear(x=x[0], weight=np.ascontiguousarray(x[1].T), name=core_name)
            elif k == "matmul":
                acc = mb.matmul(x=x[0], y=x[1], name=core_name)
            else:
                (sh, sw), (ph, pw) = a["stride"], a["padding"]
                acc = mb.conv(x=x[0], weight=x[1], strides=[sh, sw], pad_type="custom", pad=[ph, ph, pw, pw],
                              name=core_name)
            # Emit the epilogue as ordinary MIL ops; Core ML's own compiler re-fuses them.
            for i, (fn, idx, swapped) in enumerate(epi):
                step_name = name if i == len(epi) - 1 else f"{name}_e{i}"
                if idx is None:
                    acc = (mb.gelu(x=acc, mode="TANH_APPROXIMATION", name=step_name) if fn == "gelu"
                           else getattr(mb, _UNARY[fn])(x=acc, name=step_name))
                else:
                    lhs, rhs = (x[idx], acc) if swapped else (acc, x[idx])
                    acc = getattr(mb, _BINARY[fn])(x=lhs, y=rhs, name=step_name)
            return acc
        raise NotImplementedError(k)

    # ----- where did Core ML put each op?

    def _compute_plan(self, ct) -> dict[str, str]:
        """Map our op names (op<id>) -> device Core ML chose ('ANE', 'GPU', 'CPU')."""
        try:
            from coremltools.models.compute_plan import MLComputePlan
            plan = MLComputePlan.load_from_path(path=self.model.get_compiled_model_path(),
                                                compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[self.units]))
            short = {"MLNeuralEngineComputeDevice": "ANE", "MLGPUComputeDevice": "GPU", "MLCPUComputeDevice": "CPU"}
            out: dict[str, str] = {}
            for mop in plan.model_structure.program.functions["main"].block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(mop)
                if usage is None:
                    continue
                dev = short.get(type(usage.preferred_compute_device).__name__, "?")
                for o in mop.outputs:
                    out[o.name] = dev
            return out
        except Exception as e:   # the compute plan API is best-effort (macOS 14.4+)
            return {"<error>": str(e)}

    def device_of(self, v: ir.Value) -> Optional[str]:
        name = self.out_names.get(v, f"op{v.id}")
        return self.placement.get(name)

    def run(self, feeds: dict[ir.Value, np.ndarray]) -> dict[ir.Value, np.ndarray]:
        inputs = {self.in_names[v]: np.ascontiguousarray(arr.reshape(v.type.shape or (1,)), dtype=v.type.np_dtype)
                  for v, arr in feeds.items()}
        res = self.model.predict(inputs)
        return {v: np.asarray(res[name]).reshape(v.type.shape).astype(v.type.np_dtype)
                for v, name in self.out_names.items()}

    def report(self) -> str:
        devs = [self.device_of(op.result) for op in self.g.compute_ops()]
        counts = {d: devs.count(d) for d in sorted(set(devs), key=str)}
        summary = ", ".join(f"{d or 'fused/renamed'}: {n}" for d, n in counts.items())
        return f"coreml ({COMPUTE_UNITS[self.units]}): {len(devs)} ops -> {summary}"


class CoreMLTarget:
    name = "coreml"

    def __init__(self, units: str = "ne"):
        self.units = units

    def check(self, op: ir.Op) -> Optional[str]:
        if op.kind not in SUPPORTED:
            return f"{op.kind}: no Core ML (MIL) equivalent"
        if any(v.type.rank > 5 for v in op.inputs) or op.result.type.rank > 5:
            return "Core ML supports tensors of rank <= 5"
        def is_const(v: ir.Value) -> bool:
            return v.producer is not None and v.producer.is_const

        if op.kind == "conv2d" and not is_const(op.inputs[1]):
            return "conv2d: Core ML needs constant weights"
        if op.kind == "layernorm" and not all(is_const(v) for v in op.inputs[1:]):
            return "layernorm: Core ML needs constant gamma/beta"
        return None

    def compile(self, g: ir.Graph) -> CoreMLExecutable:
        return CoreMLExecutable(g, self.units)
