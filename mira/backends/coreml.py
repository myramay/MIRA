"""Core ML backend: IR subgraph -> MIL program -> Apple Neural Engine.

Apple doesn't publish the Neural Engine's instruction set, so this is as low
as a public compiler can go: we emit MIL (Core ML's own tensor IR), Core ML
compiles it, and Core ML decides per op whether it runs on the Neural Engine,
GPU, or CPU. After compiling we ask Core ML for its compute plan, which says
where each op was actually placed, and map that back to our IR ops.
"""
from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import sys
import tempfile
import warnings
from typing import Optional

import numpy as np

from .. import ir

_UNARY = {"relu": "relu", "sigmoid": "sigmoid", "tanh": "tanh", "exp": "exp", "log": "log", "sqrt": "sqrt",
          "abs": "abs", "sign": "sign", "erf": "erf"}
MIL_DTYPE = {"f16": "fp16", "f32": "fp32", "i32": "int32"}
# ops lowered for i32 tensors (others with integer operands fall back to the CPU)
INT_OK = {"add", "sub", "mul", "maximum", "minimum", "neg", "greater", "greater_equal", "equal", "where", "cast",
          "reshape", "transpose", "slice", "concat", "broadcast", "reduce_sum", "reduce_max", "gather", "argmax",
          "argmin", "dynamic_slice", "dynamic_update_slice", "scatter_add", "pad"}
_COMPARE = {"greater": "greater", "greater_equal": "greater_equal", "equal": "equal"}
_BINARY = {"add": "add", "sub": "sub", "mul": "mul", "div": "real_div", "pow": "pow",
           "maximum": "maximum", "minimum": "minimum"}
_REDUCE = {"reduce_sum": "reduce_sum", "reduce_mean": "reduce_mean", "reduce_max": "reduce_max"}
SUPPORTED = set(_UNARY) | set(_BINARY) | set(_REDUCE) | set(_COMPARE) | {
    "neg", "gelu", "matmul", "conv2d", "maxpool2d", "softmax", "transpose", "reshape", "concat", "layernorm", "cast",
    "where", "broadcast", "slice", "pad", "dequantize", "gather", "argmax", "argmin", "scatter_add",
    "dynamic_slice", "dynamic_update_slice"}

COMPUTE_UNITS = {"ne": "CPU_AND_NE", "all": "ALL", "gpu": "CPU_AND_GPU", "cpu": "CPU_ONLY"}


_libc = ctypes.CDLL(None)


@contextlib.contextmanager
def capture_native_output(sink: list[str]):
    """Capture what Core ML's native code prints (fds 1 and 2), e.g. ANE compiler failures.

    The C library buffers stdout, so we fflush() it before restoring the descriptors;
    otherwise the text would only appear when the process exits.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        for fd in saved:
            os.dup2(tmp.fileno(), fd)
        try:
            yield
        finally:
            _libc.fflush(None)
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, orig in saved.items():
                os.dup2(orig, fd)
                os.close(orig)
            tmp.seek(0)
            text = tmp.read().decode(errors="replace").replace("E5RT encountered", "\nE5RT encountered")
            sink.extend(line for line in text.splitlines() if line.strip())


ANE_FAILURE = "ANECCompile() FAILED"
KEEP_INPUTS = 4      # see CoreMLExecutable.run


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
        mdt = {"f16": mtypes.fp16, "f32": mtypes.fp32, "i32": mtypes.int32}
        specs, self.in_names = {}, {}
        for i, v in enumerate(g.inputs):
            name = f"in{i}"
            self.in_names[v] = name
            specs[name] = mb.TensorSpec(shape=v.type.shape or (1,), dtype=mdt[v.type.dtype])
        self.out_names = {v: f"out{i}" for i, v in enumerate(g.outputs)}

        with Function(specs, opset_version=ct.target.macOS15) as func:
            env: dict[ir.Value, object] = {}
            for v in g.inputs:   # Core ML inputs have rank >= 1; scalars come in as [1] and are squeezed
                inp = func.inputs[self.in_names[v]]
                env[v] = mb.squeeze(x=inp) if v.type.rank == 0 else inp
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

        floats = [op.result.type.dtype for op in g.compute_ops() if op.result.type.is_float]
        precision = ct.precision.FLOAT16 if all(d == "f16" for d in floats) else ct.precision.FLOAT32
        self.native_log: list[str] = []
        with capture_native_output(self.native_log):
            self.model = ct.convert(prog, convert_to="mlprogram",
                                    compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[units]),
                                    minimum_deployment_target=ct.target.macOS15,
                                    compute_precision=precision)
            self.placement = self._compute_plan(ct)
        self.warmed_up = False
        self._recent_inputs: list[dict[str, np.ndarray]] = []

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
        mil_dt = MIL_DTYPE[op.result.type.dtype]
        if k in _COMPARE:     # MIL comparisons return bool; Mira's are 1.0 / 0.0
            return mb.cast(x=getattr(mb, _COMPARE[k])(x=x[0], y=x[1]), dtype=mil_dt, name=name)
        if k == "where":
            cond = mb.not_equal(x=x[0], y=op.inputs[0].type.np_dtype(0))
            return mb.select(cond=cond, a=x[1], b=x[2], name=name)
        if k == "broadcast":
            shape = tuple(a["shape"])
            src = op.inputs[0].type.shape
            if len(src) < len(shape):
                src = (1,) * (len(shape) - len(src)) + src
                x0 = mb.reshape(x=x[0], shape=list(src))
            else:
                x0 = x[0]
            return mb.tile(x=x0, reps=[t // s for s, t in zip(src, shape)], name=name)
        if k == "slice":
            return mb.slice_by_size(x=x[0], begin=list(a["begin"]), size=list(a["size"]), name=name)
        if k == "pad":
            flat = [n for pair in a["pads"] for n in pair]
            return mb.pad(x=x[0], pad=flat, mode="constant", constant_val=np_dt(0), name=name)
        if k == "dequantize":   # weights stored as int8, expanded on load (Core ML keeps them compressed)
            return mb.constexpr_affine_dequantize(quantized_data=a["q"], scale=a["scale"].astype(np_dt),
                                                  zero_point=np.int8(0), axis=a["axis"], name=name)
        if k in _REDUCE:
            return getattr(mb, _REDUCE[k])(x=x[0], axes=list(a["axes"]), keep_dims=a["keepdims"], name=name)
        if k == "softmax":
            return mb.softmax(x=x[0], axis=a["axis"], name=name)
        if k == "transpose":
            return mb.transpose(x=x[0], perm=list(a["perm"]), name=name)
        if k == "reshape":
            return mb.reshape(x=x[0], shape=np.array(a["shape"], dtype=np.int32), name=name)
        if k == "concat":
            return mb.concat(values=x, axis=a["axis"], name=name)
        if k == "cast":
            return mb.cast(x=x[0], dtype=MIL_DTYPE[a["dtype"]], name=name)
        if k == "gather":
            return mb.gather(x=x[0], indices=x[1], axis=a["axis"], name=name)
        if k in ("argmax", "argmin"):
            return getattr(mb, f"reduce_{k}")(x=x[0], axis=a["axis"], keep_dims=a["keepdims"], name=name)
        if k == "scatter_add":     # MIL scatter takes 1-D indices: flatten them and the matching update axes
            base, idx, upd = op.inputs
            ax = a["axis"] % base.type.rank
            flat_upd = base.type.shape[:ax] + (idx.type.numel,) + base.type.shape[ax + 1:]
            return mb.scatter(data=x[0], indices=mb.reshape(x=x[1], shape=[idx.type.numel]),
                              updates=mb.reshape(x=x[2], shape=list(flat_upd)), axis=ax, mode="add", name=name)
        if k in ("dynamic_slice", "dynamic_update_slice"):
            src = op.inputs[0].type.shape
            size = tuple(a["size"]) if k == "dynamic_slice" else op.inputs[1].type.shape
            start = x[1] if k == "dynamic_slice" else x[2]
            hi = np.array([d - n for d, n in zip(src, size)], dtype=np.int32)
            begin = mb.minimum(x=mb.maximum(x=start, y=np.int32(0)), y=hi)      # clamp, like the reference
            if k == "dynamic_slice":
                return mb.slice_by_size(x=x[0], begin=begin, size=list(size), name=name)
            end = mb.add(x=begin, y=np.array(size, dtype=np.int32))
            return mb.slice_update(x=x[0], update=x[1], begin=begin, end=end, name=name)
        if k == "layernorm":
            return mb.layer_norm(x=x[0], axes=[-1], gamma=x[1], beta=x[2], epsilon=np_dt(a["eps"]), name=name)
        if k == "maxpool2d":
            s = a["size"]
            return mb.max_pool(x=x[0], kernel_sizes=[s, s], strides=[a["stride"]] * 2, pad_type="valid", name=name)
        if k in ("matmul", "conv2d"):
            epi = a.get("epilogue", ())
            core_name = name if not epi else f"{name}_core"
            wp = op.inputs[1].producer
            w_const = wp is not None and wp.is_const
            if k == "matmul" and wp is not None and wp.kind == "dequantize" and op.inputs[0].type.rank <= 3:
                # int8 weight: keep it compressed in the model; `linear` wants it as [N, K]
                qt = np.ascontiguousarray(wp.attrs["q"].T)
                w = mb.constexpr_affine_dequantize(quantized_data=qt, scale=wp.attrs["scale"].astype(np_dt),
                                                   zero_point=np.int8(0), axis=0)
                acc = mb.linear(x=x[0], weight=w, name=core_name)
            elif k == "matmul" and w_const and op.inputs[1].type.rank == 2 and op.inputs[0].type.rank <= 3:
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
        # Core ML keeps the NumPy-backed input buffers of the last prediction and, after the model
        # sits idle, releases them on its own background thread without Python's GIL, which crashes
        # the interpreter (MLE5ExecutionStream resetAfterLingering -> array_dealloc). Holding our
        # own references to the last few inputs means Core ML never drops the final one; we free
        # them later, here, on a Python thread.
        self._recent_inputs.append(inputs)
        del self._recent_inputs[:-KEEP_INPUTS]
        if not self.warmed_up:      # the ANE program is built on first use; catch its failures too
            with capture_native_output(self.native_log):
                res = self.model.predict(inputs)
            self.warmed_up = True
        else:
            res = self.model.predict(inputs)
        return {v: np.asarray(res[name]).reshape(v.type.shape).astype(v.type.np_dtype)
                for v, name in self.out_names.items()}

    def report(self) -> str:
        devs = [self.device_of(op.result) for op in self.g.compute_ops()]
        counts = {d: devs.count(d) for d in sorted(set(devs), key=str)}
        summary = ", ".join(f"{d or 'fused/renamed'}: {n}" for d, n in counts.items())
        text = f"coreml ({COMPUTE_UNITS[self.units]}): {len(devs)} ops -> {summary}"
        if self.ane_rejected:
            text += ("\nwarning: Apple's ANE compiler rejected part of this model; Core ML runs that part on "
                     "the CPU/GPU instead (results are still correct)")
        return text

    @property
    def ane_rejected(self) -> bool:
        return any(ANE_FAILURE in line for line in self.native_log)


class CoreMLTarget:
    name = "coreml"

    def __init__(self, units: str = "ne"):
        self.units = units

    def check(self, op: ir.Op) -> Optional[str]:
        if op.kind not in SUPPORTED:
            return f"{op.kind}: no Core ML (MIL) equivalent"
        if any(v.type.rank > 5 for v in op.inputs) or op.result.type.rank > 5:
            return "Core ML supports tensors of rank <= 5"
        uses_int = any(not v.type.is_float for v in op.inputs) or not op.result.type.is_float
        if uses_int and op.kind not in INT_OK:
            return f"{op.kind} on i32: not lowered to Core ML"
        def is_const(v: ir.Value) -> bool:
            return v.producer is not None and v.producer.is_constant_like

        if op.kind == "conv2d" and not is_const(op.inputs[1]):
            return "conv2d: Core ML needs constant weights"
        if op.kind == "layernorm" and not all(is_const(v) for v in op.inputs[1:]):
            return "layernorm: Core ML needs constant gamma/beta"
        return None

    def compile(self, g: ir.Graph) -> CoreMLExecutable:
        return CoreMLExecutable(g, self.units)
