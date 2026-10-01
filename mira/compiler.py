"""The compiler driver: source -> tokens -> AST -> IR -> optimized IR -> partitions -> executables."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from . import ir, passes
from . import syntax as S
from .backends import get_target
from .backends.cpu import CPUTarget
from .elaborate import Elaborator, elaborate
from .errors import MiraError
from .parser import parse
from .partition import Placement, Segment, partition
from .types import TensorType


# ------------------------------------------------------------------ one compiled graph

@dataclass
class GraphRunner:
    """A graph split into device segments, each compiled for its device."""
    graph: ir.Graph
    segments: list[Segment]
    placement: Placement
    target: str
    last_timings: list[tuple[str, float]] = field(default_factory=list)

    def run_env(self, feeds: dict[ir.Value, np.ndarray]) -> dict[ir.Value, np.ndarray]:
        env = dict(feeds)
        for op in self.graph.ops:   # outputs that folded to constants
            if op.is_const and op.result in self.graph.outputs:
                env[op.result] = op.attrs["value"]
        self.last_timings = []
        for seg in self.segments:
            t0 = time.perf_counter()
            env.update(seg.executable.run({v: env[v] for v in seg.graph.inputs}))
            self.last_timings.append((seg.graph.name, time.perf_counter() - t0))
        return env

    def run_list(self, values: list[np.ndarray]) -> list[np.ndarray]:
        env = self.run_env(dict(zip(self.graph.inputs, values)))
        return [env[v] for v in self.graph.outputs]

    def children(self) -> list["GraphRunner"]:
        out = []
        for seg in self.segments:
            out.extend(getattr(seg.executable, "sub", {}).values())
        return out

    def annotations(self) -> dict[ir.Op, str]:
        ann: dict[ir.Op, str] = {}
        for seg in self.segments:
            ex = seg.executable
            for op in seg.graph.compute_ops():
                where = "host" if op.is_control else seg.device
                if hasattr(ex, "device_of"):
                    d = ex.device_of(op.result)
                    where += f"->{d}" if d else ""
                if op in self.placement.reason and not op.is_control:
                    where += f"  ({self.placement.reason[op]})"
                ann[op] = where
        for child in self.children():
            ann.update(child.annotations())
        return ann

    def summary_lines(self, indent: str = "  ") -> list[str]:
        lines = []
        for i, seg in enumerate(self.segments):
            n = len(seg.graph.compute_ops())
            ins = ", ".join(f"{v!r}:{v.type}" for v in seg.graph.inputs)
            lines.append(f"{indent}[{i}] {seg.device:<8} {n:3d} ops   inputs: {ins}")
            lines.append(f"{indent}    " + seg.executable.report().replace("\n", f"\n{indent}    "))
            for (op, name), child in getattr(seg.executable, "sub", {}).items():
                lines.append(f"{indent}    {op.kind} {name}: {child.graph.name}")
                lines.extend(child.summary_lines(indent + "      "))
        return lines


def compile_graph(graph: ir.Graph, tgt, cost_model: bool = True) -> GraphRunner:
    """Partition `graph` for `tgt` and compile every segment (recursing into control-flow bodies)."""
    cpu = CPUTarget()
    segments, placement = partition(graph, tgt, cost_model=cost_model)

    def compile_sub(g: ir.Graph) -> GraphRunner:
        return compile_graph(g, tgt, cost_model)

    for seg in segments:
        if seg.device == tgt.name and tgt.name != "cpu":
            seg.executable = tgt.compile(seg.graph)
        else:
            seg.executable = cpu.compile(seg.graph, compile_sub)
    return GraphRunner(graph, segments, placement, tgt.name)


# ------------------------------------------------------------------ a compiled program

@dataclass
class CompiledProgram:
    target: str
    runner: GraphRunner
    pass_log: list[tuple[str, str]] = field(default_factory=list)
    compile_seconds: float = 0.0

    @property
    def graph(self) -> ir.Graph:
        return self.runner.graph

    @property
    def segments(self) -> list[Segment]:
        return self.runner.segments

    @property
    def placement(self) -> Placement:
        return self.runner.placement

    @property
    def last_timings(self) -> list[tuple[str, float]]:
        return self.runner.last_timings

    @property
    def input_names(self) -> list[str]:
        return [v.name for v in self.graph.inputs]

    def input_types(self) -> dict[str, TensorType]:
        return {v.name: v.type for v in self.graph.inputs}

    def __call__(self, **inputs: np.ndarray):
        return self.run(inputs)

    def run(self, inputs: dict[str, np.ndarray]) -> Union[np.ndarray, tuple]:
        """Run with named inputs. Returns one array, or a tuple for multi-value programs."""
        missing = [v.name for v in self.graph.inputs if v.name not in inputs]
        if missing:
            raise MiraError(f"missing input(s): {', '.join(missing)}")
        extra = set(inputs) - {v.name for v in self.graph.inputs}
        if extra:
            raise MiraError(f"unknown input(s): {', '.join(sorted(extra))}")
        feeds = {}
        for v in self.graph.inputs:
            arr = np.asarray(inputs[v.name])
            if tuple(arr.shape) != v.type.shape:
                raise MiraError(f"input '{v.name}' should have shape {list(v.type.shape)}, got {list(arr.shape)}")
            feeds[v] = arr.astype(v.type.np_dtype)
        env = self.runner.run_env(feeds)
        outs = [env[v] for v in self.graph.outputs]
        return outs[0] if len(outs) == 1 else tuple(outs)

    # ----- reports

    def placement_report(self) -> str:
        return ir.format_graph(self.graph, self.runner.annotations())

    def summary(self) -> str:
        head = f"target {self.target}: {len(self.segments)} segment(s), compiled in {self.compile_seconds:.2f}s"
        return "\n".join([head] + self.runner.summary_lines())


# ------------------------------------------------------------------ dynamic shapes

class DynamicProgram:
    """A program whose entry function has shape symbols bound only at run time.

    NPUs need static shapes, so each new combination of input shapes is compiled
    on first use and cached (specialization). With `buckets`, a symbol is rounded
    up to the next bucket size and the inputs are zero-padded, so e.g. batch sizes
    1..128 share a handful of compiled programs. Padding is only correct when rows
    along that symbol are independent (true for batch dimensions); outputs are
    sliced back along every axis the return type declares with that symbol.
    """

    def __init__(self, source: str, target: str, fn: S.FnDecl, compile_kwargs: dict,
                 buckets: Optional[dict[str, list[int]]] = None):
        self.source, self.target, self.fn = source, target, fn
        self.kwargs = compile_kwargs
        self.buckets = {k: sorted(v) for k, v in (buckets or {}).items()}
        self.cache: dict[tuple, CompiledProgram] = {}

    @property
    def specializations(self) -> list[dict[str, int]]:
        return [dict(k) for k in self.cache]

    def bind(self, inputs: dict[str, np.ndarray]) -> dict[str, int]:
        el = Elaborator(S.Module(), "bind")
        dims = dict(self.kwargs.get("dims") or {})
        weights = self.kwargs.get("weights") or {}
        for p in self.fn.params:
            if p.type.is_int:
                continue
            arr = weights.get(p.name) if p.is_const else inputs.get(p.name)
            if arr is None:
                if p.is_const:
                    continue
                raise MiraError(f"missing input '{p.name}'")
            arr = np.asarray(arr)
            el.unify(p.type, TensorType(p.type.dtype, tuple(arr.shape)), dims, f"input '{p.name}'", p.loc)
        return dims

    def run(self, inputs: dict[str, np.ndarray]):
        actual = self.bind(inputs)
        dims = dict(actual)
        for sym, sizes in self.buckets.items():
            if sym in dims:
                dims[sym] = next((b for b in sizes if b >= dims[sym]), dims[sym])
        key = tuple(sorted(dims.items()))
        if key not in self.cache:
            kw = dict(self.kwargs, dims=dims)
            self.cache[key] = compile_source(self.source, self.target, **kw)
        prog = self.cache[key]

        padded = {}
        for p in self.fn.params:
            if p.is_const or p.type.is_int:
                continue
            arr = np.asarray(inputs[p.name])
            pads = [(0, dims[d] - actual[d]) if isinstance(d, str) and dims[d] != actual[d] else (0, 0)
                    for d in p.type.dims]
            padded[p.name] = np.pad(arr, pads) if any(hi for _, hi in pads) else arr
        out = prog.run(padded)

        outs = list(out) if isinstance(out, tuple) else [out]
        for i, (o, rt) in enumerate(zip(outs, self.fn.rets)):
            idx = tuple(slice(0, actual[d]) if isinstance(d, str) and d in actual else slice(None) for d in rt.dims)
            outs[i] = o[idx]
        return outs[0] if len(outs) == 1 else tuple(outs)

    def __call__(self, **inputs: np.ndarray):
        return self.run(inputs)


def _unbound_symbols(fn: S.FnDecl, dims: dict, weights: dict) -> set[str]:
    syms = set()
    for p in fn.params:
        if p.type.is_int:
            if p.name not in dims:
                syms.add(p.name)
            continue
        for d in p.type.dims:
            if isinstance(d, str) and d not in dims:
                if p.is_const and p.name not in weights:
                    raise MiraError(f"const parameter '{p.name}' has a runtime dimension '{d}' but no weight "
                                    f"was supplied to fix its shape", p.loc)
                syms.add(d)
    return syms


# ------------------------------------------------------------------ entry points

def compile_source(source: str, target: str = "cpu", *, filename: str = "<input>", entry: str = "main",
                   weights: Optional[dict[str, np.ndarray]] = None, dims: Optional[dict[str, int]] = None,
                   seed: Optional[int] = 0, optimize: bool = True, fuse: bool = True,
                   precision: Optional[str] = "auto", cost_model: bool = True, quantize: Optional[str] = None,
                   dynamic: bool = True, buckets: Optional[dict[str, list[int]]] = None,
                   **target_options) -> Union[CompiledProgram, DynamicProgram]:
    """Compile Mira source for `target` ("cpu", "npu-sim", or "coreml").

    precision="auto" lowers to fp16 for accelerator targets (what NPUs compute in)
    and leaves the program's own dtypes alone on the CPU. quantize="int8" stores
    large weights as int8 with per-channel scales.

    If the entry function has shape symbols that neither `dims` nor the weights
    pin down, a DynamicProgram is returned that compiles per input shape.
    """
    t0 = time.perf_counter()
    try:
        module = parse(source, filename)
        fn = module.functions.get(entry)
        if dynamic and fn is not None:
            unbound = _unbound_symbols(fn, dims or {}, weights or {})
            if unbound:
                kw = dict(filename=filename, entry=entry, weights=weights, dims=dims, seed=seed, optimize=optimize,
                          fuse=fuse, precision=precision, cost_model=cost_model, quantize=quantize, dynamic=False,
                          **target_options)
                return DynamicProgram(source, target, fn, kw, buckets)
        graph = elaborate(module, entry, weights, dims, seed)
    except MiraError as e:
        e.source = source
        raise
    return compile_ir(graph, target, optimize=optimize, fuse=fuse, precision=precision, cost_model=cost_model,
                      quantize=quantize, started=t0, **target_options)


def compile_ir(graph: ir.Graph, target: str = "cpu", *, optimize: bool = True, fuse: bool = True,
               precision: Optional[str] = "auto", cost_model: bool = True, quantize: Optional[str] = None,
               started: Optional[float] = None, **target_options) -> CompiledProgram:
    """The back half of the compiler, shared by every front end: optimize, quantize, partition, compile."""
    t0 = started if started is not None else time.perf_counter()
    if precision == "auto":
        precision = None if target == "cpu" else "f16"
    log: list[tuple[str, str]] = [("input", str(graph))]
    if optimize:
        passes.optimize(graph, fuse=fuse, precision=precision, log=log)
    elif precision == "f16":
        passes.to_f16(graph)
    if quantize:
        from . import quantize as Q
        Q.quantize_weights(graph, quantize, log=log)
    runner = compile_graph(graph, get_target(target, **target_options), cost_model)
    return CompiledProgram(target, runner, log, time.perf_counter() - t0)


class ShapeSpecialized:
    """Compiles a model once per distinct set of input shapes (for models with symbolic dimensions)."""

    def __init__(self, build, input_names: list[str]):
        self.build = build
        self.input_names = input_names
        self.cache: dict[tuple, CompiledProgram] = {}

    @property
    def specializations(self) -> list[dict[str, tuple[int, ...]]]:
        return [dict(k) for k in self.cache]

    def run(self, inputs: dict[str, np.ndarray]):
        key = tuple((n, tuple(np.shape(inputs[n]))) for n in self.input_names if n in inputs)
        if key not in self.cache:
            self.cache[key] = self.build(dict(key))
        return self.cache[key].run(inputs)

    def __call__(self, **inputs: np.ndarray):
        return self.run(inputs)


def compile_onnx(model, target: str = "cpu", *, input_shapes: Optional[dict[str, tuple[int, ...]]] = None,
                 **kw) -> Union[CompiledProgram, ShapeSpecialized]:
    """Import an ONNX model (path, bytes or ModelProto) and compile it for `target`.

    Inputs with symbolic dimensions (like a dynamic batch size) give a ShapeSpecialized
    program that compiles once per input shape seen, unless `input_shapes` pins them.
    """
    from .frontends.onnx_import import import_onnx, onnx_inputs
    t0 = time.perf_counter()
    shapes = dict(input_shapes or {})
    inputs = onnx_inputs(model)
    if any(None in dims and name not in shapes for name, dims, _ in inputs):
        return ShapeSpecialized(lambda s: compile_onnx(model, target, input_shapes={**shapes, **s}, **kw),
                                [name for name, _, _ in inputs])
    graph = import_onnx(model, shapes)
    return compile_ir(graph, target, started=t0, **kw)


def compile_file(path: str, target: str = "cpu", **kw) -> Union[CompiledProgram, DynamicProgram, ShapeSpecialized]:
    """Compile a .mira source file, or import and compile an .onnx model."""
    if path.endswith(".onnx"):
        for k in ("entry", "weights", "dims", "seed", "dynamic", "buckets", "filename"):
            kw.pop(k, None)
        return compile_onnx(path, target, **kw)
    with open(path) as f:
        return compile_source(f.read(), target, filename=path, **kw)


__all__ = ["compile_source", "compile_file", "compile_onnx", "compile_ir", "CompiledProgram", "DynamicProgram",
           "ShapeSpecialized", "GraphRunner"]
