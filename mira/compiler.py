"""The compiler driver: source -> tokens -> AST -> IR -> optimized IR -> partitions -> executables."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import ir, passes
from .backends import get_target
from .backends.cpu import CPUTarget
from .elaborate import elaborate
from .errors import MiraError
from .parser import parse
from .partition import Placement, Segment, partition


@dataclass
class CompiledProgram:
    target: str
    graph: ir.Graph
    segments: list[Segment]
    placement: Placement
    pass_log: list[tuple[str, str]] = field(default_factory=list)
    compile_seconds: float = 0.0
    last_timings: list[tuple[str, float]] = field(default_factory=list)

    @property
    def input_names(self) -> list[str]:
        return [v.name for v in self.graph.inputs]

    def input_types(self) -> dict[str, "ir.TensorType"]:
        return {v.name: v.type for v in self.graph.inputs}

    def __call__(self, **inputs: np.ndarray) -> np.ndarray:
        return self.run(inputs)

    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        env: dict[ir.Value, np.ndarray] = {}
        missing = [v.name for v in self.graph.inputs if v.name not in inputs]
        if missing:
            raise MiraError(f"missing input(s): {', '.join(missing)}")
        extra = set(inputs) - {v.name for v in self.graph.inputs}
        if extra:
            raise MiraError(f"unknown input(s): {', '.join(sorted(extra))}")
        for v in self.graph.inputs:
            arr = np.asarray(inputs[v.name])
            if tuple(arr.shape) != v.type.shape:
                raise MiraError(f"input '{v.name}' should have shape {list(v.type.shape)}, got {list(arr.shape)}")
            env[v] = arr.astype(v.type.np_dtype)
        for op in self.graph.ops:   # outputs that folded to constants
            if op.is_const and op.result in self.graph.outputs:
                env[op.result] = op.attrs["value"]
        self.last_timings = []
        for seg in self.segments:
            t0 = time.perf_counter()
            env.update(seg.executable.run({v: env[v] for v in seg.graph.inputs}))
            self.last_timings.append((seg.graph.name, time.perf_counter() - t0))
        return env[self.graph.outputs[0]]

    # ----- reports

    def placement_report(self) -> str:
        ann = {}
        for seg in self.segments:
            ex = seg.executable
            for op in seg.graph.compute_ops():
                where = seg.device
                if hasattr(ex, "device_of"):
                    d = ex.device_of(op.result)
                    where += f"->{d}" if d else ""
                if op in self.placement.reason:
                    where += f"  ({self.placement.reason[op]})"
                ann[op] = where
        return ir.format_graph(self.graph, ann)

    def summary(self) -> str:
        lines = [f"target {self.target}: {len(self.segments)} segment(s), compiled in {self.compile_seconds:.2f}s"]
        for i, seg in enumerate(self.segments):
            n = len(seg.graph.compute_ops())
            ins = ", ".join(f"{v!r}:{v.type}" for v in seg.graph.inputs)
            lines.append(f"  [{i}] {seg.device:<8} {n:3d} ops   inputs: {ins}")
            lines.append("      " + seg.executable.report().replace("\n", "\n      "))
        return "\n".join(lines)


def compile_source(source: str, target: str = "cpu", *, filename: str = "<input>", entry: str = "main",
                   weights: Optional[dict[str, np.ndarray]] = None, dims: Optional[dict[str, int]] = None,
                   seed: Optional[int] = 0, optimize: bool = True, fuse: bool = True,
                   precision: Optional[str] = "auto", cost_model: bool = True,
                   **target_options) -> CompiledProgram:
    """Compile Mira source for `target` ("cpu", "npu-sim", or "coreml").

    precision="auto" lowers to fp16 for accelerator targets (what NPUs compute in)
    and leaves the program's own dtypes alone on the CPU.
    """
    t0 = time.perf_counter()
    try:
        module = parse(source, filename)
        graph = elaborate(module, entry, weights, dims, seed)
    except MiraError as e:
        e.source = source
        raise
    if precision == "auto":
        precision = None if target == "cpu" else "f16"
    log: list[tuple[str, str]] = [("elaborate", str(graph))]
    if optimize:
        passes.optimize(graph, fuse=fuse, precision=precision, log=log)
    elif precision == "f16":
        passes.to_f16(graph)
    tgt = get_target(target, **target_options)
    segments, placement = partition(graph, tgt, cost_model=cost_model)
    cpu = CPUTarget()
    for seg in segments:
        seg.executable = (tgt if seg.device == tgt.name else cpu).compile(seg.graph)
    return CompiledProgram(target, graph, segments, placement, log, time.perf_counter() - t0)


def compile_file(path: str, target: str = "cpu", **kw) -> CompiledProgram:
    with open(path) as f:
        return compile_source(f.read(), target, filename=path, **kw)
