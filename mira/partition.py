"""Graph partitioning: split the graph between the accelerator and the CPU.

This is what lets MIRA run *any* program on an NPU target: every op the
accelerator supports goes there, everything else falls back to the CPU, and
data moves between them at segment boundaries.

1. Placement: ask the target whether it can run each op (with a reason if not).
2. Scheduling: list-schedule the ops, preferring to stay on the current device,
   so independent ops are grouped and we get as few device switches as possible.
3. Cost model: an accelerator segment with no heavy compute (only elementwise
   and reshapes) isn't worth the transfer overhead, so it's moved to the CPU and
   we repeat until nothing changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import ir
from .backends.base import HEAVY_KINDS


@dataclass
class Segment:
    device: str
    graph: ir.Graph
    executable: object = None


@dataclass
class Placement:
    device: dict[ir.Op, str] = field(default_factory=dict)
    reason: dict[ir.Op, str] = field(default_factory=dict)


def is_heavy(op: ir.Op) -> bool:
    """Worth an accelerator trip: big compute, or a loop/branch whose body has some."""
    return op.kind in HEAVY_KINDS or any(is_heavy(inner) for _, sub in op.subgraphs() for inner in sub.ops)


def _schedule(ops: list[ir.Op], device: dict[ir.Op, str]) -> list[tuple[str, list[ir.Op]]]:
    order = {op: i for i, op in enumerate(ops)}
    producer = {r: op for op in ops for r in op.results}
    deps = {op: {producer[v] for v in op.inputs if v in producer} for op in ops}
    users: dict[ir.Op, list[ir.Op]] = {op: [] for op in ops}
    for op, ds in deps.items():
        for d in ds:
            users[d].append(op)
    remaining = {op: len(ds) for op, ds in deps.items()}
    ready = sorted((op for op in ops if remaining[op] == 0), key=order.get)
    groups: list[tuple[str, list[ir.Op]]] = []
    current: Optional[str] = None
    while ready:
        same = [op for op in ready if device[op] == current]
        if not same:
            current = device[ready[0]]
            groups.append((current, []))
            same = [op for op in ready if device[op] == current]
        op = same[0]
        ready.remove(op)
        groups[-1][1].append(op)
        for u in users[op]:
            remaining[u] -= 1
            if remaining[u] == 0:
                ready.append(u)
        ready.sort(key=order.get)
    return groups


def partition(g: ir.Graph, target, cost_model: bool = True) -> tuple[list[Segment], Placement]:
    compute = g.compute_ops()
    placement = Placement()
    for op in compute:
        if op.is_control and not getattr(target, "supports_control_flow", False):
            reason = "control flow runs on the host; its sub-graphs are compiled for the target separately"
        else:
            reason = target.check(op) if target.name != "cpu" else None
        placement.device[op] = "cpu" if reason else target.name
        if reason and target.name != "cpu":
            placement.reason[op] = reason

    while True:
        groups = _schedule(compute, placement.device)
        demoted = False
        if cost_model and target.name != "cpu":
            for dev, ops in groups:
                if dev != "cpu" and not any(is_heavy(op) for op in ops):
                    for op in ops:
                        placement.device[op] = "cpu"
                        placement.reason[op] = "kept on CPU: segment has no heavy compute, transfers would dominate"
                    demoted = True
        if not demoted:
            break

    const_of = {op.result: op for op in g.ops if op.is_constant_like}
    produced_in: dict[ir.Value, int] = {}
    for i, (_, ops) in enumerate(groups):
        for op in ops:
            for r in op.results:
                produced_in[r] = i

    segments: list[Segment] = []
    for i, (dev, ops) in enumerate(groups):
        consts: list[ir.Op] = []
        inputs: list[ir.Value] = []
        for op in ops:
            for v in op.inputs:
                if v in const_of:
                    if const_of[v] not in consts:
                        consts.append(const_of[v])
                elif produced_in.get(v) != i and v not in inputs:
                    inputs.append(v)
        needed_later = set(g.outputs)
        for j, (_, later) in enumerate(groups):
            if j != i:
                for op in later:
                    needed_later.update(op.inputs)
        outputs = [r for op in ops for r in op.results if r in needed_later]
        segments.append(Segment(dev, ir.Graph(f"{g.name}.seg{i}.{dev}", inputs, consts + ops, outputs)))
    return segments, placement
