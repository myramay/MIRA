"""Mira's intermediate representation (IR).

A `Graph` is a list of `Op`s in SSA form: every `Value` is defined exactly once
(by an op, or as a graph input) and ops appear in a valid execution order.

    graph main(%x: f16[8, 784]) -> (f16[8, 10]) {
      %1 = const() {value=<f16[784, 256]>}          : f16[784, 256]
      %2 = matmul(%x, %1)                           : f16[8, 256]
      %3 = relu(%2)                                 : f16[8, 256]
      ...
      return %9
    }
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import ops as O
from .errors import Loc
from .types import TensorType

_ids = itertools.count(1)


@dataclass(eq=False)
class Value:
    type: TensorType
    producer: Optional["Op"] = None
    name: Optional[str] = None            # graph inputs keep their source names
    id: int = field(default_factory=lambda: next(_ids))

    def __repr__(self) -> str:
        return f"%{self.name or self.id}"


@dataclass(eq=False)
class Op:
    kind: str
    inputs: list[Value]
    attrs: dict
    result: Value
    loc: Optional[Loc] = None

    @property
    def is_const(self) -> bool:
        return self.kind == "const"


@dataclass(eq=False)
class Graph:
    name: str
    inputs: list[Value]
    ops: list[Op]
    outputs: list[Value]

    # ----- construction -----

    def add(self, kind: str, inputs: list[Value], attrs: Optional[dict] = None, loc: Optional[Loc] = None) -> Value:
        """Append an op, inferring its result type (raises ops.OpTypeError)."""
        attrs = attrs or {}
        ty = O.infer(kind, [v.type for v in inputs], attrs)
        v = Value(ty)
        op = Op(kind, list(inputs), attrs, v, loc)
        v.producer = op
        self.ops.append(op)
        return v

    def const(self, array: np.ndarray, loc: Optional[Loc] = None) -> Value:
        return self.add("const", [], {"value": np.asarray(array)}, loc)

    # ----- queries -----

    def users(self) -> dict[Value, list[Op]]:
        uses: dict[Value, list[Op]] = {}
        for op in self.ops:
            for v in op.inputs:
                uses.setdefault(v, []).append(op)
        return uses

    def replace_uses(self, old: Value, new: Value) -> None:
        for op in self.ops:
            op.inputs = [new if v is old else v for v in op.inputs]
        self.outputs = [new if v is old else v for v in self.outputs]

    def compute_ops(self) -> list[Op]:
        return [op for op in self.ops if not op.is_const]

    # ----- checking -----

    def verify(self) -> None:
        """Check SSA/ordering invariants and that every type is still correct."""
        defined = set(self.inputs)
        for op in self.ops:
            for v in op.inputs:
                if v not in defined:
                    raise AssertionError(f"{op.kind} uses {v} before it is defined")
            expect = O.infer(op.kind, [v.type for v in op.inputs], op.attrs)
            if expect != op.result.type:
                raise AssertionError(f"{op.kind} result type {op.result.type} != inferred {expect}")
            if op.result.producer is not op:
                raise AssertionError(f"{op.result} has wrong producer")
            defined.add(op.result)
        for v in self.outputs:
            if v not in defined:
                raise AssertionError(f"output {v} is not defined")

    # ----- printing -----

    def __str__(self) -> str:
        return format_graph(self)


def format_attr(k: str, v) -> str:
    if isinstance(v, np.ndarray):
        if v.size <= 4:
            return f"{k}={np.array2string(v.astype(np.float32).ravel(), precision=4, separator=',')}"
        from .types import type_of_array
        return f"{k}=<{type_of_array(v)}>"
    if k == "epilogue":
        steps = [f"{fn}" if idx is None else f"{fn}(in{idx}{',swapped' if sw else ''})" for fn, idx, sw in v]
        return f"epilogue=[{' -> '.join(steps)}]"
    return f"{k}={v}"


def format_op(op: Op) -> str:
    args = ", ".join(map(repr, op.inputs))
    attrs = ", ".join(format_attr(k, v) for k, v in op.attrs.items())
    text = f"{op.result!r} = {op.kind}({args})" + (f" {{{attrs}}}" if attrs else "")
    return f"{text:<60} : {op.result.type}"


def format_graph(g: Graph, annotate: Optional[dict[Op, str]] = None) -> str:
    ins = ", ".join(f"{v!r}: {v.type}" for v in g.inputs)
    outs = ", ".join(str(v.type) for v in g.outputs)
    lines = [f"graph {g.name}({ins}) -> ({outs}) {{"]
    for op in g.ops:
        line = "  " + format_op(op)
        if annotate and op in annotate:
            line += f"   @{annotate[op]}"
        lines.append(line)
    lines.append("  return " + ", ".join(map(repr, g.outputs)))
    lines.append("}")
    return "\n".join(lines)
