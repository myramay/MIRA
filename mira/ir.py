"""MIRA's intermediate representation (IR).

A `Graph` is a list of `Op`s in SSA form: every `Value` is defined exactly once
(by an op, or as a graph input) and ops appear in a valid execution order.

    graph main(%x: f16[8, 784]) -> (f16[8, 10]) {
      %1 = const() {value=<f16[784, 256]>}          : f16[784, 256]
      %2 = matmul(%x, %1)                           : f16[8, 256]
      %3 = relu(%2)                                 : f16[8, 256]
      ...
      return %9
    }

Control flow is structured, as in MLIR's `scf` dialect: an `if` or `while` op
owns sub-graphs (its branches, or its condition and body). Values from the
enclosing graph are passed in explicitly as op inputs ("captures"), so every
graph is self-contained and can be compiled on its own.
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


CONTROL = {"if", "while"}


@dataclass(eq=False)
class Op:
    kind: str
    inputs: list[Value]
    attrs: dict
    results: list[Value]            # one for ordinary ops; control flow ops can have several
    loc: Optional[Loc] = None

    @property
    def result(self) -> Value:
        return self.results[0]

    @property
    def is_const(self) -> bool:
        return self.kind == "const"

    @property
    def is_constant_like(self) -> bool:
        """Constants, including compressed (int8) ones: no runtime inputs, embedded where used."""
        return self.kind in ("const", "dequantize")

    @property
    def is_control(self) -> bool:
        return self.kind in CONTROL

    def subgraphs(self) -> list[tuple[str, "Graph"]]:
        return [(k, v) for k, v in self.attrs.items() if isinstance(v, Graph)]


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
        op = Op(kind, list(inputs), attrs, [v], loc)
        v.producer = op
        self.ops.append(op)
        return v

    def add_control(self, kind: str, inputs: list[Value], attrs: dict, result_types: list[TensorType],
                    loc: Optional[Loc] = None) -> list[Value]:
        results = [Value(t) for t in result_types]
        op = Op(kind, list(inputs), attrs, results, loc)
        for v in results:
            v.producer = op
        check_control(op)
        self.ops.append(op)
        return results

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
        return [op for op in self.ops if not op.is_constant_like]

    # ----- checking -----

    def verify(self) -> None:
        """Check SSA/ordering invariants and that every type is still correct."""
        defined = set(self.inputs)
        for op in self.ops:
            for v in op.inputs:
                if v not in defined:
                    raise AssertionError(f"{op.kind} uses {v} before it is defined")
            if op.is_control:
                check_control(op)
                for _, sub in op.subgraphs():
                    sub.verify()
            else:
                expect = O.infer(op.kind, [v.type for v in op.inputs], op.attrs)
                if expect != op.result.type:
                    raise AssertionError(f"{op.kind} result type {op.result.type} != inferred {expect}")
            for r in op.results:
                if r.producer is not op:
                    raise AssertionError(f"{r} has wrong producer")
            defined.update(op.results)
        for v in self.outputs:
            if v not in defined:
                raise AssertionError(f"output {v} is not defined")

    # ----- printing -----

    def __str__(self) -> str:
        return format_graph(self)


def _types(vs: list[Value]) -> list[TensorType]:
    return [v.type for v in vs]


def check_control(op: Op) -> None:
    """Check that a control op's sub-graphs agree with its inputs and results."""
    ins, outs = _types(op.inputs), _types(op.results)
    if op.kind == "if":
        n = op.attrs["n_then"]
        then, orelse = op.attrs["then"], op.attrs["else"]
        if op.inputs[0].type.numel != 1:
            raise AssertionError("if: condition must have one element")
        if _types(then.inputs) != ins[1:1 + n] or _types(orelse.inputs) != ins[1 + n:]:
            raise AssertionError("if: branch inputs don't match captures")
        if _types(then.outputs) != outs or _types(orelse.outputs) != outs:
            raise AssertionError("if: branch outputs don't match results")
    elif op.kind == "while":
        ns, nc = op.attrs["n_state"], op.attrs["n_cond"]
        cond, body = op.attrs["cond"], op.attrs["body"]
        state = ins[:ns]
        if outs != state:
            raise AssertionError("while: results must match the loop state")
        if _types(cond.inputs) != state + ins[ns:ns + nc] or len(cond.outputs) != 1 \
                or cond.outputs[0].type.numel != 1:
            raise AssertionError("while: bad condition graph signature")
        if _types(body.inputs) != state + ins[ns + nc:] or _types(body.outputs) != state:
            raise AssertionError("while: bad body graph signature")


def format_attr(k: str, v) -> str:
    if isinstance(v, np.ndarray):
        if v.size <= 4:
            return f"{k}={np.array2string(v.astype(np.float32).ravel(), precision=4, separator=',')}"
        dt = {"float16": "f16", "float32": "f32", "int8": "i8"}.get(v.dtype.name, v.dtype.name)
        return f"{k}=<{dt}[{', '.join(map(str, v.shape))}]>"
    if k == "epilogue":
        steps = [f"{fn}" if idx is None else f"{fn}(in{idx}{',swapped' if sw else ''})" for fn, idx, sw in v]
        return f"epilogue=[{' -> '.join(steps)}]"
    return f"{k}={v}"


def format_op(op: Op) -> str:
    args = ", ".join(map(repr, op.inputs))
    attrs = ", ".join(format_attr(k, v) for k, v in op.attrs.items() if not isinstance(v, Graph))
    res = ", ".join(map(repr, op.results))
    text = f"{res} = {op.kind}({args})" + (f" {{{attrs}}}" if attrs else "")
    return f"{text:<60} : {', '.join(str(r.type) for r in op.results)}"


def format_graph(g: Graph, annotate: Optional[dict[Op, str]] = None, indent: str = "") -> str:
    ins = ", ".join(f"{v!r}: {v.type}" for v in g.inputs)
    outs = ", ".join(str(v.type) for v in g.outputs)
    lines = [f"{indent}graph {g.name}({ins}) -> ({outs}) {{"]
    for op in g.ops:
        line = indent + "  " + format_op(op)
        if annotate and op in annotate:
            line += f"   @{annotate[op]}"
        lines.append(line)
        for _, sub in op.subgraphs():
            lines.append(format_graph(sub, annotate, indent + "    "))
    lines.append(f"{indent}  return " + ", ".join(map(repr, g.outputs)))
    lines.append(f"{indent}}}")
    return "\n".join(lines)
