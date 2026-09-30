"""Graph-level optimization passes.

Each pass takes a Graph, mutates it, and returns True if anything changed.
`optimize` runs them to a fixed point and verifies the graph after every pass,
so a buggy pass is caught at the pass that broke things, not three passes later.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Optional

import numpy as np

from . import ir
from . import ops as O

FOLD_LIMIT = 1 << 22   # don't bake gigantic folded constants into the binary


def _const_value(v: ir.Value) -> Optional[np.ndarray]:
    return v.producer.attrs["value"] if v.producer is not None and v.producer.is_const else None


# ------------------------------------------------------------------ constant folding

def const_fold(g: ir.Graph) -> bool:
    """Evaluate ops whose inputs are all constants, at compile time."""
    changed = False
    for op in g.ops:
        if op.is_const or not op.inputs or op.result.type.numel > FOLD_LIMIT:
            continue
        vals = [_const_value(v) for v in op.inputs]
        if any(v is None for v in vals):
            continue
        out = O.evaluate(op.kind, vals, op.attrs)
        op.kind, op.inputs, op.attrs = "const", [], {"value": np.asarray(out, dtype=op.result.type.np_dtype)}
        changed = True
    return changed


# ------------------------------------------------------------------ algebraic simplification

def _is_splat(v: ir.Value, x: float) -> bool:
    c = _const_value(v)
    return c is not None and bool(np.all(c == x))


def simplify(g: ir.Graph) -> bool:
    """Peephole rewrites: x+0, x*1, neg(neg x), transpose∘transpose, reshape∘reshape, ..."""
    changed = False
    for op in list(g.ops):
        r = op.result
        replacement: Optional[ir.Value] = None
        ins = op.inputs

        if op.kind in ("add", "sub") and _is_splat(ins[1], 0) and ins[0].type == r.type:
            replacement = ins[0]
        elif op.kind == "add" and _is_splat(ins[0], 0) and ins[1].type == r.type:
            replacement = ins[1]
        elif op.kind in ("mul", "div", "pow") and _is_splat(ins[1], 1) and ins[0].type == r.type:
            replacement = ins[0]
        elif op.kind == "mul" and _is_splat(ins[0], 1) and ins[1].type == r.type:
            replacement = ins[1]
        elif op.kind == "neg" and ins[0].producer is not None and ins[0].producer.kind == "neg":
            replacement = ins[0].producer.inputs[0]
        elif op.kind == "transpose":
            perm = tuple(op.attrs["perm"])
            inner = ins[0].producer
            if perm == tuple(range(len(perm))):
                replacement = ins[0]
            elif inner is not None and inner.kind == "transpose":
                p1 = inner.attrs["perm"]
                combined = tuple(p1[p] for p in perm)
                if combined == tuple(range(len(combined))):
                    replacement = inner.inputs[0]
                else:
                    op.inputs, op.attrs = [inner.inputs[0]], {"perm": combined}
                    changed = True
        elif op.kind == "reshape":
            inner = ins[0].producer
            if tuple(op.attrs["shape"]) == ins[0].type.shape:
                replacement = ins[0]
            elif inner is not None and inner.kind == "reshape":
                if inner.inputs[0].type.shape == r.type.shape:
                    replacement = inner.inputs[0]
                else:
                    op.inputs = [inner.inputs[0]]
                    changed = True
        elif op.kind == "cast":
            inner = ins[0].producer
            if ins[0].type.dtype == op.attrs["dtype"]:
                replacement = ins[0]
            elif (inner is not None and inner.kind == "cast" and inner.inputs[0].type == r.type
                  and inner.inputs[0].type.dtype == "f16"):
                replacement = inner.inputs[0]   # f16 -> f32 -> f16 is exact

        if replacement is not None and replacement is not r:
            g.replace_uses(r, replacement)
            changed = True
    return changed


# ------------------------------------------------------------------ CSE and DCE

def _key(op: ir.Op):
    if op.is_const:
        a = op.attrs["value"]
        return ("const", a.dtype.str, a.shape, hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest())
    return (op.kind, tuple(v.id for v in op.inputs), repr(sorted(op.attrs.items())))


def cse(g: ir.Graph) -> bool:
    """Common subexpression elimination: identical ops on identical inputs are computed once."""
    seen: dict[tuple, ir.Op] = {}
    changed = False
    for op in g.ops:
        k = _key(op)
        prev = seen.get(k)
        if prev is not None and (not op.is_const or np.array_equal(prev.attrs["value"], op.attrs["value"])):
            g.replace_uses(op.result, prev.result)
            changed = True
        else:
            seen[k] = op
    return changed


def dce(g: ir.Graph) -> bool:
    """Dead code elimination: drop ops whose results are never used."""
    live = set(g.outputs)
    kept = []
    for op in reversed(g.ops):
        if op.result in live:
            kept.append(op)
            live.update(op.inputs)
    kept.reverse()
    changed = len(kept) != len(g.ops)
    g.ops = kept
    return changed


# ------------------------------------------------------------------ operator fusion

FUSABLE_UNARY = {"relu", "gelu", "sigmoid", "tanh"}
FUSABLE_BINARY = {"add", "sub", "mul", "div", "maximum", "minimum"}


def fuse_epilogues(g: ir.Graph) -> bool:
    """Fold elementwise ops that consume a matmul/conv result into that op's epilogue.

        %2 = matmul(%x, %w)        ==>   %4 = matmul(%x, %w, %b) {epilogue=[add(%2) -> relu]}
        %3 = add(%2, %b)
        %4 = relu(%3)

    On an NPU this means the intermediate result never leaves on-chip memory:
    the bias add and activation are applied to the tile while it sits in the
    accumulator, instead of writing it to DRAM and reading it back.
    """
    any_change = False
    while True:
        users = g.users()
        position = {op: i for i, op in enumerate(g.ops)}
        fused = False
        for op in g.ops:
            if op.kind not in ("matmul", "conv2d") or op.result in g.outputs:
                continue
            uses = users.get(op.result, [])
            if len(uses) != 1:
                continue
            u = uses[0]
            inputs = list(op.inputs)
            if u.kind in FUSABLE_UNARY:
                step = (u.kind, None, False)
            elif u.kind in FUSABLE_BINARY and u.result.type == op.result.type:
                lhs_is_acc = u.inputs[0] is op.result
                other = u.inputs[1] if lhs_is_acc else u.inputs[0]
                if other is op.result:
                    continue
                step = (u.kind, len(inputs), not lhs_is_acc)
                inputs.append(other)
            else:
                continue
            attrs = dict(op.attrs)
            attrs["epilogue"] = tuple(op.attrs.get("epilogue", ())) + (step,)
            # The fused op takes u's place in the schedule: everything it reads
            # (op's inputs and u's other operand) is defined before u.
            new = ir.Op(op.kind, inputs, attrs, u.result, op.loc)
            u.result.producer = new
            g.ops[position[u]] = new
            g.ops.remove(op)
            fused = any_change = True
            break
        if not fused:
            return any_change


# ------------------------------------------------------------------ precision lowering

def to_f16(g: ir.Graph) -> bool:
    """Run the whole graph in fp16 (what NPUs are built for), keeping the f32 interface.

    Casts are inserted at the graph inputs and outputs; internal casts become no-ops.
    """
    if all(op.result.type.dtype == "f16" for op in g.ops) and all(v.type.dtype == "f16" for v in g.inputs):
        return False
    old_ops = g.ops
    g.ops = []
    remap: dict[ir.Value, ir.Value] = {}
    for v in g.inputs:
        remap[v] = g.add("cast", [v], {"dtype": "f16"}) if v.type.dtype == "f32" else v
    for op in old_ops:
        if op.is_const:
            remap[op.result] = g.const(op.attrs["value"].astype(np.float16), op.loc)
        elif op.kind == "cast":
            remap[op.result] = remap[op.inputs[0]]
        else:
            remap[op.result] = g.add(op.kind, [remap[v] for v in op.inputs], dict(op.attrs), op.loc)
    outs = []
    for v in g.outputs:
        nv = remap[v]
        outs.append(g.add("cast", [nv], {"dtype": "f32"}) if v.type.dtype == "f32" and nv.type.dtype != "f32" else nv)
    g.outputs = outs
    return True


# ------------------------------------------------------------------ pass manager

PIPELINE: list[tuple[str, Callable[[ir.Graph], bool]]] = [
    ("const-fold", const_fold),
    ("simplify", simplify),
    ("cse", cse),
    ("dce", dce),
]


def optimize(g: ir.Graph, fuse: bool = True, precision: Optional[str] = None,
             log: Optional[list[tuple[str, str]]] = None) -> ir.Graph:
    def run(name: str, fn) -> bool:
        changed = fn(g)
        g.verify()
        if changed and log is not None:
            log.append((name, str(g)))
        return changed

    if precision == "f16":
        run("to-f16", to_f16)
    for _ in range(10):   # iterate to a fixed point
        if not any([run(n, f) for n, f in PIPELINE]):
            break
    if fuse:
        run("fuse-epilogues", fuse_epilogues)
        run("dce", dce)
    return g
