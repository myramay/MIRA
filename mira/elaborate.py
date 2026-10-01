"""Elaboration: typed AST walk that produces IR.

This single pass does semantic analysis *and* IR generation:

  * Name resolution with nested scopes (`let` shadows, `x = ...` reassigns).
  * Shape checking. Functions may be generic over shape symbols
    (`fn dense(x: f16[B, N], ...)`); each call unifies the symbols against the
    concrete argument shapes and elaborates the body with those bindings.
    This is *monomorphization*, like C++ templates: every shape the NPU sees
    is static, which is what NPU compilers need.
  * Inlining: user function calls disappear; the graph contains only ops.
  * Compile-time evaluation: numbers, shape symbols and loop variables are
    plain Python values, so `reshape(x, [B, H * W])` and `for i in 0..L`
    are resolved here. `for` loops, and `if`/`while` on compile-time
    conditions, are resolved completely.
  * Data-dependent control flow: `if`/`while` on a *tensor* condition become
    structured `if`/`while` ops that own sub-graphs. Variables assigned inside
    become the op's results (loop-carried state for `while`); outer values
    used inside are captured as explicit inputs.
  * Automatic differentiation: `grad(y, x)` appends the backward pass (see autodiff.py).
"""
from __future__ import annotations

import difflib
import itertools
from typing import Optional, Union

import numpy as np

from . import autodiff, ir
from . import ops as O
from . import syntax as S
from .errors import Loc, MiraError
from .types import NP_DTYPES, TensorType


class DTypeValue(str):
    """A dtype used as a value, e.g. the `f32` in `cast(x, f32)`."""


CTValue = Union[int, float, bool, list, DTypeValue]   # compile-time values
Val = Union[ir.Value, tuple, CTValue]                  # tuples come from multi-value returns

REQUIRED = object()
MAX_UNROLL = 10_000
MAX_DEPTH = 64
MAX_ITERS = 100_000

BINOP_KIND = {"+": "add", "-": "sub", "*": "mul", "/": "div", "**": "pow"}


def describe(v: Val) -> str:
    if isinstance(v, ir.Value):
        return f"tensor {v.type}"
    if isinstance(v, tuple):
        return f"tuple of {len(v)} values"
    if isinstance(v, DTypeValue):
        return f"dtype {v}"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return f"number {v}"
    if isinstance(v, list):
        return "list"
    return type(v).__name__


def is_num(v: Val) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class Scope:
    def __init__(self, parent: Optional["Scope"] = None):
        self.vars: dict[str, Val] = {}
        self.parent = parent

    def lookup(self, name: str) -> Optional[Val]:
        s = self.owner(name)
        return s.vars[name] if s is not None else None

    def owner(self, name: str) -> Optional["Scope"]:
        s: Optional[Scope] = self
        while s is not None:
            if name in s.vars:
                return s
            s = s.parent
        return None

    def all_names(self) -> list[str]:
        names, s = [], self
        while s is not None:
            names.extend(s.vars)
            s = s.parent
        return names

    def snapshot(self) -> dict[tuple[int, str], tuple["Scope", Val]]:
        out, s = {}, self
        while s is not None:
            for n, v in s.vars.items():
                out[(id(s), n)] = (s, v)
            s = s.parent
        return out


def restore(snap: dict) -> dict[tuple[int, str], Val]:
    """Put snapshotted variables back; return the ones that had changed (key -> new value)."""
    changed = {}
    for key, (scope, old) in snap.items():
        new = scope.vars[key[1]]
        if new is not old:
            changed[key] = new
            scope.vars[key[1]] = old
    return changed


def assigned_names(body: list[S.Stmt]) -> list[str]:
    out: list[str] = []
    for st in body:
        if isinstance(st, S.Assign):
            out.extend(n for n in st.names if n not in out)
        if isinstance(st, S.IndexAssign) and st.name not in out:
            out.append(st.name)
        for sub in (getattr(st, "body", None), getattr(st, "then", None), getattr(st, "orelse", None)):
            if sub:
                out.extend(n for n in assigned_names(sub) if n not in out)
    return out


class GraphCtx:
    """The graph currently being built, plus what it captured from enclosing graphs."""

    def __init__(self, graph: ir.Graph, parent: Optional["GraphCtx"]):
        self.graph = graph
        self.parent = parent
        self.captures: dict[ir.Value, ir.Value] = {}   # outer value -> placeholder input
        self.captured_outer: list[ir.Value] = []       # in input order
        self._owned: set[ir.Value] = set(graph.inputs)
        self._synced = 0

    def owns(self, v: ir.Value) -> bool:
        for op in self.graph.ops[self._synced:]:
            self._owned.update(op.results)
        self._synced = len(self.graph.ops)
        self._owned.update(self.graph.inputs)
        return v in self._owned


class Elaborator:
    def __init__(self, module: S.Module, graph_name: str):
        self.module = module
        self.ctx = GraphCtx(ir.Graph(graph_name, [], [], []), None)
        self.stack: list[str] = []
        self.counter = itertools.count()

    @property
    def graph(self) -> ir.Graph:
        return self.ctx.graph

    # ================================================================ graphs and captures

    def local(self, v: ir.Value, ctx: Optional[GraphCtx] = None) -> ir.Value:
        """Make `v` usable in `ctx`'s graph, capturing it from an enclosing graph if needed."""
        ctx = ctx or self.ctx
        if ctx.owns(v):
            return v
        if v in ctx.captures:
            return ctx.captures[v]
        if ctx.parent is None:
            raise AssertionError(f"{v} is not defined in any enclosing graph")
        if v.producer is not None and v.producer.is_const:   # constants are copied, not passed in
            c = ctx.graph.const(v.producer.attrs["value"], v.producer.loc)
        else:
            c = ir.Value(v.type)
            ctx.graph.inputs.append(c)
            ctx.captured_outer.append(v)
        ctx.captures[v] = c
        return c

    def enter(self, name: str) -> GraphCtx:
        self.ctx = GraphCtx(ir.Graph(f"{self.graph.name}.{name}", [], [], []), self.ctx)
        return self.ctx

    def exit(self, ctx: GraphCtx) -> None:
        assert self.ctx is ctx
        self.ctx = ctx.parent

    # ================================================================ entry

    def entry(self, fn_name: str, weights: dict[str, np.ndarray], dims: dict[str, int],
              seed: Optional[int]) -> ir.Graph:
        fn = self.module.functions.get(fn_name)
        if fn is None:
            raise MiraError(f"no function named '{fn_name}' (the entry point)")
        bindings = dict(dims)
        args: list[Val] = []
        rng = np.random.default_rng(seed) if seed is not None else None
        for p in fn.params:
            if p.type.is_int:
                if p.name not in bindings:
                    raise MiraError(f"int parameter '{p.name}' of the entry function needs a value "
                                    f"(dims={{'{p.name}': ...}} or --dim {p.name}=...)", p.loc)
                args.append(bindings[p.name])
                continue
            ty = self.resolve_type(p.type, bindings, entry=True)
            if p.is_const:
                if p.name in weights:
                    arr = np.asarray(weights[p.name])
                    if tuple(arr.shape) != ty.shape:
                        raise MiraError(f"weight '{p.name}' has shape {list(arr.shape)} but the parameter is {ty}",
                                        p.loc)
                    arr = arr.astype(ty.np_dtype)
                elif rng is not None:
                    arr = random_weight(rng, ty)
                else:
                    raise MiraError(f"no value supplied for const parameter '{p.name}' "
                                    f"(pass weights, or use random weights)", p.loc)
                args.append(self.graph.const(arr, p.loc))
            else:
                v = ir.Value(ty, name=p.name)
                self.graph.inputs.append(v)
                args.append(v)
        unknown = set(weights) - {p.name for p in fn.params if p.is_const}
        if unknown:
            raise MiraError(f"weights given for unknown const parameters: {', '.join(sorted(unknown))}", fn.loc)
        out = self.call_function(fn, args, [p.loc for p in fn.params], fn.loc, preset=bindings)
        self.graph.outputs = list(out) if isinstance(out, tuple) else [out]
        return self.graph

    # ================================================================ types

    def resolve_dim(self, d: Union[int, str, S.Expr], bindings: dict[str, int], loc: Loc, entry: bool) -> int:
        if isinstance(d, int):
            return d
        if isinstance(d, str):
            if d in bindings:
                return bindings[d]
            if entry:
                raise MiraError(f"dimension '{d}' of the entry function isn't known; "
                                f"give it a value (e.g. dims={{'{d}': 8}} or --dim {d}=8)", loc)
            raise MiraError(f"dimension '{d}' isn't bound by any parameter", loc)
        scope = Scope()
        scope.vars.update(bindings)
        try:
            v = self.expr(d, scope)
        except MiraError as err:
            raise MiraError(f"in dimension expression: {err.message}", d.loc) from None
        n = self.ct_int(v, d.loc, "dimension")
        if n < 0:
            raise MiraError(f"dimension evaluates to {n}", d.loc)
        return n

    def resolve_type(self, t: S.TypeExpr, bindings: dict[str, int], entry: bool = False) -> TensorType:
        return TensorType(t.dtype, tuple(self.resolve_dim(d, bindings, t.loc, entry) for d in t.dims))

    def unify(self, pty: S.TypeExpr, actual: TensorType, bindings: dict[str, int], what: str, loc: Loc) -> None:
        """Bind shape symbols in `pty` to the dims of `actual`.

        Computed dims (like H / 2) can't bind anything; call_function checks them
        once every parameter has been unified.
        """
        def fail(reason: str) -> MiraError:
            return MiraError(f"{what} has type {actual}, but the parameter expects {pty} ({reason})", loc)

        if pty.dtype != actual.dtype:
            raise fail(f"{actual.dtype} != {pty.dtype}")
        if len(pty.dims) != actual.rank:
            raise fail(f"rank {actual.rank} != {len(pty.dims)}")
        for d, a in zip(pty.dims, actual.shape):
            if isinstance(d, int):
                if d != a:
                    raise fail(f"{a} != {d}")
            elif isinstance(d, str):
                if d in bindings and bindings[d] != a:
                    raise fail(f"{d} is already {bindings[d]}, but got {a}")
                bindings[d] = a

    # ================================================================ functions

    def call_function(self, fn: S.FnDecl, args: list[Val], arg_locs: list[Loc], call_loc: Loc,
                      preset: Optional[dict[str, int]] = None) -> Union[ir.Value, tuple]:
        if fn.name in self.stack:
            raise MiraError(f"recursive call to '{fn.name}' (recursion can't be compiled to a static graph)", call_loc)
        if len(self.stack) >= MAX_DEPTH:
            raise MiraError("function calls nested too deeply", call_loc)
        if len(args) != len(fn.params):
            raise MiraError(f"'{fn.name}' takes {len(fn.params)} arguments, got {len(args)}", call_loc)

        bindings = dict(preset or {})
        # 1. compile-time int parameters bind first, so shapes can mention them
        for p, a, aloc in zip(fn.params, args, arg_locs):
            if p.type.is_int:
                bindings[p.name] = self.ct_int(a, aloc, f"argument '{p.name}' of '{fn.name}'")
        # 2. unify tensor arguments with parameter types, binding shape symbols
        tensors: dict[str, ir.Value] = {}
        for p, a, aloc in zip(fn.params, args, arg_locs):
            if p.type.is_int:
                continue
            if p.is_const and self.stack:
                raise MiraError("'const' is only allowed on the entry function's parameters", p.loc)
            if not isinstance(a, ir.Value):
                if is_num(a):
                    a = self.graph.const(np.array(a, dtype=NP_DTYPES[p.type.dtype]), aloc)
                else:
                    raise MiraError(f"argument '{p.name}' of '{fn.name}' must be a tensor, got {describe(a)}", aloc)
            self.unify(p.type, a.type, bindings, f"argument '{p.name}' of '{fn.name}'", aloc)
            tensors[p.name] = a
        # 3. check computed dims (like H / 2), now that every symbol is bound
        for p, aloc in zip(fn.params, arg_locs):
            if p.name in tensors:
                want = self.resolve_type(p.type, bindings)
                if want != tensors[p.name].type:
                    raise MiraError(f"argument '{p.name}' of '{fn.name}' has type {tensors[p.name].type}, "
                                    f"but the parameter expects {p.type} = {want}", aloc)

        scope = Scope()
        for sym, val in bindings.items():
            scope.vars[sym] = val
        for p in fn.params:
            if p.name in tensors:
                if p.name in scope.vars:
                    raise MiraError(f"parameter '{p.name}' has the same name as a shape dimension", p.loc)
                scope.vars[p.name] = tensors[p.name]

        self.stack.append(fn.name)
        try:
            result = self.block(fn.body, scope, nested=False)
            if result is None:
                raise MiraError(f"function '{fn.name}' doesn't return a value", fn.loc)
            value, rloc = result
            ret_tys = [self.resolve_type(t, bindings) for t in fn.rets]
            values = list(value) if isinstance(value, tuple) else [value]
            if len(values) != len(ret_tys):
                raise MiraError(f"'{fn.name}' returns {len(values)} value(s), but its signature declares "
                                f"{len(ret_tys)}", rloc)
            out = []
            for v, want, decl in zip(values, ret_tys, fn.rets):
                v = self.local(self.as_tensor(v, rloc, dtype=want.dtype))
                if v.type != want:
                    raise MiraError(f"'{fn.name}' returns {v.type}, but its signature says {want}", rloc,
                                    notes=[f"declared return type: {decl}"])
                out.append(v)
            return tuple(out) if len(fn.rets) > 1 else out[0]
        finally:
            self.stack.pop()

    # ================================================================ statements

    def block(self, body: list[S.Stmt], scope: Scope, nested: bool) -> Optional[tuple[Val, Loc]]:
        for i, st in enumerate(body):
            if isinstance(st, S.Let):
                self.let(st, scope)
            elif isinstance(st, S.Assign):
                self.assign(st, scope)
            elif isinstance(st, S.IndexAssign):
                self.index_assign(st, scope)
            elif isinstance(st, S.For):
                start = self.ct_int(self.expr(st.start, scope), st.start.loc, "loop bound")
                stop = self.ct_int(self.expr(st.stop, scope), st.stop.loc, "loop bound")
                if stop - start > MAX_UNROLL:
                    raise MiraError(f"loop has {stop - start} iterations; at most {MAX_UNROLL} can be unrolled", st.loc)
                for it in range(start, stop):
                    inner = Scope(scope)
                    inner.vars[st.var] = it
                    self.block(st.body, inner, nested=True)
            elif isinstance(st, S.If):
                self.if_stmt(st, scope)
            elif isinstance(st, S.While):
                self.while_stmt(st, scope)
            elif isinstance(st, S.Return):
                if nested:
                    raise MiraError("'return' must be the last statement of a function, not inside for/if/while; "
                                    "assign to a variable instead", st.loc)
                if i != len(body) - 1:
                    raise MiraError("unreachable code after 'return'", body[i + 1].loc)
                return self.expr(st.value, scope), st.value.loc
        return None

    def destructure(self, names: list[str], v: Val, loc: Loc) -> list[Val]:
        if len(names) == 1:
            if isinstance(v, tuple):
                raise MiraError(f"this produces {len(v)} values; unpack them like 'let a, b = ...'", loc)
            return [v]
        if not isinstance(v, tuple) or len(v) != len(names):
            raise MiraError(f"can't unpack {describe(v)} into {len(names)} names", loc)
        return list(v)

    def let(self, st: S.Let, scope: Scope) -> None:
        for n in st.names:
            if n in scope.vars:
                raise MiraError(f"'{n}' is already defined in this scope; use '{n} = ...' to reassign", st.loc)
        v = self.expr(st.value, scope)
        if st.type is not None:
            want = self.resolve_type(st.type, self.shape_bindings(scope))
            v = self.as_tensor(v, st.value.loc, dtype=want.dtype)
            if v.type != want:
                raise MiraError(f"'{st.names[0]}' is declared {want} but the value is {v.type}", st.value.loc)
        for n, x in zip(st.names, self.destructure(st.names, v, st.value.loc)):
            scope.vars[n] = x

    def assign(self, st: S.Assign, scope: Scope) -> None:
        values = self.destructure(st.names, self.expr(st.value, scope), st.value.loc)
        for n, v in zip(st.names, values):
            owner = scope.owner(n)
            if owner is None:
                raise MiraError(f"assignment to undefined variable '{n}'; declare it with 'let'", st.loc)
            old = owner.vars[n]
            if isinstance(old, ir.Value):
                v = self.as_tensor(v, st.value.loc, dtype=old.type.dtype)
                if v.type != old.type:
                    raise MiraError(f"'{n}' has type {old.type}; can't assign a {v.type}", st.value.loc)
            elif isinstance(v, (ir.Value, tuple)):
                raise MiraError(f"'{n}' is a compile-time {describe(old)}; can't assign a {describe(v)}", st.loc)
            owner.vars[n] = v

    def truthy(self, v: Val, loc: Loc) -> Optional[bool]:
        """Compile-time truth value of a condition, or None if it's a runtime tensor."""
        if isinstance(v, ir.Value):
            if v.type.numel != 1:
                raise MiraError(f"a condition must be a single value, got {v.type}; reduce it first "
                                f"(e.g. with max or sum)", loc)
            return None
        if isinstance(v, bool) or is_num(v):
            return bool(v)
        raise MiraError(f"a condition must be a bool, a number, or a one-element tensor; got {describe(v)}", loc)

    # ----- if

    def if_stmt(self, st: S.If, scope: Scope) -> None:
        cond = self.expr(st.cond, scope)
        static = self.truthy(cond, st.cond.loc)
        if static is not None:          # decided at compile time
            self.block(st.then if static else st.orelse, Scope(scope), nested=True)
            return

        n = next(self.counter)
        branches = []
        for label, body in (("then", st.then), ("else", st.orelse)):
            snap = scope.snapshot()
            ctx = self.enter(f"if{n}.{label}")
            self.block(body, Scope(scope), nested=True)
            self.exit(ctx)
            branches.append((ctx, restore(snap)))

        keys = list(dict.fromkeys(k for _, changed in branches for k in changed))
        snap = scope.snapshot()
        result_types = []
        for key in keys:
            owner, orig = snap[key]
            if not isinstance(orig, ir.Value):
                raise MiraError(f"'{key[1]}' is a compile-time value; it can't change inside an 'if' whose "
                                f"condition is only known at runtime", st.loc)
            result_types.append(orig.type)
        for ctx, changed in branches:
            ctx.graph.outputs = [self.local(changed.get(k, snap[k][1]), ctx) for k in keys]

        (then_ctx, _), (else_ctx, _) = branches
        inputs = [self.local(cond)] + [self.local(v) for v in then_ctx.captured_outer] + \
                 [self.local(v) for v in else_ctx.captured_outer]
        results = self.graph.add_control(
            "if", inputs, {"then": then_ctx.graph, "else": else_ctx.graph, "n_then": len(then_ctx.captured_outer)},
            result_types, st.loc)
        for key, r in zip(keys, results):
            snap[key][0].vars[key[1]] = r

    # ----- while

    def while_stmt(self, st: S.While, scope: Scope) -> None:
        cond = self.expr(st.cond, scope)
        if self.truthy(cond, st.cond.loc) is not None:     # compile-time loop: run it now
            for _ in range(MAX_UNROLL):
                if not self.truthy(self.expr(st.cond, scope), st.cond.loc):
                    return
                self.block(st.body, Scope(scope), nested=True)
            raise MiraError(f"compile-time while loop ran more than {MAX_UNROLL} iterations", st.loc)

        names = []
        for name in assigned_names(st.body):
            v = scope.lookup(name)
            if v is None:
                continue
            if not isinstance(v, ir.Value):
                raise MiraError(f"'{name}' is a compile-time value; it can't change inside a 'while' whose "
                                f"condition is only known at runtime", st.loc)
            names.append(name)
        init = [scope.lookup(nm) for nm in names]
        n = next(self.counter)

        def state_scope(ctx: GraphCtx) -> Scope:
            s = Scope(scope)
            for nm, v in zip(names, init):
                ph = ir.Value(v.type)
                ctx.graph.inputs.append(ph)
                s.vars[nm] = ph
            return s

        cond_ctx = self.enter(f"while{n}.cond")
        c = self.expr(st.cond, state_scope(cond_ctx))
        if not isinstance(c, ir.Value) or c.type.numel != 1:
            raise MiraError("the loop condition must stay a one-element tensor", st.cond.loc)
        cond_ctx.graph.outputs = [self.local(c)]
        self.exit(cond_ctx)

        body_ctx = self.enter(f"while{n}.body")
        s = state_scope(body_ctx)
        snap = scope.snapshot()
        self.block(st.body, Scope(s), nested=True)
        if restore(snap):
            raise MiraError("a compile-time value changed inside a runtime 'while' loop", st.loc)
        body_ctx.graph.outputs = [self.local(s.vars[nm]) for nm in names]
        self.exit(body_ctx)

        inputs = [self.local(v) for v in init] + [self.local(v) for v in cond_ctx.captured_outer] + \
                 [self.local(v) for v in body_ctx.captured_outer]
        results = self.graph.add_control(
            "while", inputs, {"cond": cond_ctx.graph, "body": body_ctx.graph, "n_state": len(names),
                              "n_cond": len(cond_ctx.captured_outer), "max_iters": MAX_ITERS},
            [v.type for v in init], st.loc)
        for nm, r in zip(names, results):
            scope.owner(nm).vars[nm] = r

    def shape_bindings(self, scope: Scope) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in scope.all_names():
            v = scope.lookup(n)
            if isinstance(v, int) and not isinstance(v, bool):
                out.setdefault(n, v)
        return out

    # ================================================================ expressions

    def expr(self, e: S.Expr, scope: Scope) -> Val:
        if isinstance(e, S.Num):
            return e.value
        if isinstance(e, S.Bool):
            return e.value
        if isinstance(e, S.DType):
            return DTypeValue(e.name)
        if isinstance(e, S.ListLit):
            return [self.expr(i, scope) for i in e.items]
        if isinstance(e, S.TupleLit):
            return tuple(self.expr(i, scope) for i in e.items)
        if isinstance(e, S.Name):
            v = scope.lookup(e.ident)
            if v is None:
                if e.ident in self.module.functions or e.ident in BUILTINS:
                    raise MiraError(f"'{e.ident}' is a function; call it like {e.ident}(...)", e.loc)
                hint = difflib.get_close_matches(e.ident, scope.all_names(), n=1)
                raise MiraError(f"undefined name '{e.ident}'", e.loc,
                                notes=[f"did you mean '{hint[0]}'?"] if hint else None)
            return v
        if isinstance(e, S.Unary):
            v = self.expr(e.operand, scope)
            if isinstance(v, ir.Value):
                return self.op("neg", [v], {}, e.loc)
            if is_num(v):
                return -v
            raise MiraError(f"can't negate a {describe(v)}", e.loc)
        if isinstance(e, S.Binary):
            return self.binary(e.op, self.expr(e.lhs, scope), self.expr(e.rhs, scope), e.loc)
        if isinstance(e, S.Call):
            return self.call(e, scope)
        if isinstance(e, S.Index):
            return self.index(e, scope)
        raise AssertionError(e)

    def binary(self, op: str, a: Val, b: Val, loc: Loc) -> Val:
        if op in S.COMPARE_OPS and not isinstance(a, ir.Value) and not isinstance(b, ir.Value):
            if not ((is_num(a) or isinstance(a, bool)) and (is_num(b) or isinstance(b, bool))):
                raise MiraError(f"can't compare {describe(a)} and {describe(b)}", loc)
            return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[op]
        if is_num(a) and is_num(b):
            if op == "@":
                raise MiraError("'@' needs tensor operands", loc)
            if op == "/":
                if b == 0:
                    raise MiraError("division by zero in a compile-time expression", loc)
                return a // b if isinstance(a, int) and isinstance(b, int) and a % b == 0 else a / b
            return {"+": a + b, "-": a - b, "*": a * b, "**": a ** b}[op]
        if not isinstance(a, ir.Value) and not isinstance(b, ir.Value):
            raise MiraError(f"can't apply '{op}' to {describe(a)} and {describe(b)}", loc)
        dtype = a.type.dtype if isinstance(a, ir.Value) else b.type.dtype
        a = self.as_tensor(a, loc, dtype)
        b = self.as_tensor(b, loc, dtype)
        if op == "@":
            return self.op("matmul", [a, b], {}, loc)
        if op in S.COMPARE_OPS:     # tensor comparisons give 1.0 / 0.0 masks
            if op == "<":
                return self.op("greater", [b, a], {}, loc)
            if op == "<=":
                return self.op("greater_equal", [b, a], {}, loc)
            if op == ">":
                return self.op("greater", [a, b], {}, loc)
            if op == ">=":
                return self.op("greater_equal", [a, b], {}, loc)
            eq = self.op("equal", [a, b], {}, loc)
            return eq if op == "==" else self.op("sub", [self.as_tensor(1, loc, dtype), eq], {}, loc)
        return self.op(BINOP_KIND[op], [a, b], {}, loc)

    def as_tensor(self, v: Val, loc: Loc, dtype: str = "f32") -> ir.Value:
        """Tensors pass through; numbers become scalar constants of `dtype`."""
        if isinstance(v, ir.Value):
            return v
        if is_num(v):
            if dtype == "i32" and float(v) != int(v):
                raise MiraError(f"can't mix the number {v} with an i32 tensor; cast the tensor to f32 first", loc)
            return self.graph.const(np.array(v, dtype=NP_DTYPES[dtype]), loc)
        raise MiraError(f"expected a tensor, got {describe(v)}", loc)

    def op(self, kind: str, inputs: list[ir.Value], attrs: dict, loc: Loc) -> ir.Value:
        try:
            return self.graph.add(kind, [self.local(v) for v in inputs], attrs, loc)
        except O.OpTypeError as err:
            raise MiraError(str(err), loc) from None

    # ================================================================ indexing

    def subscripts(self, items: list[S.IndexItem], shape: tuple[int, ...], scope: Scope, loc: Loc):
        """Classify each subscript as ("int", k) / ("slice", a, b) / ("tensor", value); pad with full slices."""
        if len(items) > len(shape):
            raise MiraError(f"too many indices ({len(items)}) for a rank-{len(shape)} tensor", loc)
        out = []
        for axis, (it, d) in enumerate(zip(items + [None] * (len(shape) - len(items)), shape)):
            if it is None:
                out.append(("slice", 0, d))
                continue
            if it.is_slice:
                bounds = []
                for b, default in ((it.start, 0), (it.stop, d)):
                    v = default if b is None else self.expr(b, scope)
                    if isinstance(v, ir.Value):
                        raise MiraError("slice bounds must be compile-time numbers; for a runtime start use "
                                        "dynamic_slice(x, start, size)", it.loc)
                    v = self.ct_int(v, it.loc, "slice bound")
                    bounds.append(max(0, min(d, v + d if v < 0 else v)))
                a, b = bounds
                if b <= a:
                    raise MiraError(f"empty slice {a}:{b} on an axis of size {d}", it.loc)
                out.append(("slice", a, b))
                continue
            v = self.expr(it.start, scope)
            if isinstance(v, ir.Value):
                if v.type.dtype != "i32":
                    raise MiraError(f"tensor indices must be i32, got {v.type}; use cast(i, i32)", it.loc)
                out.append(("tensor", self.local(v)))
            else:
                k = self.ct_int(v, it.loc, "index")
                if not -d <= k < d:
                    raise MiraError(f"index {k} is out of range for an axis of size {d}", it.loc)
                out.append(("int", k % d))
        return out

    def index(self, e: S.Index, scope: Scope) -> ir.Value:
        x = self.expr(e.base, scope)
        if not isinstance(x, ir.Value):
            raise MiraError(f"can't index a {describe(x)}", e.loc)
        x = self.local(x)
        subs = self.subscripts(e.items, x.type.shape, scope, e.loc)
        # 1. every static subscript at once, as one slice op
        begin = tuple(s[1] if s[0] != "tensor" else 0 for s in subs)
        size = tuple(1 if s[0] == "int" else s[2] - s[1] if s[0] == "slice" else d
                     for s, d in zip(subs, x.type.shape))
        if size != x.type.shape:
            x = self.op("slice", [x], {"begin": begin, "size": size}, e.loc)
        # 2. drop the axes indexed by a single integer
        kept = [i for i, s in enumerate(subs) if s[0] != "int"]
        if len(kept) != len(subs):
            x = self.op("reshape", [x], {"shape": tuple(size[i] for i in kept)}, e.loc)
        # 3. runtime (tensor) indices become gathers
        pos_shift = 0
        for new_axis, i in enumerate(kept):
            if subs[i][0] == "tensor":
                idx = subs[i][1]
                x = self.op("gather", [x, idx], {"axis": new_axis + pos_shift}, e.loc)
                pos_shift += idx.type.rank - 1
        return x

    def start_vector(self, parts: list[Union[int, ir.Value]], loc: Loc) -> ir.Value:
        """i32[n] start position from compile-time ints and one-element i32 tensors."""
        if all(isinstance(p, int) for p in parts):
            return self.graph.const(np.array(parts, dtype=np.int32), loc)
        pieces = []
        for p in parts:
            if isinstance(p, int):
                pieces.append(self.graph.const(np.array([p], dtype=np.int32), loc))
            else:
                if p.type.dtype != "i32" or p.type.numel != 1:
                    raise MiraError(f"a runtime position must be a one-element i32 tensor, got {p.type}", loc)
                pieces.append(self.op("reshape", [p], {"shape": (1,)}, loc))
        return self.op("concat", pieces, {"axis": 0}, loc)

    def index_assign(self, st: S.IndexAssign, scope: Scope) -> None:
        owner = scope.owner(st.name)
        if owner is None:
            raise MiraError(f"assignment to undefined variable '{st.name}'; declare it with 'let'", st.loc)
        x = owner.vars[st.name]
        if not isinstance(x, ir.Value):
            raise MiraError(f"'{st.name}' is a compile-time {describe(x)}; only tensors can be indexed", st.loc)
        x = self.local(x)
        subs = self.subscripts(st.items, x.type.shape, scope, st.loc)
        starts, region, value_shape = [], [], []
        for s, d in zip(subs, x.type.shape):
            if s[0] == "int":
                starts.append(s[1])
                region.append(1)
            elif s[0] == "slice":
                starts.append(s[1])
                region.append(s[2] - s[1])
                value_shape.append(s[2] - s[1])
            else:
                if s[1].type.numel != 1:
                    raise MiraError("indexed assignment takes one position per axis; use scatter_add for many",
                                    st.loc)
                starts.append(s[1])
                region.append(1)
        v = self.as_tensor(self.expr(st.value, scope), st.value.loc, dtype=x.type.dtype)
        if v.type.dtype != x.type.dtype:
            raise MiraError(f"can't store a {v.type} into '{st.name}' ({x.type})", st.value.loc)
        if v.type.shape != tuple(value_shape):
            v = self.op("broadcast", [v], {"shape": tuple(value_shape)}, st.value.loc)
        v = self.op("reshape", [v], {"shape": tuple(region)}, st.value.loc)
        start = self.start_vector(starts, st.loc)
        owner.vars[st.name] = self.op("dynamic_update_slice", [x, v, start], {}, st.loc)

    # ================================================================ calls

    def call(self, e: S.Call, scope: Scope) -> Val:
        if e.callee in self.module.functions:
            fn = self.module.functions[e.callee]
            by_name = {p.name: i for i, p in enumerate(fn.params)}
            args: list[Optional[Val]] = [None] * len(fn.params)
            locs: list[Loc] = [e.loc] * len(fn.params)
            for i, a in enumerate(e.args):
                if a.name is None:
                    if i >= len(fn.params):
                        raise MiraError(f"'{fn.name}' takes {len(fn.params)} arguments, got {len(e.args)}", a.loc)
                    idx = i
                elif a.name in by_name:
                    idx = by_name[a.name]
                else:
                    raise MiraError(f"'{fn.name}' has no parameter named '{a.name}'", a.loc)
                if args[idx] is not None:
                    raise MiraError(f"argument '{fn.params[idx].name}' given twice", a.loc)
                args[idx] = self.expr(a.value, scope)
                locs[idx] = a.loc
            missing = [fn.params[i].name for i, a in enumerate(args) if a is None]
            if missing:
                raise MiraError(f"call to '{fn.name}' is missing argument(s): {', '.join(missing)}", e.loc)
            args = [self.local(a) if isinstance(a, ir.Value) else a for a in args]
            return self.call_function(fn, args, locs, e.loc)
        if e.callee in BUILTINS:
            params, handler = BUILTINS[e.callee]
            bound = self.bind_builtin(e, params, scope)
            return handler(self, bound, e.loc)
        hint = difflib.get_close_matches(e.callee, list(self.module.functions) + list(BUILTINS), n=1)
        raise MiraError(f"unknown function '{e.callee}'", e.loc, notes=[f"did you mean '{hint[0]}'?"] if hint else None)

    def bind_builtin(self, e: S.Call, params: list[tuple[str, object]], scope: Scope) -> dict[str, tuple[Val, Loc]]:
        names = [n for n, _ in params]
        out: dict[str, tuple[Val, Loc]] = {}
        for i, a in enumerate(e.args):
            if a.name is None:
                if i >= len(params):
                    raise MiraError(f"{e.callee}() takes at most {len(params)} arguments", a.loc)
                name = names[i]
            elif a.name in names:
                name = a.name
            else:
                raise MiraError(f"{e.callee}() has no argument named '{a.name}' (expected one of: {', '.join(names)})",
                                a.loc)
            if name in out:
                raise MiraError(f"argument '{name}' given twice", a.loc)
            out[name] = (self.expr(a.value, scope), a.loc)
        for name, default in params:
            if name not in out:
                if default is REQUIRED:
                    raise MiraError(f"{e.callee}() is missing required argument '{name}'", e.loc)
                out[name] = (default, e.loc)
        return out

    # ----- argument coercions -----

    def tensor(self, arg: tuple[Val, Loc]) -> ir.Value:
        v, loc = arg
        if not isinstance(v, ir.Value):
            raise MiraError(f"expected a tensor, got {describe(v)}", loc)
        return self.local(v)

    def ct_int(self, v: Val, loc: Loc, what: str = "argument") -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            if isinstance(v, float) and v.is_integer():
                return int(v)
            raise MiraError(f"{what} must be a compile-time integer, got {describe(v)}", loc)
        return v

    def int_(self, arg: tuple[Val, Loc]) -> int:
        return self.ct_int(arg[0], arg[1])

    def ints(self, arg: tuple[Val, Loc]) -> list[int]:
        v, loc = arg
        if not isinstance(v, list):
            raise MiraError(f"expected a list of integers like [1, 2], got {describe(v)}", loc)
        return [self.ct_int(x, loc) for x in v]

    def pair(self, arg: tuple[Val, Loc]) -> tuple[int, int]:
        v, loc = arg
        if isinstance(v, list):
            xs = self.ints(arg)
            if len(xs) != 2:
                raise MiraError("expected one integer or a pair [a, b]", loc)
            return xs[0], xs[1]
        x = self.int_(arg)
        return x, x

    def float_(self, arg: tuple[Val, Loc]) -> float:
        v, loc = arg
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise MiraError(f"expected a compile-time number, got {describe(v)}", loc)
        return float(v)

    def bool_(self, arg: tuple[Val, Loc]) -> bool:
        v, loc = arg
        if not isinstance(v, bool):
            raise MiraError(f"expected true or false, got {describe(v)}", loc)
        return v


# ==================================================================== builtins

def _unary(kind):
    def h(el: Elaborator, a, loc):
        v, vloc = a["x"]
        if isinstance(v, (int, float)) and not isinstance(v, bool):   # compile-time math, e.g. sqrt(64)
            with np.errstate(all="raise"):
                try:
                    return float(O.UNARY[kind](np.float64(v)))
                except FloatingPointError:
                    raise MiraError(f"{kind}({v}) is undefined", loc) from None
        return el.op(kind, [el.tensor(a["x"])], {}, loc)
    return [("x", REQUIRED)], h


def _binary_fn(kind):
    def h(el: Elaborator, a, loc):
        x, y = a["a"][0], a["b"][0]
        if not isinstance(x, ir.Value) and not isinstance(y, ir.Value):
            raise MiraError(f"{kind}() needs at least one tensor argument", loc)
        dt = x.type.dtype if isinstance(x, ir.Value) else y.type.dtype
        return el.op(kind, [el.as_tensor(x, a["a"][1], dt), el.as_tensor(y, a["b"][1], dt)], {}, loc)
    return [("a", REQUIRED), ("b", REQUIRED)], h


def _reduce(kind):
    def h(el: Elaborator, a, loc):
        x = el.tensor(a["x"])
        axis_v, axis_loc = a["axis"]
        if axis_v is None:
            axes = list(range(x.type.rank))
        elif isinstance(axis_v, list):
            axes = el.ints(a["axis"])
        else:
            axes = [el.int_(a["axis"])]
        return el.op(kind, [x], {"axes": tuple(axes), "keepdims": el.bool_(a["keepdims"])}, loc)
    return [("x", REQUIRED), ("axis", None), ("keepdims", False)], h


def _axis_op(kind):
    return ([("x", REQUIRED), ("axis", -1)],
            lambda el, a, loc: el.op(kind, [el.tensor(a["x"])], {"axis": el.int_(a["axis"])}, loc))


def _transpose(el: Elaborator, a, loc):
    x = el.tensor(a["x"])
    if a["perm"][0] is None:
        if x.type.rank < 2:
            raise MiraError(f"transpose() of rank-{x.type.rank} tensor needs an explicit perm", loc)
        perm = list(range(x.type.rank))
        perm[-2], perm[-1] = perm[-1], perm[-2]
    else:
        perm = el.ints(a["perm"])
    return el.op("transpose", [x], {"perm": tuple(perm)}, loc)


def _reshape(el: Elaborator, a, loc):
    x = el.tensor(a["x"])
    shape = el.ints(a["shape"])
    if shape.count(-1) > 1:
        raise MiraError("reshape: at most one dimension can be -1", a["shape"][1])
    if -1 in shape:
        known = int(np.prod([d for d in shape if d != -1], dtype=np.int64))
        if known == 0 or x.type.numel % known:
            raise MiraError(f"reshape: can't infer -1 when reshaping {x.type} to {shape}", a["shape"][1])
        shape[shape.index(-1)] = x.type.numel // known
    return el.op("reshape", [x], {"shape": tuple(shape)}, loc)


def _flatten(el: Elaborator, a, loc):
    x = el.tensor(a["x"])
    ax = el.int_(a["axis"])
    try:
        ax = O.norm_axis(ax, x.type.rank + 1, "flatten")
    except O.OpTypeError as err:
        raise MiraError(str(err), loc) from None
    lead = int(np.prod(x.type.shape[:ax], dtype=np.int64))
    return el.op("reshape", [x], {"shape": (lead, x.type.numel // max(lead, 1))}, loc)


def _conv2d(el: Elaborator, a, loc):
    return el.op("conv2d", [el.tensor(a["x"]), el.tensor(a["w"])],
                 {"stride": el.pair(a["stride"]), "padding": el.pair(a["padding"])}, loc)


def _maxpool(el: Elaborator, a, loc):
    size = el.int_(a["size"])
    stride = size if a["stride"][0] is None else el.int_(a["stride"])
    return el.op("maxpool2d", [el.tensor(a["x"])], {"size": size, "stride": stride}, loc)


def _layernorm(el: Elaborator, a, loc):
    return el.op("layernorm", [el.tensor(a["x"]), el.tensor(a["gamma"]), el.tensor(a["beta"])],
                 {"eps": el.float_(a["eps"])}, loc)


def _concat(el: Elaborator, a, loc):
    vals, vloc = a["xs"]
    if not isinstance(vals, list) or not vals:
        raise MiraError("concat() takes a non-empty list of tensors, like concat([a, b], axis=0)", vloc)
    return el.op("concat", [el.tensor((v, vloc)) for v in vals], {"axis": el.int_(a["axis"])}, loc)


def _cast(el: Elaborator, a, loc):
    x = el.tensor(a["x"])
    dt, dloc = a["dtype"]
    if not isinstance(dt, DTypeValue):
        raise MiraError(f"cast() needs a dtype like f16 or f32, got {describe(dt)}", dloc)
    if x.type.dtype == dt:
        return x
    return el.op("cast", [x], {"dtype": str(dt)}, loc)


def _size(el: Elaborator, a, loc):
    x = el.tensor(a["x"])
    try:
        return x.type.shape[O.norm_axis(el.int_(a["axis"]), x.type.rank, "size")]
    except O.OpTypeError as err:
        raise MiraError(str(err), loc) from None


def _grad(el: Elaborator, a, loc):
    y = el.tensor(a["y"])
    wrt, wloc = a["wrt"]
    many = isinstance(wrt, (list, tuple))
    xs = [el.tensor((w, wloc)) for w in (wrt if many else [wrt])]
    try:
        gs = autodiff.gradients(el.graph, y, xs, loc)
    except autodiff.GradError as err:
        raise MiraError(str(err), loc) from None
    return tuple(gs) if many else gs[0]


def _where(el: Elaborator, a, loc):
    c, x, y = (a[k][0] for k in ("cond", "a", "b"))
    if not isinstance(c, ir.Value):
        raise MiraError("where() needs a tensor condition; use if for compile-time choices", a["cond"][1])
    vals = [v for v in (x, y) if isinstance(v, ir.Value)]
    dt = vals[0].type.dtype if vals else "f32"
    return el.op("where", [c, el.as_tensor(x, a["a"][1], dt), el.as_tensor(y, a["b"][1], dt)], {}, loc)


def _slice(el: Elaborator, a, loc):
    return el.op("slice", [el.tensor(a["x"])], {"begin": tuple(el.ints(a["begin"])), "size": tuple(el.ints(a["size"]))},
                 loc)


def _pad(el: Elaborator, a, loc):
    x = el.tensor(a["x"])
    raw, ploc = a["pads"]
    if not isinstance(raw, list) or not all(isinstance(p, list) and len(p) == 2 for p in raw):
        raise MiraError("pad() takes pads like [[1, 1], [0, 2]]: one [before, after] pair per axis", ploc)
    pads = tuple((el.ct_int(p[0], ploc), el.ct_int(p[1], ploc)) for p in raw)
    return el.op("pad", [x], {"pads": pads}, loc)


def _dtype_arg(el: Elaborator, arg) -> str:
    dt, dloc = arg
    if not isinstance(dt, DTypeValue):
        raise MiraError(f"expected a dtype like f32 or i32, got {describe(dt)}", dloc)
    return str(dt)


def _filled(value):
    def h(el: Elaborator, a, loc):
        shape = el.ints(a["shape"]) if isinstance(a["shape"][0], list) else [el.int_(a["shape"])]
        v = value if value is not None else el.float_(a["value"])
        dt = _dtype_arg(el, a["dtype"])
        return el.graph.const(np.full(shape, v, dtype=NP_DTYPES[dt]), loc)
    return h


def _arange(el: Elaborator, a, loc):
    start = el.int_(a["start"])
    stop = None if a["stop"][0] is None else el.int_(a["stop"])
    lo, hi = (0, start) if stop is None else (start, stop)
    return el.graph.const(np.arange(lo, hi, dtype=NP_DTYPES[_dtype_arg(el, a["dtype"])]), loc)


def _gather(el: Elaborator, a, loc):
    return el.op("gather", [el.tensor(a["x"]), el.tensor(a["indices"])], {"axis": el.int_(a["axis"])}, loc)


def _scatter_add(el: Elaborator, a, loc):
    return el.op("scatter_add", [el.tensor(a["base"]), el.tensor(a["indices"]), el.tensor(a["updates"])],
                 {"axis": el.int_(a["axis"])}, loc)


def _arg(kind):
    return ([("x", REQUIRED), ("axis", -1), ("keepdims", False)],
            lambda el, a, loc: el.op(kind, [el.tensor(a["x"])], {"axis": el.int_(a["axis"]),
                                                                 "keepdims": el.bool_(a["keepdims"])}, loc))


def _start(el: Elaborator, arg) -> ir.Value:
    v, loc = arg
    if isinstance(v, ir.Value):
        return el.local(v)
    if not isinstance(v, list):
        raise MiraError("a start position is a list like [t, 0] or an i32 tensor", loc)
    return el.start_vector([el.local(p) if isinstance(p, ir.Value) else el.ct_int(p, loc) for p in v], loc)


def _dynamic_slice(el: Elaborator, a, loc):
    return el.op("dynamic_slice", [el.tensor(a["x"]), _start(el, a["start"])], {"size": tuple(el.ints(a["size"]))},
                 loc)


def _dynamic_update_slice(el: Elaborator, a, loc):
    x = el.tensor(a["x"])
    upd = el.as_tensor(a["update"][0], a["update"][1], dtype=x.type.dtype)
    return el.op("dynamic_update_slice", [x, upd, _start(el, a["start"])], {}, loc)


def _broadcast(el: Elaborator, a, loc):
    return el.op("broadcast", [el.tensor(a["x"])], {"shape": tuple(el.ints(a["shape"]))}, loc)


BUILTINS: dict[str, tuple[list[tuple[str, object]], object]] = {
    **{k: _unary(k) for k in ["relu", "gelu", "sigmoid", "tanh", "exp", "log", "sqrt", "abs", "sign", "erf"]},
    "zeros": ([("shape", REQUIRED), ("dtype", DTypeValue("f32"))], _filled(0)),
    "ones": ([("shape", REQUIRED), ("dtype", DTypeValue("f32"))], _filled(1)),
    "full": ([("shape", REQUIRED), ("value", REQUIRED), ("dtype", DTypeValue("f32"))], _filled(None)),
    "arange": ([("start", REQUIRED), ("stop", None), ("dtype", DTypeValue("i32"))], _arange),
    "gather": ([("x", REQUIRED), ("indices", REQUIRED), ("axis", 0)], _gather),
    "scatter_add": ([("base", REQUIRED), ("indices", REQUIRED), ("updates", REQUIRED), ("axis", 0)], _scatter_add),
    "argmax": _arg("argmax"),
    "argmin": _arg("argmin"),
    "dynamic_slice": ([("x", REQUIRED), ("start", REQUIRED), ("size", REQUIRED)], _dynamic_slice),
    "dynamic_update_slice": ([("x", REQUIRED), ("update", REQUIRED), ("start", REQUIRED)], _dynamic_update_slice),
    "grad": ([("y", REQUIRED), ("wrt", REQUIRED)], _grad),
    "where": ([("cond", REQUIRED), ("a", REQUIRED), ("b", REQUIRED)], _where),
    "slice": ([("x", REQUIRED), ("begin", REQUIRED), ("size", REQUIRED)], _slice),
    "pad": ([("x", REQUIRED), ("pads", REQUIRED)], _pad),
    "broadcast": ([("x", REQUIRED), ("shape", REQUIRED)], _broadcast),
    "maximum": _binary_fn("maximum"),
    "minimum": _binary_fn("minimum"),
    "matmul": ([("a", REQUIRED), ("b", REQUIRED)],
               lambda el, a, loc: el.op("matmul", [el.tensor(a["a"]), el.tensor(a["b"])], {}, loc)),
    "softmax": ([("x", REQUIRED), ("axis", -1)],
                lambda el, a, loc: el.op("softmax", [el.tensor(a["x"])], {"axis": el.int_(a["axis"])}, loc)),
    "sum": _reduce("reduce_sum"),
    "mean": _reduce("reduce_mean"),
    "max": _reduce("reduce_max"),
    "transpose": ([("x", REQUIRED), ("perm", None)], _transpose),
    "reshape": ([("x", REQUIRED), ("shape", REQUIRED)], _reshape),
    "flatten": ([("x", REQUIRED), ("axis", 1)], _flatten),
    "conv2d": ([("x", REQUIRED), ("w", REQUIRED), ("stride", 1), ("padding", 0)], _conv2d),
    "maxpool2d": ([("x", REQUIRED), ("size", 2), ("stride", None)], _maxpool),
    "layernorm": ([("x", REQUIRED), ("gamma", REQUIRED), ("beta", REQUIRED), ("eps", 1e-5)], _layernorm),
    "concat": ([("xs", REQUIRED), ("axis", 0)], _concat),
    "cast": ([("x", REQUIRED), ("dtype", REQUIRED)], _cast),
    "size": ([("x", REQUIRED), ("axis", REQUIRED)], _size),
    "sort": _axis_op("sort"),
    "cumprod": _axis_op("cumprod"),
}


def random_weight(rng: np.random.Generator, ty: TensorType) -> np.ndarray:
    """Deterministic, sensibly-scaled random weights (so activations stay O(1))."""
    if not ty.is_float:
        return rng.integers(0, 8, ty.shape).astype(ty.np_dtype)
    if ty.rank == 0:
        return np.array(rng.standard_normal(), dtype=ty.np_dtype)
    if ty.rank == 1:
        return (1.0 + 0.1 * rng.standard_normal(ty.shape)).astype(ty.np_dtype)
    fan_in = int(np.prod(ty.shape[1:])) if ty.rank == 4 else ty.shape[-2]
    return (rng.standard_normal(ty.shape) / np.sqrt(fan_in)).astype(ty.np_dtype)


def elaborate(module: S.Module, entry: str = "main", weights: Optional[dict[str, np.ndarray]] = None,
              dims: Optional[dict[str, int]] = None, seed: Optional[int] = None) -> ir.Graph:
    g = Elaborator(module, entry).entry(entry, weights or {}, dims or {}, seed)
    g.verify()
    return g
