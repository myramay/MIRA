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
    are resolved here. `for` loops are fully unrolled.
"""
from __future__ import annotations

import difflib
from typing import Optional, Union

import numpy as np

from . import ir
from . import ops as O
from . import syntax as S
from .errors import Loc, MiraError
from .types import NP_DTYPES, TensorType


class DTypeValue(str):
    """A dtype used as a value, e.g. the `f32` in `cast(x, f32)`."""


CTValue = Union[int, float, bool, list, DTypeValue]   # compile-time values
Val = Union[ir.Value, CTValue]

REQUIRED = object()
MAX_UNROLL = 10_000
MAX_DEPTH = 64

BINOP_KIND = {"+": "add", "-": "sub", "*": "mul", "/": "div", "**": "pow"}


def describe(v: Val) -> str:
    if isinstance(v, ir.Value):
        return f"tensor {v.type}"
    if isinstance(v, DTypeValue):
        return f"dtype {v}"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return f"number {v}"
    if isinstance(v, list):
        return "list"
    return type(v).__name__


class Scope:
    def __init__(self, parent: Optional["Scope"] = None):
        self.vars: dict[str, Val] = {}
        self.parent = parent

    def lookup(self, name: str) -> Optional[Val]:
        s: Optional[Scope] = self
        while s is not None:
            if name in s.vars:
                return s.vars[name]
            s = s.parent
        return None

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


class Elaborator:
    def __init__(self, module: S.Module, graph_name: str):
        self.module = module
        self.graph = ir.Graph(graph_name, [], [], [])
        self.stack: list[str] = []

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
                        raise MiraError(f"weight '{p.name}' has shape {list(arr.shape)} but the parameter is {ty}", p.loc)
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
        self.graph.outputs = [self.as_tensor(out, fn.loc)]
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
                      preset: Optional[dict[str, int]] = None) -> ir.Value:
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
                if isinstance(a, (int, float)) and not isinstance(a, bool):
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
            result = self.block(fn.body, scope, in_loop=False)
            if result is None:
                raise MiraError(f"function '{fn.name}' doesn't return a value", fn.loc)
            ret_ty = self.resolve_type(fn.ret, bindings)
            value, rloc = result
            value = self.as_tensor(value, rloc, dtype=ret_ty.dtype)
            if value.type != ret_ty:
                raise MiraError(f"'{fn.name}' returns {value.type}, but its signature says {ret_ty}", rloc,
                                notes=[f"declared return type: {fn.ret}"])
            return value
        finally:
            self.stack.pop()

    # ================================================================ statements

    def block(self, body: list[S.Stmt], scope: Scope, in_loop: bool) -> Optional[tuple[Val, Loc]]:
        for i, st in enumerate(body):
            if isinstance(st, S.Let):
                if st.name in scope.vars:
                    raise MiraError(f"'{st.name}' is already defined in this scope; use '{st.name} = ...' to reassign",
                                    st.loc)
                v = self.expr(st.value, scope)
                if st.type is not None:
                    want = self.resolve_type(st.type, self.shape_bindings(scope))
                    v = self.as_tensor(v, st.value.loc, dtype=want.dtype)
                    if v.type != want:
                        raise MiraError(f"'{st.name}' is declared {want} but the value is {v.type}", st.value.loc)
                scope.vars[st.name] = v
            elif isinstance(st, S.Assign):
                owner = scope.owner(st.name)
                if owner is None:
                    raise MiraError(f"assignment to undefined variable '{st.name}'; declare it with 'let'", st.loc)
                old = owner.vars[st.name]
                v = self.expr(st.value, scope)
                if isinstance(old, ir.Value):
                    v = self.as_tensor(v, st.value.loc, dtype=old.type.dtype)
                    if v.type != old.type:
                        raise MiraError(f"'{st.name}' has type {old.type}; can't assign a {v.type}", st.value.loc)
                elif isinstance(v, ir.Value):
                    raise MiraError(f"'{st.name}' is a compile-time {describe(old)}; can't assign a tensor", st.loc)
                owner.vars[st.name] = v
            elif isinstance(st, S.For):
                start = self.ct_int(self.expr(st.start, scope), st.start.loc, "loop bound")
                stop = self.ct_int(self.expr(st.stop, scope), st.stop.loc, "loop bound")
                if stop - start > MAX_UNROLL:
                    raise MiraError(f"loop has {stop - start} iterations; at most {MAX_UNROLL} can be unrolled", st.loc)
                for it in range(start, stop):
                    inner = Scope(scope)
                    inner.vars[st.var] = it
                    self.block(st.body, inner, in_loop=True)
            elif isinstance(st, S.Return):
                if in_loop:
                    raise MiraError("'return' inside a loop isn't supported (loops are unrolled at compile time)",
                                    st.loc)
                if i != len(body) - 1:
                    raise MiraError("unreachable code after 'return'", body[i + 1].loc)
                return self.expr(st.value, scope), st.value.loc
        return None

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
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return -v
            raise MiraError(f"can't negate a {describe(v)}", e.loc)
        if isinstance(e, S.Binary):
            return self.binary(e.op, self.expr(e.lhs, scope), self.expr(e.rhs, scope), e.loc)
        if isinstance(e, S.Call):
            return self.call(e, scope)
        raise AssertionError(e)

    def binary(self, op: str, a: Val, b: Val, loc: Loc) -> Val:
        a_num = isinstance(a, (int, float)) and not isinstance(a, bool)
        b_num = isinstance(b, (int, float)) and not isinstance(b, bool)
        if a_num and b_num:
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
        return self.op(BINOP_KIND[op], [a, b], {}, loc)

    def as_tensor(self, v: Val, loc: Loc, dtype: str = "f32") -> ir.Value:
        """Tensors pass through; numbers become scalar constants of `dtype`."""
        if isinstance(v, ir.Value):
            return v
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return self.graph.const(np.array(v, dtype=NP_DTYPES[dtype]), loc)
        raise MiraError(f"expected a tensor, got {describe(v)}", loc)

    def op(self, kind: str, inputs: list[ir.Value], attrs: dict, loc: Loc) -> ir.Value:
        try:
            return self.graph.add(kind, inputs, attrs, loc)
        except O.OpTypeError as err:
            raise MiraError(str(err), loc) from None

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
        return v

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


BUILTINS: dict[str, tuple[list[tuple[str, object]], object]] = {
    **{k: _unary(k) for k in ["relu", "gelu", "sigmoid", "tanh", "exp", "log", "sqrt", "abs"]},
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
