"""AST node definitions: the tree the parser builds from tokens."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from .errors import Loc


# ----- types as written in source -----

@dataclass
class TypeExpr:
    dtype: str                             # "f16" | "f32"
    dims: list[Union[int, str, "Expr"]]    # fixed size | shape symbol like "B" | expression like H / 2
    loc: Loc

    @property
    def is_int(self) -> bool:
        """`int` parameters are compile-time integers, not tensors."""
        return self.dtype == "int"

    def __str__(self) -> str:
        if self.is_int:
            return "int"
        dims = (d if isinstance(d, (int, str)) else format_expr(d) for d in self.dims)
        return f"{self.dtype}[{', '.join(map(str, dims))}]"


# ----- expressions -----

@dataclass
class Expr:
    loc: Loc


@dataclass
class Num(Expr):
    value: Union[int, float]


@dataclass
class Bool(Expr):
    value: bool


@dataclass
class DType(Expr):
    name: str


@dataclass
class Name(Expr):
    ident: str


@dataclass
class ListLit(Expr):
    items: list[Expr]


@dataclass
class Unary(Expr):
    op: str          # "-"
    operand: Expr


@dataclass
class Binary(Expr):
    op: str          # "+", "-", "*", "/", "@", "**"
    lhs: Expr
    rhs: Expr


@dataclass
class Arg:
    name: Optional[str]   # None for positional
    value: Expr
    loc: Loc


@dataclass
class Call(Expr):
    callee: str
    args: list[Arg]


# ----- statements -----

@dataclass
class Stmt:
    loc: Loc


@dataclass
class Let(Stmt):
    name: str
    type: Optional[TypeExpr]
    value: Expr


@dataclass
class Assign(Stmt):
    name: str
    value: Expr


@dataclass
class For(Stmt):
    var: str
    start: Expr
    stop: Expr
    body: list[Stmt]


@dataclass
class Return(Stmt):
    value: Expr


# ----- top level -----

@dataclass
class Param:
    name: str
    type: TypeExpr
    is_const: bool
    loc: Loc


@dataclass
class FnDecl:
    name: str
    params: list[Param]
    ret: TypeExpr
    body: list[Stmt]
    loc: Loc


@dataclass
class Module:
    functions: dict[str, FnDecl] = field(default_factory=dict)


# ----- pretty printer (AST -> source) -----

_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "@": 2, "**": 4}


def format_expr(e: Expr, parent_prec: int = 0) -> str:
    if isinstance(e, Num):
        return repr(e.value)
    if isinstance(e, Bool):
        return "true" if e.value else "false"
    if isinstance(e, DType):
        return e.name
    if isinstance(e, Name):
        return e.ident
    if isinstance(e, ListLit):
        return "[" + ", ".join(format_expr(i) for i in e.items) + "]"
    if isinstance(e, Unary):
        s = "-" + format_expr(e.operand, 3)
        return f"({s})" if parent_prec > 3 else s
    if isinstance(e, Binary):
        p = _PREC[e.op]
        # "**" is right-associative; everything else is left-associative.
        lp, rp = (p + 1, p) if e.op == "**" else (p, p + 1)
        s = f"{format_expr(e.lhs, lp)} {e.op} {format_expr(e.rhs, rp)}"
        return f"({s})" if p < parent_prec else s
    if isinstance(e, Call):
        args = ", ".join((f"{a.name}={format_expr(a.value)}" if a.name else format_expr(a.value)) for a in e.args)
        return f"{e.callee}({args})"
    raise TypeError(e)


def format_module(m: Module) -> str:
    out: list[str] = []

    def stmts(body: list[Stmt], indent: str) -> None:
        for s in body:
            if isinstance(s, Let):
                ann = f": {s.type}" if s.type else ""
                out.append(f"{indent}let {s.name}{ann} = {format_expr(s.value)}")
            elif isinstance(s, Assign):
                out.append(f"{indent}{s.name} = {format_expr(s.value)}")
            elif isinstance(s, For):
                out.append(f"{indent}for {s.var} in {format_expr(s.start)}..{format_expr(s.stop)} {{")
                stmts(s.body, indent + "  ")
                out.append(f"{indent}}}")
            elif isinstance(s, Return):
                out.append(f"{indent}return {format_expr(s.value)}")

    for fn in m.functions.values():
        params = ", ".join(f"{'const ' if p.is_const else ''}{p.name}: {p.type}" for p in fn.params)
        out.append(f"fn {fn.name}({params}) -> {fn.ret} {{")
        stmts(fn.body, "  ")
        out.append("}")
        out.append("")
    return "\n".join(out)
