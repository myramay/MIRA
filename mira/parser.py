"""Recursive-descent parser: tokens -> AST.

Grammar (EBNF):

    module   := { NEWLINE | fn }
    fn       := "fn" IDENT "(" [ param { "," param } ] ")" "->" ( type | "(" type { "," type } ")" ) block
    param    := [ "const" ] IDENT ":" type  |  IDENT ":" "int"      (compile-time integer)
    type     := DTYPE "[" [ dim { "," dim } ] "]"
    dim      := expr            (an INT, a shape symbol IDENT, or arithmetic on them like H / 2)
    block    := "{" { NEWLINE | stmt } "}"
    stmt     := "let" names [ ":" type ] "=" expr
              | names "=" expr
              | "for" IDENT "in" expr ".." expr block
              | "if" expr block [ "else" ( block | if-stmt ) ]
              | "while" expr block
              | "return" expr { "," expr }
    names    := IDENT { "," IDENT } | "(" IDENT { "," IDENT } ")"
    expr     := sum [ ("<" | "<=" | ">" | ">=" | "==" | "!=") sum ]
    sum      := term { ("+" | "-") term }
    term     := unary { ("*" | "/" | "@") unary }
    unary    := "-" unary | power
    power    := primary [ "**" unary ]            (right-associative, like Python)
    primary  := INT | FLOAT | "true" | "false" | DTYPE | IDENT [ "(" args ")" ]
              | "(" expr ")" | "(" expr "," [ expr { "," expr } ] ")" | "[" [ expr { "," expr } ] "]"
    args     := [ arg { "," arg } ]
    arg      := [ IDENT "=" ] expr
"""
from __future__ import annotations

from . import syntax as S
from .errors import MiraError
from .lexer import Token, tokenize


class Parser:
    def __init__(self, tokens: list[Token]):
        self.toks = tokens
        self.pos = 0

    # ----- token helpers -----

    @property
    def tok(self) -> Token:
        return self.toks[self.pos]

    def peek(self, offset: int = 1) -> Token:
        return self.toks[min(self.pos + offset, len(self.toks) - 1)]

    def at(self, kind: str, text: str | None = None) -> bool:
        return self.tok.kind == kind and (text is None or self.tok.text == text)

    def accept(self, kind: str, text: str | None = None) -> Token | None:
        if self.at(kind, text):
            t = self.tok
            self.pos += 1
            return t
        return None

    def expect(self, kind: str, text: str | None = None, what: str | None = None) -> Token:
        t = self.accept(kind, text)
        if t is None:
            wanted = what or (repr(text) if text else kind)
            got = {"newline": "end of line", "eof": "end of file"}.get(self.tok.kind, repr(self.tok.text))
            raise MiraError(f"expected {wanted}, found {got}", self.tok.loc)
        return t

    def skip_newlines(self) -> None:
        while self.accept("newline"):
            pass

    # ----- declarations -----

    def module(self) -> S.Module:
        m = S.Module()
        self.skip_newlines()
        while not self.at("eof"):
            fn = self.fn()
            if fn.name in m.functions:
                raise MiraError(f"function '{fn.name}' is defined twice", fn.loc)
            m.functions[fn.name] = fn
            self.skip_newlines()
        return m

    def fn(self) -> S.FnDecl:
        start = self.expect("kw", "fn", "'fn'")
        name = self.expect("ident", what="function name").text
        self.expect("op", "(")
        params: list[S.Param] = []
        if not self.at("op", ")"):
            while True:
                is_const = self.accept("kw", "const") is not None
                ptok = self.expect("ident", what="parameter name")
                self.expect("op", ":")
                if (it := self.accept("ident", "int")) is not None:
                    if is_const:
                        raise MiraError("'const' applies to tensor parameters, not int", it.loc)
                    ptype = S.TypeExpr("int", [], it.loc)
                else:
                    ptype = self.type()
                params.append(S.Param(ptok.text, ptype, is_const, ptok.loc))
                if not self.accept("op", ","):
                    break
        self.expect("op", ")")
        self.skip_newlines()              # allow the "-> ..." on its own line
        self.expect("op", "->", "'->' and a return type")
        if self.accept("op", "("):          # tuple return: -> (f32[..], f32[..])
            rets = [self.type()]
            while self.accept("op", ","):
                rets.append(self.type())
            self.expect("op", ")")
        else:
            rets = [self.type()]
        self.skip_newlines()
        body = self.block()
        return S.FnDecl(name, params, rets, body, start.loc)

    def type(self) -> S.TypeExpr:
        t = self.expect("dtype", what="a tensor type like f16[8, 128]")
        self.expect("op", "[")
        dims: list[int | str | S.Expr] = []
        if not self.at("op", "]"):
            while True:
                d = self.expr()
                if isinstance(d, S.Num) and isinstance(d.value, int):
                    dims.append(d.value)
                elif isinstance(d, S.Name):
                    dims.append(d.ident)
                else:
                    dims.append(d)   # computed dimension, e.g. H / 2
                if not self.accept("op", ","):
                    break
        self.expect("op", "]")
        return S.TypeExpr(t.text, dims, t.loc)

    # ----- statements -----

    def block(self) -> list[S.Stmt]:
        self.expect("op", "{")
        body: list[S.Stmt] = []
        self.skip_newlines()
        while not self.at("op", "}"):
            if self.at("eof"):
                raise MiraError("unclosed '{'", self.tok.loc)
            body.append(self.stmt())
            if not self.at("op", "}"):
                self.expect("newline", what="end of line after statement")
            self.skip_newlines()
        self.expect("op", "}")
        return body

    def stmt(self) -> S.Stmt:
        t = self.tok
        if self.accept("kw", "let"):
            names = self.name_list("variable name")
            ty = None
            if self.accept("op", ":"):
                if len(names) > 1:
                    raise MiraError("type annotations aren't allowed when destructuring", self.tok.loc)
                ty = self.type()
            self.expect("op", "=")
            return S.Let(t.loc, names, ty, self.expr())
        if self.accept("kw", "return"):
            value = self.expr()
            if self.at("op", ","):                 # return a, b
                items = [value]
                while self.accept("op", ","):
                    items.append(self.expr())
                value = S.TupleLit(value.loc, items)
            return S.Return(t.loc, value)
        if self.accept("kw", "for"):
            var = self.expect("ident", what="loop variable").text
            self.expect("kw", "in", "'in'")
            start = self.expr()
            self.expect("op", "..", "'..'")
            stop = self.expr()
            return S.For(t.loc, var, start, stop, self.block())
        if self.accept("kw", "if"):
            return self.if_rest(t)
        if self.accept("kw", "while"):
            cond = self.expr()
            return S.While(t.loc, cond, self.block())
        if self.at("ident") and self.peek().kind == "op" and self.peek().text in ("=", ","):
            names = self.name_list("variable name")
            self.expect("op", "=")
            return S.Assign(t.loc, names, self.expr())
        raise MiraError(f"expected a statement (let, return, for, if, while, or assignment), found {t.text!r}",
                        t.loc)

    def if_rest(self, t) -> S.If:
        cond = self.expr()
        then = self.block()
        orelse: list[S.Stmt] = []
        if self.accept("kw", "else"):
            if (t2 := self.accept("kw", "if")) is not None:     # else if ...
                orelse = [self.if_rest(t2)]
            else:
                orelse = self.block()
        return S.If(t.loc, cond, then, orelse)

    def name_list(self, what: str) -> list[str]:
        """IDENT {, IDENT}  or  ( IDENT {, IDENT} )"""
        paren = self.accept("op", "(") is not None
        names = [self.expect("ident", what=what).text]
        while self.accept("op", ","):
            names.append(self.expect("ident", what=what).text)
        if paren:
            self.expect("op", ")")
        if len(set(names)) != len(names):
            raise MiraError("the same name appears twice", self.tok.loc)
        return names

    # ----- expressions -----

    def expr(self) -> S.Expr:
        lhs = self.sum()
        if self.tok.kind == "op" and self.tok.text in S.COMPARE_OPS:
            op = self.tok
            self.pos += 1
            lhs = S.Binary(op.loc, op.text, lhs, self.sum())
            if self.tok.kind == "op" and self.tok.text in S.COMPARE_OPS:
                raise MiraError("comparisons can't be chained; use parentheses", self.tok.loc)
        return lhs

    def sum(self) -> S.Expr:
        lhs = self.term()
        while self.at("op", "+") or self.at("op", "-"):
            op = self.tok
            self.pos += 1
            lhs = S.Binary(op.loc, op.text, lhs, self.term())
        return lhs

    def term(self) -> S.Expr:
        lhs = self.unary()
        while self.at("op", "*") or self.at("op", "/") or self.at("op", "@"):
            op = self.tok
            self.pos += 1
            lhs = S.Binary(op.loc, op.text, lhs, self.unary())
        return lhs

    def unary(self) -> S.Expr:
        if (t := self.accept("op", "-")) is not None:
            return S.Unary(t.loc, "-", self.unary())
        return self.power()

    def power(self) -> S.Expr:
        base = self.primary()
        if (t := self.accept("op", "**")) is not None:
            return S.Binary(t.loc, "**", base, self.unary())
        return base

    def primary(self) -> S.Expr:
        t = self.tok
        if self.accept("int"):
            return S.Num(t.loc, int(t.text))
        if self.accept("float"):
            return S.Num(t.loc, float(t.text))
        if self.accept("kw", "true"):
            return S.Bool(t.loc, True)
        if self.accept("kw", "false"):
            return S.Bool(t.loc, False)
        if self.accept("dtype"):
            return S.DType(t.loc, t.text)
        if self.accept("ident"):
            if self.accept("op", "("):
                return S.Call(t.loc, t.text, self.args())
            return S.Name(t.loc, t.text)
        if self.accept("op", "("):
            e = self.expr()
            if self.accept("op", ","):             # tuple literal (a, b)
                items = [e]
                if not self.at("op", ")"):
                    while True:
                        items.append(self.expr())
                        if not self.accept("op", ","):
                            break
                self.expect("op", ")")
                return S.TupleLit(t.loc, items)
            self.expect("op", ")")
            return e
        if self.accept("op", "["):
            items: list[S.Expr] = []
            if not self.at("op", "]"):
                while True:
                    items.append(self.expr())
                    if not self.accept("op", ","):
                        break
            self.expect("op", "]")
            return S.ListLit(t.loc, items)
        got = "end of line" if t.kind == "newline" else repr(t.text)
        raise MiraError(f"expected an expression, found {got}", t.loc)

    def args(self) -> list[S.Arg]:
        args: list[S.Arg] = []
        seen_kw = False
        if not self.at("op", ")"):
            while True:
                loc = self.tok.loc
                if self.at("ident") and self.peek().kind == "op" and self.peek().text == "=":
                    name = self.expect("ident").text
                    self.expect("op", "=")
                    args.append(S.Arg(name, self.expr(), loc))
                    seen_kw = True
                else:
                    if seen_kw:
                        raise MiraError("positional argument after keyword argument", loc)
                    args.append(S.Arg(None, self.expr(), loc))
                if not self.accept("op", ","):
                    break
        self.expect("op", ")")
        return args


def parse(source: str, filename: str = "<input>") -> S.Module:
    try:
        return Parser(tokenize(source, filename)).module()
    except MiraError as e:
        e.source = source
        raise
