"""Lexer: source text -> list of tokens.

Newlines are significant (they end statements), except inside (...) and [...],
so long argument lists can be wrapped across lines.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import Loc, MiraError

KEYWORDS = {"fn", "let", "return", "for", "in", "const", "true", "false", "if", "else", "while"}
DTYPES = {"f16", "f32"}

# Longest operators first so "->" wins over "-" and ".." over ".".
OPERATORS = ["->", "..", "**", "<=", ">=", "==", "!=", "<", ">",
             "+", "-", "*", "/", "@", "=", ":", ",", "(", ")", "[", "]", "{", "}"]


@dataclass(frozen=True)
class Token:
    kind: str   # "ident", "int", "float", "dtype", "kw", "op", "newline", "eof"
    text: str
    loc: Loc

    def __repr__(self) -> str:
        return f"{self.kind}:{self.text!r}@{self.loc.line}:{self.loc.col}"


def tokenize(source: str, filename: str = "<input>") -> list[Token]:
    tokens: list[Token] = []
    i, line, col = 0, 1, 1
    depth = 0  # nesting of () and []; newlines inside them are ignored

    def loc() -> Loc:
        return Loc(line, col, filename)

    while i < len(source):
        c = source[i]

        if c == "#":  # comment to end of line
            while i < len(source) and source[i] != "\n":
                i += 1
                col += 1
            continue

        if c == "\n":
            if depth == 0 and tokens and tokens[-1].kind != "newline":
                tokens.append(Token("newline", "\\n", loc()))
            i += 1
            line += 1
            col = 1
            continue

        if c in " \t\r":
            i += 1
            col += 1
            continue

        start = loc()

        if c.isdigit() or (c == "." and i + 1 < len(source) and source[i + 1].isdigit()):
            j = i
            is_float = False
            while j < len(source) and source[j].isdigit():
                j += 1
            # A "." starts a fraction only if it isn't the range operator "..".
            if j < len(source) and source[j] == "." and source[j:j + 2] != "..":
                is_float = True
                j += 1
                while j < len(source) and source[j].isdigit():
                    j += 1
            if j < len(source) and source[j] in "eE":
                k = j + 1
                if k < len(source) and source[k] in "+-":
                    k += 1
                if k < len(source) and source[k].isdigit():
                    is_float = True
                    j = k
                    while j < len(source) and source[j].isdigit():
                        j += 1
            text = source[i:j]
            tokens.append(Token("float" if is_float else "int", text, start))
            col += j - i
            i = j
            continue

        if c.isalpha() or c == "_":
            j = i
            while j < len(source) and (source[j].isalnum() or source[j] == "_"):
                j += 1
            text = source[i:j]
            kind = "kw" if text in KEYWORDS else "dtype" if text in DTYPES else "ident"
            tokens.append(Token(kind, text, start))
            col += j - i
            i = j
            continue

        for op in OPERATORS:
            if source.startswith(op, i):
                if op in "([":
                    depth += 1
                elif op in ")]":
                    depth = max(0, depth - 1)
                tokens.append(Token("op", op, start))
                i += len(op)
                col += len(op)
                break
        else:
            raise MiraError(f"unexpected character {c!r}", start)

    if tokens and tokens[-1].kind != "newline":
        tokens.append(Token("newline", "\\n", loc()))
    tokens.append(Token("eof", "", loc()))
    return tokens
