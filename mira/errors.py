"""Source locations and compiler errors.

Every token, AST node and IR op carries a `Loc` so that any phase of the
compiler can point the user at the exact spot in their source file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Loc:
    line: int
    col: int
    file: str = "<input>"

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


class MiraError(Exception):
    """A user-facing compile error, rendered with the offending source line."""

    def __init__(self, message: str, loc: Optional[Loc] = None, notes: Optional[list[str]] = None):
        super().__init__(message)
        self.message = message
        self.loc = loc
        self.notes = notes or []
        self.source: Optional[str] = None  # attached by the driver for pretty printing

    def render(self, source: Optional[str] = None) -> str:
        source = source if source is not None else self.source
        head = f"{self.loc}: error: {self.message}" if self.loc else f"error: {self.message}"
        lines = [head]
        if self.loc and source:
            src_lines = source.splitlines()
            if 1 <= self.loc.line <= len(src_lines):
                text = src_lines[self.loc.line - 1]
                lines.append(f"  {self.loc.line:4d} | {text}")
                lines.append("       | " + " " * (self.loc.col - 1) + "^")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()
