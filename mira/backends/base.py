"""The interface every backend implements."""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from .. import ir

HEAVY_KINDS = {"matmul", "conv2d", "softmax", "layernorm", "maxpool2d"}


class Executable(Protocol):
    def run(self, feeds: dict[ir.Value, np.ndarray]) -> dict[ir.Value, np.ndarray]: ...

    def report(self) -> str: ...


class Target(Protocol):
    name: str

    def check(self, op: ir.Op) -> Optional[str]:
        """None if this target can run `op`; otherwise the reason it can't."""

    def compile(self, g: ir.Graph) -> Executable:
        """Compile a subgraph whose inputs are fed by `run` and whose outputs are returned."""
