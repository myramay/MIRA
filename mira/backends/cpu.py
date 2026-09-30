"""CPU backend: executes the IR op by op with the NumPy reference semantics.

This is the "golden model" every other backend is tested against, and the
fallback device for ops an accelerator can't run.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .. import ir
from .. import ops as O


class CPUExecutable:
    def __init__(self, g: ir.Graph):
        self.g = g

    def run(self, feeds: dict[ir.Value, np.ndarray]) -> dict[ir.Value, np.ndarray]:
        env = dict(feeds)
        for op in self.g.ops:
            env[op.result] = O.evaluate(op.kind, [env[v] for v in op.inputs], op.attrs)
        return {v: env[v] for v in self.g.outputs}

    def report(self) -> str:
        return f"cpu: {len(self.g.compute_ops())} ops interpreted with NumPy"


class CPUTarget:
    name = "cpu"

    def check(self, op: ir.Op) -> Optional[str]:
        return None

    def compile(self, g: ir.Graph) -> CPUExecutable:
        return CPUExecutable(g)
