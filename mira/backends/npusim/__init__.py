"""The MNPU-1 simulator target."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ... import ir
from .codegen import CodeGen, check, format_program
from .machine import Config, Machine, Stats, format_timeline


class SimExecutable:
    def __init__(self, g: ir.Graph, cfg: Config, double_buffer: bool):
        self.g = g
        self.cfg = cfg
        self.compiled = CodeGen(cfg, double_buffer).compile(g)
        self.machine = Machine(cfg)
        self.last_stats: Optional[Stats] = None
        self.last_trace: list = []

    def run(self, feeds: dict[ir.Value, np.ndarray]) -> dict[ir.Value, np.ndarray]:
        c, m = self.compiled, self.machine
        m.dram = {name: np.zeros(n, dtype=np.float16) for name, n in c.sizes.items()}
        for name, data in c.consts.items():
            m.dram[name] = data.copy()
        for v, arr in feeds.items():
            m.dram[c.storage[v][0]] = np.ascontiguousarray(arr, dtype=np.float16).ravel().copy()
        self.last_trace = []
        self.last_stats = m.run(c.program, trace=self.last_trace)
        out = {}
        for v in self.g.outputs:
            name, _ = c.storage[v]
            out[v] = m.dram[name].reshape(v.type.shape).copy()
        return out

    def report(self) -> str:
        head = f"mnpu-sim: {len(self.compiled.program):,} instructions"
        return head + ("\n" + self.last_stats.summary(self.cfg) if self.last_stats else "")

    def assembly(self, limit: Optional[int] = None) -> str:
        return format_program(self.compiled, limit)

    def timeline(self, limit: int = 40) -> str:
        return format_timeline(self.last_trace, limit=limit)


class SimTarget:
    name = "npu-sim"

    def __init__(self, cfg: Config = Config(), double_buffer: bool = True):
        self.cfg = cfg
        self.double_buffer = double_buffer

    def check(self, op: ir.Op) -> Optional[str]:
        return check(op, self.cfg)

    def compile(self, g: ir.Graph) -> SimExecutable:
        return SimExecutable(g, self.cfg, self.double_buffer)
