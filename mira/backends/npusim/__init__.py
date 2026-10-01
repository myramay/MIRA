"""The MNPU-1 simulator target."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ... import ir
from ..cpu import CPUExecutable
from .codegen import CodeGen, check, format_program
from .machine import Config, Machine, Stats, format_timeline


class SimExecutable:
    def __init__(self, g: ir.Graph, cfg: Config, double_buffer: bool, fast: bool):
        self.g = g
        self.cfg = cfg
        self.fast = fast
        self.double_buffer = double_buffer
        self.compiled = CodeGen(cfg, double_buffer).compile(g)
        self.machine = Machine(cfg, functional=not fast)
        # fast mode: timing only; the numbers come from the reference implementation
        self.reference = CPUExecutable(g) if fast else None
        self.last_stats: Optional[Stats] = None
        self.last_trace: list = []

    def run(self, feeds: dict[ir.Value, np.ndarray]) -> dict[ir.Value, np.ndarray]:
        c, m = self.compiled, self.machine
        m.itemsize = {name: np.dtype(dt).itemsize for name, (_, dt) in c.sizes.items()}
        if not self.fast:
            m.dram = {name: np.zeros(n, dtype=dt) for name, (n, dt) in c.sizes.items()}
            for name, data in c.consts.items():
                m.dram[name] = data.copy()
            for v, arr in feeds.items():
                m.dram[c.storage[v][0]] = np.ascontiguousarray(arr, dtype=np.float16).ravel().copy()
        self.last_trace = []
        self.last_stats = m.run(c.program, trace=self.last_trace)
        if self.fast:
            return self.reference.run(feeds)
        out = {}
        for v in self.g.outputs:
            name, _ = c.storage[v]
            out[v] = m.dram[name].reshape(v.type.shape).astype(np.float16)
        return out

    def report(self) -> str:
        head = f"mnpu-sim: {len(self.compiled.program):,} instructions" + (" (fast: timing only)" if self.fast else "")
        return head + ("\n" + self.last_stats.summary(self.cfg) if self.last_stats else "")

    def assembly(self, limit: Optional[int] = None) -> str:
        return format_program(self.compiled, limit)

    def timeline(self, limit: int = 40) -> str:
        return format_timeline(self.last_trace, limit=limit)


class SimTarget:
    name = "npu-sim"

    def __init__(self, cfg: Config = Config(), double_buffer: bool = True, fast: bool = False):
        self.cfg = cfg
        self.double_buffer = double_buffer
        self.fast = fast

    def check(self, op: ir.Op) -> Optional[str]:
        return check(op, self.cfg)

    def compile(self, g: ir.Graph) -> SimExecutable:
        return SimExecutable(g, self.cfg, self.double_buffer, self.fast)
