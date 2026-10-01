"""CPU backend: executes the IR op by op with the NumPy reference semantics.

This is the "golden model" every other backend is tested against, and the
fallback device for ops an accelerator can't run. It's also the host that
drives control flow: an `if` or `while` op runs here, calling into its
sub-graphs, which were compiled for the target like any other graph.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .. import ir
from .. import ops as O


class CPUExecutable:
    def __init__(self, g: ir.Graph, compile_sub: Optional[Callable[[ir.Graph], object]] = None):
        self.g = g
        self.sub: dict[tuple[ir.Op, str], object] = {}
        for op in g.ops:
            for name, sub in op.subgraphs():
                if compile_sub is None:
                    raise RuntimeError("control flow needs a sub-graph compiler")
                self.sub[(op, name)] = compile_sub(sub)

    def run(self, feeds: dict[ir.Value, np.ndarray]) -> dict[ir.Value, np.ndarray]:
        env = dict(feeds)
        for op in self.g.ops:
            if op.is_control:
                outs = self.control(op, [env[v] for v in op.inputs])
                env.update(zip(op.results, outs))
            else:
                env[op.result] = O.evaluate(op.kind, [env[v] for v in op.inputs], op.attrs)
        return {v: env[v] for v in self.g.outputs}

    def control(self, op: ir.Op, args: list[np.ndarray]) -> list[np.ndarray]:
        if op.kind == "if":
            n = op.attrs["n_then"]
            taken = bool(args[0].ravel()[0])
            branch = "then" if taken else "else"
            return self.sub[(op, branch)].run_list(args[1:1 + n] if taken else args[1 + n:])
        ns, nc = op.attrs["n_state"], op.attrs["n_cond"]
        state, cond_caps, body_caps = args[:ns], args[ns:ns + nc], args[ns + nc:]
        cond, body = self.sub[(op, "cond")], self.sub[(op, "body")]
        for _ in range(op.attrs["max_iters"]):
            if not bool(cond.run_list(state + cond_caps)[0].ravel()[0]):
                return state
            state = body.run_list(state + body_caps)
        raise RuntimeError(f"while loop exceeded {op.attrs['max_iters']} iterations")

    def report(self) -> str:
        n_ctrl = sum(op.is_control for op in self.g.ops)
        extra = f" ({n_ctrl} control-flow op(s))" if n_ctrl else ""
        return f"cpu: {len(self.g.compute_ops())} ops interpreted with NumPy{extra}"


class CPUTarget:
    name = "cpu"

    def check(self, op: ir.Op) -> Optional[str]:
        return None

    def compile(self, g: ir.Graph, compile_sub=None) -> CPUExecutable:
        return CPUExecutable(g, compile_sub)
