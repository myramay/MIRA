"""Backend registry."""
from __future__ import annotations


def get_target(name: str, **options):
    if name == "cpu":
        from .cpu import CPUTarget
        return CPUTarget()
    if name in ("npu-sim", "sim"):
        from .npusim import SimTarget
        return SimTarget(double_buffer=options.get("double_buffer", True))
    if name in ("coreml", "ane"):
        from .coreml import CoreMLTarget
        return CoreMLTarget(units=options.get("units", "ne"))
    raise ValueError(f"unknown target '{name}' (choose cpu, npu-sim, coreml)")


TARGETS = ["cpu", "npu-sim", "coreml"]
