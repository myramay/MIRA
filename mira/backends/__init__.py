"""Backend registry."""
from __future__ import annotations


def get_target(name: str, **options):
    if name == "cpu":
        from .cpu import CPUTarget
        return CPUTarget()
    if name in ("npu-sim", "sim"):
        from .npusim import SimTarget
        from .npusim.machine import Config
        cfg = options.get("sim_config") or Config()
        if isinstance(cfg, dict):
            cfg = Config.from_overrides(cfg)
        return SimTarget(cfg, double_buffer=options.get("double_buffer", True), fast=options.get("sim_fast", False))
    if name in ("coreml", "ane"):
        from .coreml import CoreMLTarget
        return CoreMLTarget(units=options.get("units", "ne"))
    if name in ("mlir", "iree"):
        from .mlir import MLIRTarget
        return MLIRTarget(backend=options.get("mlir_backend", "cpu"))
    raise ValueError(f"unknown target '{name}' (choose cpu, npu-sim, coreml, mlir)")


TARGETS = ["cpu", "npu-sim", "coreml", "mlir"]
