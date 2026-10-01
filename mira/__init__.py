"""Mira: a small tensor language that compiles to CPUs, a simulated NPU, the Apple Neural Engine, and MLIR."""
from .compiler import CompiledProgram, DynamicProgram, compile_file, compile_source
from .errors import MiraError

__all__ = ["compile_source", "compile_file", "CompiledProgram", "DynamicProgram", "MiraError"]
