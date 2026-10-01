"""Which optional targets this machine can run."""
import importlib.util
import platform
import shutil
import subprocess

HAS_COREML = importlib.util.find_spec("coremltools") is not None and platform.system() == "Darwin"
HAS_IREE = importlib.util.find_spec("iree") is not None and importlib.util.find_spec("iree.compiler") is not None


def _has_metal_toolchain() -> bool:
    if platform.system() != "Darwin" or not shutil.which("xcrun"):
        return False
    return subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metallib"], capture_output=True).returncode == 0


HAS_METAL = HAS_IREE and _has_metal_toolchain()

# accelerator targets to test against the CPU reference
ACCEL = ["npu-sim"] + (["coreml"] if HAS_COREML else []) + (["mlir"] if HAS_IREE else [])
