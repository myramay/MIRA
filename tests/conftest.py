"""Which optional targets this machine can run."""
import importlib.util
import os
import platform
import shutil
import subprocess
import tempfile

# compiled-model cache for this test session only (don't fill the user's ~/.cache/mira)
os.environ.setdefault("MIRA_CACHE_DIR", tempfile.mkdtemp(prefix="mira-test-cache-"))

HAS_COREML = importlib.util.find_spec("coremltools") is not None and platform.system() == "Darwin"
HAS_IREE = importlib.util.find_spec("iree") is not None and importlib.util.find_spec("iree.compiler") is not None


def _has_metal_toolchain() -> bool:
    if platform.system() != "Darwin" or not shutil.which("xcrun"):
        return False
    return subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metallib"], capture_output=True).returncode == 0


HAS_METAL = HAS_IREE and _has_metal_toolchain()

# accelerator targets to test against the CPU reference
ACCEL = ["npu-sim"] + (["coreml"] if HAS_COREML else []) + (["mlir"] if HAS_IREE else [])

# example programs with a `main` entry point (gpt.mira has its own entries and its own tests)
import glob as _glob

MAIN_EXAMPLES = sorted(p for p in _glob.glob("examples/*.mira") if "fn main(" in open(p).read())
