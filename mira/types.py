"""Tensor types used by the IR. After elaboration every shape is concrete."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NP_DTYPES = {"f16": np.float16, "f32": np.float32}
BYTES = {"f16": 2, "f32": 4}


@dataclass(frozen=True)
class TensorType:
    dtype: str                 # "f16" | "f32"
    shape: tuple[int, ...]

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.numel * BYTES[self.dtype]

    @property
    def np_dtype(self):
        return NP_DTYPES[self.dtype]

    def with_dtype(self, dtype: str) -> "TensorType":
        return TensorType(dtype, self.shape)

    def __str__(self) -> str:
        return f"{self.dtype}[{', '.join(map(str, self.shape))}]"


def type_of_array(a: np.ndarray) -> TensorType:
    for name, dt in NP_DTYPES.items():
        if a.dtype == dt:
            return TensorType(name, tuple(a.shape))
    raise TypeError(f"unsupported array dtype {a.dtype}")
