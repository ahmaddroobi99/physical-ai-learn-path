"""Show the storage and error of a uniform int8 quantization."""

from __future__ import annotations

import numpy as np


def quantize_int8(weights: np.ndarray, scale: float = 32.0) -> dict[str, object]:
    values = np.asarray(weights, dtype=np.float32)
    quantized = np.clip(np.round(values * scale), -128, 127).astype(np.int8)
    reconstructed = quantized.astype(np.float32) / np.float32(scale)
    return {
        "quantized": quantized,
        "reconstructed": reconstructed,
        "fp32_bytes": int(values.nbytes),
        "int8_bytes": int(quantized.nbytes),
        "storage_ratio": float(values.nbytes / quantized.nbytes),
        "mean_abs_error": float(np.mean(np.abs(values - reconstructed))),
    }
