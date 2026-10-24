"""Time one matrix multiply and place it on CUDA when a GPU is present."""

from __future__ import annotations

import time

import torch


def matmul_benchmark(n: int = 2048, seed: int = 7, device: torch.device | None = None) -> dict[str, object]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    left = torch.randn(n, n, generator=generator).to(device)
    right = torch.randn(n, n, generator=generator).to(device)

    _ = left @ right
    if device.type == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()
    product = left @ right
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "device": str(device),
        "cuda": bool(torch.cuda.is_available()),
        "input_shape": tuple(left.shape),
        "output_shape": tuple(product.shape),
        "elapsed_ms": elapsed_ms,
    }
