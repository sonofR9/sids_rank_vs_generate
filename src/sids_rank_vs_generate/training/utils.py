from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def move_to_device(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def autocast(device: torch.device, precision: bool | str):
    """Autocast with legacy bool support for the generation trainers."""

    if isinstance(precision, bool):
        precision = "bf16" if precision else "fp32"
    enabled = precision != "fp32" and device.type == "cuda"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def math_sdp_only(device: torch.device):
    """Use the attention backend available on V100, including on newer GPUs."""

    if device.type != "cuda":
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        pass
    else:
        return sdpa_kernel(SDPBackend.MATH)
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
    )
