from __future__ import annotations

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


def autocast(device: torch.device, enabled: bool):
    use_mixed_precision = enabled and device.type == "cuda"
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return torch.amp.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=use_mixed_precision,
    )


def grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    use_float16 = (
        enabled
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    )
    return torch.amp.GradScaler("cuda", enabled=use_float16)
