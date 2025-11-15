from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def prune_encoder_layers(encoder: nn.Module, encoder_cfg: dict | None) -> None:
    """
    Apply unstructured pruning to encoder blocks in-place.
    Only Conv1d layers are touched and they can be filtered by kernel size.
    """
    if not encoder_cfg:
        return
    if not encoder_cfg.get("enabled", True):
        return
    amount = float(encoder_cfg.get("prune_amount", 0.0))
    if amount <= 0.0:
        return
    prune_pointwise_only = encoder_cfg.get("prune_pointwise_only", False)
    min_channels = encoder_cfg.get("min_channels", 0)
    for name, module in encoder.named_modules():
        params_to_prune = []
        if isinstance(module, nn.Conv1d.MaskedConv1d) or isinstance(module, nn.Conv1d):
            params_to_prune.append((module, 'weight'))
        if prune_pointwise_only:
            out_channels = module.out_channels
            if out_channels <= min_channels:
                continue
        
        kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )


def quantize_model(model: nn.Module, quant_cfg: dict | None, device: str | torch.device | None = None) -> nn.Module:
    """
    Dynamically quantize the model if configuration enables it and the device supports it.
    """
    if not quant_cfg or not quant_cfg.get("enabled", False):
        return model
    device_str = str(device).lower() if device is not None else "cpu"
    if device_str != "cpu":
        # PyTorch dynamic quantization is currently CPU-only.
        return model
    dtype_name = quant_cfg.get("dtype", "qint8")
    dtype = getattr(torch, dtype_name, torch.qint8)
    module_types = _resolve_module_types(quant_cfg.get("modules", None))
    return torch.quantization.quantize_dynamic(
        model,
        {m for m in module_types},
        dtype=dtype,
    )


def _resolve_module_types(modules: Iterable[str] | None) -> list[type[nn.Module]]:
    if not modules:
        return [nn.Linear]
    resolved: list[type[nn.Module]] = []
    for class_name in modules:
        cls = getattr(nn, class_name, None)
        if isinstance(cls, type) and issubclass(cls, nn.Module):
            resolved.append(cls)
    return resolved or [nn.Linear]
