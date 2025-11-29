from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from nemo.collections.asr.parts.submodules.jasper import MaskedConv1d


Conv1dTypes = (nn.Conv1d, MaskedConv1d)


def _get_conv_attr(module: nn.Module, attr: str, default=None):
    if hasattr(module, attr):
        return getattr(module, attr)
    conv = getattr(module, "conv", None)
    if conv is not None and hasattr(conv, attr):
        return getattr(conv, attr)
    return default


def _is_pointwise_conv(module: nn.Module) -> bool:
    if not isinstance(module, Conv1dTypes):
        return False
    kernel_size = _get_conv_attr(module, "kernel_size", None)
    groups = _get_conv_attr(module, "groups", 1)
    if kernel_size is None:
        return False
    ks = kernel_size if isinstance(kernel_size, Iterable) else (kernel_size,)
    return tuple(ks) == (1,) and groups == 1


def _gather_prunable_params(encoder: nn.Module, encoder_cfg: dict) -> list[tuple[nn.Module, str]]:
    prune_pointwise_only = encoder_cfg.get("prune_pointwise_only", False)
    min_channels = int(encoder_cfg.get("min_channels", 0))

    params_to_prune: list[tuple[nn.Module, str]] = []
    seen: set[int] = set()
    for _, module in encoder.named_modules():
        if isinstance(module, MaskedConv1d):
            target = module.conv
        elif isinstance(module, nn.Conv1d):
            target = module
        else:
            continue
        if id(target) in seen:
            continue
        seen.add(id(target))

        if prune_pointwise_only and not _is_pointwise_conv(target):
            continue
        out_ch = _get_conv_attr(target, "out_channels", 0)
        if out_ch <= min_channels:
            continue
        params_to_prune.append((target, "weight"))
    return params_to_prune


def prune_encoder_layers(encoder: nn.Module, encoder_cfg: dict | None) -> None:
    """
    Apply pruning to encoder blocks in-place.
    Only Conv1d/MaskedConv1d layers are touched. Pruning can be limited to
    pointwise convolutions via config.
    """
    if not encoder_cfg:
        return
    if not encoder_cfg.get("enabled", True):
        return
    amount = float(encoder_cfg.get("prune_amount", 0.0))
    if amount <= 0.0:
        return

    params_to_prune = _gather_prunable_params(encoder, encoder_cfg)
    if not params_to_prune:
        return

    method = encoder_cfg.get("method", "l1_unstructured")
    amount = min(max(amount, 0.0), 0.95)

    if method == "ln_structured":
        # Removes entire output channels (structured), keeps masks removable.
        for module, name in params_to_prune:
            prune.ln_structured(
                module,
                name=name,
                amount=amount,
                n=2,
                dim=0,
            )
    else:
        # Default to global L1 unstructured pruning
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

    # Remove the mask re-parametrization so saved checkpoints don't include it.
    for module, _ in params_to_prune:
        if hasattr(module, "weight_mask"):
            prune.remove(module, "weight")


def trim_encoder_blocks(encoder_module: nn.Module, encoder_cfg: dict | None) -> list[int]:
    """
    Drop entire Jasper blocks to shrink depth and runtime.
    The ConvASREncoder stores blocks in encoder_module.encoder (Sequential).
    """
    if not encoder_cfg:
        return []
    keep_blocks: Sequence[int] | None = encoder_cfg.get("keep_blocks", None)
    drop_blocks: Sequence[int] | None = encoder_cfg.get("drop_blocks", None)
    keep_first_n = encoder_cfg.get("keep_first_n", None)
    keep_last_n = encoder_cfg.get("keep_last_n", None)

    seq: nn.Sequential | None = getattr(encoder_module, "encoder", None)
    if not isinstance(seq, nn.Sequential):
        return []

    total = len(seq)
    if total == 0:
        return []

    if keep_blocks:
        keep = [i for i in keep_blocks if 0 <= i < total]
    elif drop_blocks:
        drop_set = {i for i in drop_blocks if 0 <= i < total}
        keep = [i for i in range(total) if i not in drop_set]
    elif keep_first_n is not None:
        keep = list(range(min(int(keep_first_n), total)))
        if keep_last_n:
            tail = list(range(max(total - int(keep_last_n), 0), total))
            keep = sorted(set(keep + tail))
    elif keep_last_n is not None:
        keep = list(range(max(total - int(keep_last_n), 0), total))
    else:
        return []

    keep = sorted(set(keep))
    if len(keep) == 0 or len(keep) == total:
        return keep

    new_blocks = [seq[i] for i in keep]
    new_seq = nn.Sequential(*new_blocks)
    if hasattr(encoder_module, "encoder"):
        encoder_module.encoder = new_seq
    return keep


def drop_classifier_head(model: nn.Module, decoder_cfg: dict | None) -> None:
    """
    Replace the large speaker-classification head with an identity so only
    embeddings remain. This shrinks checkpoint size and removes unnecessary
    compute for embedding-only inference.
    """
    if not decoder_cfg or decoder_cfg.get("keep_classifier", True):
        return
    decoder = getattr(model, "decoder", None)
    if decoder is None or not hasattr(decoder, "final"):
        return
    new_classes = int(decoder_cfg.get("target_num_classes", 2))
    in_features = getattr(decoder.final, "in_features", None)
    if in_features is None and hasattr(decoder, "emb_layers"):
        last = decoder.emb_layers[-1]
        if isinstance(last, nn.Sequential):
            conv = last[-1]
            in_features = getattr(conv, "out_channels", None)
    in_features = in_features or 192
    decoder.final = nn.Linear(in_features, new_classes, bias=False)


def cast_model_precision(model: nn.Module, precision: str | None) -> nn.Module:
    if precision not in {"float16", "bfloat16"}:
        return model
    target_dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return model.to(dtype=target_dtype)


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
        inplace=True,
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
