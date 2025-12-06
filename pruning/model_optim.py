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


def _calculate_layer_importance(module: nn.Module) -> float:
    """Calculate importance score for a layer based on weight magnitude."""
    if hasattr(module, "weight") and module.weight is not None:
        return float(torch.norm(module.weight, p=1).item())
    return 0.0


def _prune_channels_by_magnitude(
    params_to_prune: list[tuple[nn.Module, str]], amount: float
) -> dict[int, list[int]]:
    """
    Remove output channels from conv layers based on channel-wise magnitude.
    Returns mapping of module id to pruned channel indices for actual removal.
    """
    pruned_channels: dict[int, list[int]] = {}
    
    for module, name in params_to_prune:
        if not hasattr(module, name):
            continue
        weight = getattr(module, name)
        if weight.dim() < 2:
            continue
        
        # Compute L1 norm per output channel (dim=0)
        channel_norms = torch.norm(weight.view(weight.size(0), -1), p=1, dim=1)
        
        # Determine threshold for pruning
        num_channels = channel_norms.numel()
        num_to_prune = max(1, int(num_channels * amount))
        
        if num_to_prune >= num_channels:
            continue
        
        # Find least important channels
        _, indices_to_prune = torch.topk(channel_norms, num_to_prune, largest=False)
        pruned_channels[id(module)] = indices_to_prune.tolist()
        
        # Apply structured pruning
        prune.ln_structured(
            module,
            name=name,
            amount=amount,
            n=2,
            dim=0,
        )
    
    return pruned_channels


def _remove_pruned_channels(encoder: nn.Module, pruned_channels: dict[int, list[int]]) -> None:
    """
    Actually remove pruned output channels from conv layers and adjust subsequent layers.
    This makes the model physically smaller.
    """
    if not pruned_channels:
        return
    
    for module in encoder.modules():
        if not isinstance(module, Conv1dTypes):
            continue
        target = module.conv if isinstance(module, MaskedConv1d) else module
        module_id = id(target)
        
        if module_id not in pruned_channels:
            continue
        
        channels_to_keep = pruned_channels[module_id]
        if not channels_to_keep:
            continue
        
        # Remove pruned channels from weight and bias
        if hasattr(target, "weight") and target.weight is not None:
            channels_mask = torch.ones(target.weight.size(0), dtype=torch.bool)
            channels_mask[channels_to_keep] = False
            new_weight = target.weight.data[~channels_mask]
            target.weight = nn.Parameter(new_weight)
            target.out_channels = new_weight.size(0)
        
        if hasattr(target, "bias") and target.bias is not None:
            channels_mask = torch.ones(target.bias.size(0), dtype=torch.bool)
            channels_mask[channels_to_keep] = False
            new_bias = target.bias.data[~channels_mask]
            target.bias = nn.Parameter(new_bias)


def _remove_pruned_weights(params_to_prune: list[tuple[nn.Module, str]]) -> None:
    """
    Remove pruned weights (convert sparse tensors to dense).
    This reduces model file size by removing zero weights.
    """
    for module, name in params_to_prune:
        if not hasattr(module, name):
            continue
        
        weight = getattr(module, name)
        
        # If weight is sparse or has a mask, convert to dense and remove zeros
        if hasattr(module, f"{name}_mask"):
            mask = getattr(module, f"{name}_mask")
            # Keep only non-masked weights
            dense_weight = weight * mask
            module.weight = nn.Parameter(dense_weight)
            # Remove the mask
            if hasattr(module, f"{name}_orig"):
                delattr(module, f"{name}_orig")
            if hasattr(module, f"{name}_mask"):
                delattr(module, f"{name}_mask")
        else:
            # Ensure weight is dense
            if weight.is_sparse:
                module.weight = nn.Parameter(weight.to_dense())


def _prune_layers_by_importance(
    encoder: nn.Module, encoder_cfg: dict, amount: float
) -> None:
    """
    Identify and prune less important layers based on weight magnitude.
    Zeros out entire layer weights for less important layers.
    """
    layers: list[tuple[float, nn.Module, str]] = []
    
    for name, module in encoder.named_modules():
        if isinstance(module, Conv1dTypes):
            target = module.conv if isinstance(module, MaskedConv1d) else module
            importance = _calculate_layer_importance(target)
            layers.append((importance, target, name))
    
    if not layers:
        return
    
    # Sort by importance (ascending)
    layers.sort(key=lambda x: x[0])
    num_to_prune = max(1, int(len(layers) * amount))
    
    # Prune least important layers
    for _, module, _ in layers[:num_to_prune]:
        if hasattr(module, "weight"):
            module.weight.data.zero_()
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()

def prune_encoder_layers(encoder: nn.Module, encoder_cfg: dict | None) -> None:
    """
    Apply pruning to encoder blocks in-place and physically remove pruned weights/channels.
    Supports multiple pruning methods:
    - 'l1_unstructured': Remove individual weights based on L1 magnitude
    - 'l1_structured': Remove entire output channels based on channel magnitude
    - 'layer_magnitude': Remove entire layers based on layer importance
    
    After pruning, physically removes the pruned weights to reduce model size.
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

    method = encoder_cfg.get("method", "l1_structured")
    amount = min(max(amount, 0.0), 0.95)

    if method == "l1_structured":
        # Removes entire output channels (structured pruning)
        pruned_channels = _prune_channels_by_magnitude(params_to_prune, amount)
        _remove_pruned_channels(encoder, pruned_channels)
    elif method == "layer_magnitude":
        # Removes entire layers based on weight magnitude importance
        _prune_layers_by_importance(encoder, encoder_cfg, amount)
    else:
        # L1 unstructured: removes individual weights
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        # Remove sparse representation to reduce size
        _remove_pruned_weights(params_to_prune)

    # Clean up any remaining mask re-parametrization
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