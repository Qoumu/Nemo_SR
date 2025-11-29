import argparse
from pathlib import Path

import yaml
from nemo.collections.asr.models import EncDecSpeakerLabelModel

from model_optim import (
    cast_model_precision,
    drop_classifier_head,
    prune_encoder_layers,
    quantize_model,
    trim_encoder_blocks,
)


def load_config(config_path: str | None) -> dict:
    base = {
        "model": {
            "pretrained_name": "titanet_large",
            "device": "cpu",
            "precision": "float32",
            "encoder": {
                "enabled": True,
                "prune_amount": 0.0,
                "method": "l1_unstructured",
                "keep_blocks": None,
                "drop_blocks": None,
                "keep_first_n": None,
                "keep_last_n": None,
            },
            "decoder": {"keep_classifier": True, "target_num_classes": 2},
            "quantization": {"enabled": False},
        }
    }
    if not config_path:
        return base
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return base
    with cfg_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return merge_dicts(base, loaded)


def merge_dicts(base: dict, override: dict) -> dict:
    if not isinstance(base, dict):
        return override
    merged = dict(base)
    for key, value in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def optimize_and_export(config: dict, output_path: Path, model_name: str | None = None, device: str | None = None, prune: bool | None = None, quantize: bool | None = None) -> Path:
    model_cfg = config.get("model", {})

    pretrained_name = model_name or model_cfg.get("pretrained_name", "titanet_large")
    device = device or model_cfg.get("device", "cpu")
    precision = model_cfg.get("precision", "float32")

    model = EncDecSpeakerLabelModel.from_pretrained(model_name=pretrained_name)

    encoder_cfg = dict(model_cfg.get("encoder", {}))
    quant_cfg = dict(model_cfg.get("quantization", {}))
    decoder_cfg = dict(model_cfg.get("decoder", {}))

    if prune is not None:
        encoder_cfg["enabled"] = prune
    if quantize is not None:
        quant_cfg["enabled"] = quantize

    # Depth trimming first to reduce compute/parameters before masking.
    kept_blocks = trim_encoder_blocks(model.encoder, encoder_cfg)
    try:
        if kept_blocks:
            jasper_cfg = list(model.cfg.encoder.jasper)
            model.cfg.encoder.jasper = [jasper_cfg[i] for i in kept_blocks if i < len(jasper_cfg)]
    except Exception:
        # If config format shifts across NeMo versions, skip config rewrite.
        pass

    drop_classifier_head(model, decoder_cfg)
    try:
        if not decoder_cfg.get("keep_classifier", True):
            if "decoder" in model.cfg and "num_classes" in model.cfg.decoder:
                model.cfg.decoder.num_classes = decoder_cfg.get("target_num_classes", 2)
    except Exception:
        pass

    prune_encoder_layers(model.encoder, encoder_cfg)
    model = quantize_model(model, quant_cfg, device=device)
    if not quant_cfg.get("enabled", False):
        model = cast_model_precision(model, precision)

    model.eval()
    model.to(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune/quantize TitaNet encoder-decoder and export to .nemo")
    parser.add_argument("--config", default="configs/lightweight_titanet.yaml", help="Path to optimization config file")
    parser.add_argument("--output", required=True, help="Destination .nemo file")
    parser.add_argument("--device", default=None, help="Target device for optimization (defaults to config value)")
    parser.add_argument("--model-name", default=None, help="Optional pretrained model name override")
    parser.add_argument("--no-prune", action="store_true", help="Disable pruning regardless of config")
    parser.add_argument("--no-quantize", action="store_true", help="Disable quantization regardless of config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_path = Path(args.output)
    prune_override = False if args.no_prune else None
    quant_override = False if args.no_quantize else None
    optimize_and_export(
        config=cfg,
        output_path=output_path,
        model_name=args.model_name,
        device=args.device,
        prune=prune_override,
        quantize=quant_override,
    )
    print(f"Optimized model saved to {output_path}")


if __name__ == "__main__":
    main()
