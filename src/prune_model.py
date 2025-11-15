import argparse
from pathlib import Path

import yaml
from nemo.collections.asr.models import EncDecSpeakerLabelModel

from model_optim import prune_encoder_layers, quantize_model


def load_config(config_path: str | None) -> dict:
    base = {
        "model": {
            "pretrained_name": "titanet_large",
            "device": "cpu",
            "precision": "float32",
            "encoder": {"enabled": True, "prune_amount": 0.0},
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

    model = EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)

    encoder_cfg = dict(model_cfg.get("encoder", {}))
    quant_cfg = dict(model_cfg.get("quantization", {}))

    if prune is not None:
        encoder_cfg["enabled"] = prune
    if quantize is not None:
        quant_cfg["enabled"] = quantize

    prune_encoder_layers(model.encoder, encoder_cfg)
    model = quantize_model(model, quant_cfg, device=device)

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
