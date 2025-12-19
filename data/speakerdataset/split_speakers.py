"""
Split a JSON-lines manifest into known and unknown speaker sets.
- Known speakers: 80% of speakers (for enrollment/training)
- Unknown speakers: 20% of speakers (for validation/testing)

Manifest schema per line:
  {"audio_filepath": "...", "offset": 0, "duration": 4.16, "label": "speaker_id"}
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_manifest(json_path: str) -> list[dict]:
    """Load a JSONL manifest (or legacy nested JSON) into a list of dicts."""
    entries = []
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        cfg = None

    # Legacy nested schema
    if isinstance(cfg, dict) and "speakers" in cfg:
        for spk in cfg.get("speakers", []):
            label = spk.get("id")
            for clip in spk.get("clips", []):
                entries.append(
                    {"audio_filepath": clip, "offset": 0, "duration": None, "label": label}
                )
        return entries

    # JSON lines
    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {line_no} of {json_path}: {e}") from e
        entries.append(obj)
    return entries


def save_manifest(path: str, data: list[dict]) -> None:
    """Save a list of manifest rows to JSONL format."""
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def split_speakers(
    input_json: str,
    known_output: str,
    unknown_output: str,
    known_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[int, int]:
    """
    Split speakers into known and unknown sets using a JSONL manifest.
    
    Args:
        input_json: Path to input speaker JSON
        known_output: Path to save known speakers (80%)
        unknown_output: Path to save unknown speakers (20%)
        known_ratio: Proportion of speakers for known set (default 0.8)
        seed: Random seed for reproducibility
        
    Returns:
        (num_known_speakers, num_unknown_speakers)
    """
    random.seed(seed)
    
    entries = load_manifest(input_json)
    if not entries:
        raise ValueError("No entries found in input JSON manifest")

    speakers: dict[str, list[dict]] = defaultdict(list)
    for line_no, entry in enumerate(entries, start=1):
        label = entry.get("label") or entry.get("id") or entry.get("speaker")
        if label is None:
            raise ValueError(f"Entry {line_no} in {input_json} is missing a label.")
        label = str(label)
        if "audio_filepath" not in entry:
            raise ValueError(f"Entry {line_no} in {input_json} is missing audio_filepath.")

        duration = entry.get("duration")
        if duration is not None:
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                duration = None

        normalized = {
            "audio_filepath": entry["audio_filepath"],
            "offset": float(entry.get("offset", 0.0) or 0.0),
            "duration": duration,
            "label": label,
        }
        speakers[label].append(normalized)
    
    if not speakers:
        raise ValueError("No speakers found in input JSON")
    
    speaker_ids = list(speakers.keys())
    random.shuffle(speaker_ids)
    
    split_idx = int(len(speaker_ids) * known_ratio)
    known_ids = speaker_ids[:split_idx]
    unknown_ids = speaker_ids[split_idx:]
    
    known_entries = [clip for sid in known_ids for clip in speakers[sid]]
    unknown_entries = [clip for sid in unknown_ids for clip in speakers[sid]]
    
    print(f"Total speakers: {len(speaker_ids)}")
    print(f"Known speakers: {len(known_ids)}")
    print(f"Unknown speakers: {len(unknown_ids)}")
    
    total_clips = len(known_entries) + len(unknown_entries)
    known_clips = len(known_entries)
    unknown_clips = len(unknown_entries)
    
    print(f"\nKnown clips: {known_clips} ({100*known_clips/total_clips:.1f}%)")
    print(f"Unknown clips: {unknown_clips} ({100*unknown_clips/total_clips:.1f}%)")
    
    save_manifest(known_output, known_entries)
    print(f"\nSaved known speakers to: {known_output}")
    
    save_manifest(unknown_output, unknown_entries)
    print(f"Saved unknown speakers to: {unknown_output}")
    
    return len(known_ids), len(unknown_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a speaker manifest into known (80%) and unknown (20%) sets"
    )
    parser.add_argument(
        "--input",
        default="../../dataspeakersknownspeaker.json",
        help=(
            "Path to input speaker manifest (JSONL or legacy nested JSON) "
            "(default: ../../dataspeakersknownspeaker.json)"
        ),
    )
    parser.add_argument(
        "--known-output",
        default="known_speakers.json",
        help="Path to save known speakers manifest (default: known_speakers.json)",
    )
    parser.add_argument(
        "--unknown-output",
        default="unknown_speakers.json",
        help="Path to save unknown speakers manifest (default: unknown_speakers.json)",
    )
    parser.add_argument(
        "--known-ratio",
        type=float,
        default=0.8,
        help="Ratio of speakers for known set (default: 0.8 = 80%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Ensure output directories exist
    Path(args.known_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.unknown_output).parent.mkdir(parents=True, exist_ok=True)
    
    split_speakers(
        input_json=str(input_path),
        known_output=args.known_output,
        unknown_output=args.unknown_output,
        known_ratio=args.known_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
