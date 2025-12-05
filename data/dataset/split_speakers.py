"""
Split LibriSpeech dataset into known and unknown speaker sets.
- Known speakers: 80% of speakers (for enrollment/training)
- Unknown speakers: 20% of speakers (for validation/testing)
"""

import json
import random
import argparse
from pathlib import Path


def load_speaker_json(json_path: str) -> dict:
    """Load speaker JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_speaker_json(path: str, data: dict) -> None:
    """Save speaker JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_speakers(
    input_json: str,
    known_output: str,
    unknown_output: str,
    known_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[int, int]:
    """
    Split speakers into known and unknown sets.
    
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
    
    # Load full dataset
    data = load_speaker_json(input_json)
    config = data.get("config", {"target_sr": 16000})
    speakers = data.get("speakers", [])
    
    if not speakers:
        raise ValueError("No speakers found in input JSON")
    
    print(f"Total speakers: {len(speakers)}")
    
    # Shuffle and split speakers
    shuffled = speakers.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(speakers) * known_ratio)
    known_speakers = shuffled[:split_idx]
    unknown_speakers = shuffled[split_idx:]
    
    print(f"Known speakers: {len(known_speakers)}")
    print(f"Unknown speakers: {len(unknown_speakers)}")
    
    # Count clips
    known_clips = sum(len(s.get("clips", [])) for s in known_speakers)
    unknown_clips = sum(len(s.get("clips", [])) for s in unknown_speakers)
    total_clips = known_clips + unknown_clips
    
    print(f"\nKnown clips: {known_clips} ({100*known_clips/total_clips:.1f}%)")
    print(f"Unknown clips: {unknown_clips} ({100*unknown_clips/total_clips:.1f}%)")
    
    # Save known speakers
    known_data = {"config": config, "speakers": known_speakers}
    save_speaker_json(known_output, known_data)
    print(f"\nSaved known speakers to: {known_output}")
    
    # Save unknown speakers
    unknown_data = {"config": config, "speakers": unknown_speakers}
    save_speaker_json(unknown_output, unknown_data)
    print(f"Saved unknown speakers to: {unknown_output}")
    
    return len(known_speakers), len(unknown_speakers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split LibriSpeech speakers into known (80%) and unknown (20%) sets"
    )
    parser.add_argument(
        "--input",
        default="../../dataspeakersknownspeaker.json",
        help="Path to input speaker JSON file (default: ../../dataspeakersknownspeaker.json)",
    )
    parser.add_argument(
        "--known-output",
        default="known_speakers.json",
        help="Path to save known speakers (default: known_speakers.json)",
    )
    parser.add_argument(
        "--unknown-output",
        default="unknown_speakers.json",
        help="Path to save unknown speakers (default: unknown_speakers.json)",
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
