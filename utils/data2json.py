"""
Utility to build a JSON-lines manifest from a folder of audio clips.

Each line in the output file looks like:
  {"audio_filepath": "path/to/audio.flac", "offset": 0, "duration": 4.16, "label": "speaker_id"}

The default layout targets LibriSpeech-style folders where files live under
``data/dataset/Libri-speech/<speaker>/<chapter>/<clip>.flac``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import soundfile as sf

# Default audio suffixes to collect. Extend via --extensions if needed.
KNOWN_AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a speaker dataset (e.g., LibriSpeech) and emit a JSONL manifest "
            "compatible with parser.load_waveforms_from_json."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/dataset/Libri-speech"),
        help="Root folder containing the speaker dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON file to write.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="(Deprecated) kept for backward compatibility; unused in JSONL output.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Optional cap on the number of clips per speaker.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=",".join(KNOWN_AUDIO_SUFFIXES),
        help="Comma-separated list of audio file extensions to include.",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help=(
            "Base path used to store clip paths. If the audio files are located under "
            "this directory, relative paths will be written; otherwise absolute paths "
            "are used."
        ),
    )
    return parser.parse_args()


def normalise_extensions(exts: str) -> set[str]:
    values = set()
    for token in exts.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.startswith("."):
            token = f".{token}"
        values.add(token.lower())
    return values or set(KNOWN_AUDIO_SUFFIXES)


def iter_audio_files(root: Path, suffixes: Iterable[str]) -> Iterable[Path]:
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in suffixes:
            yield file_path


def speaker_id_from_path(audio_path: Path, dataset_root: Path) -> str:
    rel = audio_path.relative_to(dataset_root)
    # LibriSpeech layout => speaker/chapter/file. First component is speaker id.
    if rel.parts:
        return rel.parts[0]
    return audio_path.stem


def make_path_exportable(path: Path, base: Path) -> str:
    try:
        rel = path.relative_to(base)
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def audio_duration_seconds(path: Path) -> float | None:
    """Return clip duration in seconds using audio metadata."""
    try:
        info = sf.info(path)
        if info.frames and info.samplerate:
            return round(info.frames / info.samplerate, 4)
    except Exception:
        return None
    return None


def build_manifest(
    dataset_root: Path,
    *,
    suffixes: Iterable[str],
    max_clips: int | None,
    export_base: Path,
) -> tuple[list[dict], dict[str, int]]:
    manifest: list[dict] = []
    speaker_counts: dict[str, int] = defaultdict(int)

    for audio_path in iter_audio_files(dataset_root, suffixes):
        speaker_id = speaker_id_from_path(audio_path, dataset_root)
        if max_clips is not None and speaker_counts[speaker_id] >= max_clips:
            continue

        speaker_counts[speaker_id] += 1
        manifest.append(
            {
                "audio_filepath": make_path_exportable(audio_path, export_base),
                "offset": 0,
                "duration": audio_duration_seconds(audio_path),
                "label": speaker_id,
            }
        )

    return manifest, speaker_counts


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    suffixes = normalise_extensions(args.extensions)
    export_base = args.relative_to.resolve()
    manifest, speaker_counts = build_manifest(
        dataset_root,
        suffixes=suffixes,
        max_clips=args.max_clips,
        export_base=export_base,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in manifest:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote {len(manifest)} clips across {len(speaker_counts)} speakers to {args.output}")


if __name__ == "__main__":
    main()
