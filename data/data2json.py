"""
Utility to build a speaker catalog JSON from a folder of audio clips.

The default layout targets the LibriSpeech structure where files live under
``data/dataset/Libri-speech/<speaker>/<chapter>/<clip>.flac``.
The resulting JSON matches the schema expected by ``parser.load_waveforms_from_json``:

{
  "config": {"target_sr": 16000},
  "speakers": [
      {"id": "speaker_id", "clips": ["path/to/audio1.flac", ...]}
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

# Default audio suffixes to collect. Extend via --extensions if needed.
KNOWN_AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a speaker dataset (e.g., LibriSpeech) and emit a JSON catalog "
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
        help="Sampling rate metadata to store in the JSON config.",
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


def build_catalog(
    dataset_root: Path,
    *,
    suffixes: Iterable[str],
    max_clips: int | None,
    export_base: Path,
    target_sr: int,
) -> dict:
    speaker_to_clips: dict[str, list[str]] = {}
    for audio_path in iter_audio_files(dataset_root, suffixes):
        speaker_id = speaker_id_from_path(audio_path, dataset_root)
        clip_list = speaker_to_clips.setdefault(speaker_id, [])
        if max_clips is not None and len(clip_list) >= max_clips:
            continue
        clip_list.append(make_path_exportable(audio_path, export_base))

    speakers = []
    for speaker_id, clips in sorted(speaker_to_clips.items()):
        clips.sort()
        speakers.append({"id": speaker_id, "clips": clips})

    return {"config": {"target_sr": target_sr}, "speakers": speakers}


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    suffixes = normalise_extensions(args.extensions)
    export_base = args.relative_to.resolve()
    catalog = build_catalog(
        dataset_root,
        suffixes=suffixes,
        max_clips=args.max_clips,
        export_base=export_base,
        target_sr=args.target_sr,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(catalog['speakers'])} speakers "
        f"({sum(len(s['clips']) for s in catalog['speakers'])} clips) to {args.output}"
    )


if __name__ == "__main__":
    main()
