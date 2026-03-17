from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable


def _speaker_id_from_path(audio_path: Path, root: Path) -> str | None:
    rel_parts = audio_path.relative_to(root).parts
    for part in rel_parts:
        if part.isdigit():
            return part
    return None


def _collect_audio_by_speaker(root: Path, ext: str) -> dict[str, list[Path]]:
    speakers: dict[str, list[Path]] = {}
    for audio_path in root.rglob(f"*{ext}"):
        if not audio_path.is_file():
            continue
        speaker_id = _speaker_id_from_path(audio_path, root)
        if speaker_id is None:
            continue
        speakers.setdefault(speaker_id, []).append(audio_path)
    return speakers


def _select_speakers(
    speakers: dict[str, list[Path]],
    num_speakers: int,
    rng: random.Random,
) -> dict[str, list[Path]]:
    speaker_ids = sorted(speakers.keys(), key=lambda s: int(s) if s.isdigit() else s)
    if len(speaker_ids) <= num_speakers:
        chosen_ids = speaker_ids
    else:
        chosen_ids = sorted(rng.sample(speaker_ids, num_speakers))
    return {speaker_id: speakers[speaker_id] for speaker_id in chosen_ids}


def _build_label_map(speaker_ids: Iterable[str]) -> dict[str, int]:
    sorted_ids = sorted(speaker_ids, key=lambda s: int(s) if s.isdigit() else s)
    return {speaker_id: idx for idx, speaker_id in enumerate(sorted_ids)}


def _sample_clips_per_speaker(
    speaker_clips: dict[str, list[Path]],
    *,
    train_per_speaker: int,
    valid_per_speaker: int,
    test_per_speaker: int,
    rng: random.Random,
) -> dict[str, dict[str, list[Path]]]:
    splits: dict[str, dict[str, list[Path]]] = {
        "train": {},
        "valid": {},
        "test": {},
    }
    total_needed = train_per_speaker + valid_per_speaker + test_per_speaker

    for speaker_id, clips in speaker_clips.items():
        clips_sorted = sorted(clips)
        if len(clips_sorted) < total_needed:
            raise ValueError(
                f"Speaker {speaker_id} has {len(clips_sorted)} clips, "
                f"need at least {total_needed}."
            )
        chosen = rng.sample(clips_sorted, total_needed)
        idx = 0
        for split_name, take in (
            ("train", train_per_speaker),
            ("valid", valid_per_speaker),
            ("test", test_per_speaker),
        ):
            splits[split_name][speaker_id] = sorted(chosen[idx : idx + take])
            idx += take
    return splits


def select_librispeech_speakers(
    root: str | Path,
    *,
    num_speakers: int = 10,
    ext: str = ".flac",
    seed: int = 0,
) -> dict[str, list[Path]]:
    """
    Traverse LibriSpeech and select a subset of speakers.
    Returns a mapping: speaker_id -> list of clip paths.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root folder not found: {root_path}")

    speakers = _collect_audio_by_speaker(root_path, ext)
    if not speakers:
        raise ValueError(f"No audio files found under {root_path}")

    rng = random.Random(seed)
    return _select_speakers(speakers, num_speakers, rng)


def build_librispeech_sample_list(
    root: str | Path,
    *,
    num_speakers: int = 10,
    train_per_speaker: int = 10,
    valid_per_speaker: int = 2,
    test_per_speaker: int = 2,
    ext: str = ".flac",
    seed: int = 0,
) -> tuple[list[dict], dict[str, int]]:
    """
    Return a flat list of samples and a label map.

    Each speaker contributes train_per_speaker + valid_per_speaker + test_per_speaker clips.
    The list contains dicts with keys: audio_filepath, label, label_id, split.
    """
    rng = random.Random(seed)
    speakers = select_librispeech_speakers(
        root, num_speakers=num_speakers, ext=ext, seed=seed
    )
    label_map = _build_label_map(speakers.keys())
    splits = _sample_clips_per_speaker(
        speakers,
        train_per_speaker=train_per_speaker,
        valid_per_speaker=valid_per_speaker,
        test_per_speaker=test_per_speaker,
        rng=rng,
    )

    samples: list[dict] = []
    for split_name in ("train", "valid", "test"):
        for speaker_id, clips in splits[split_name].items():
            label_id = label_map[speaker_id]
            for clip in clips:
                samples.append(
                    {
                        "audio_filepath": str(clip),
                        "label": speaker_id,
                        "label_id": label_id,
                        "split": split_name,
                    }
                )
    return samples, label_map
