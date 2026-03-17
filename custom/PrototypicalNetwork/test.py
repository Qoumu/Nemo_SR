from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import build_prototypical_dataset  # noqa: E402
from utils.model_functions import load_model  # noqa: E402
from enroll import enroll_speaker  # noqa: E402
from recognition import (  # noqa: E402
    compute_embedding_from_file,
    load_enrolled_speakers,
)


def split_test_samples(
    test_samples: Iterable[dict],
    *,
    enroll_ratio: float,
    seed: int,
) -> Tuple[Dict[str, List[Path]], List[Tuple[str, Path]]]:
    """
    Split each speaker's test samples into enrollment and recognition sets.

    Returns:
        enroll_map: speaker_id -> list of Paths for enrollment
        eval_list: list of tuples (speaker_id, Path) for recognition
    """
    if not (0.0 < enroll_ratio < 1.0):
        raise ValueError("enroll_ratio must be in (0, 1).")

    grouped: dict[str, list[str]] = defaultdict(list)
    for item in test_samples:
        grouped[item["label"]].append(item["audio_filepath"])

    rng = random.Random(seed)
    enroll_map: Dict[str, List[Path]] = {}
    eval_list: List[Tuple[str, Path]] = []

    for speaker_id, clips in grouped.items():
        if len(clips) < 2:
            raise ValueError(
                f"Speaker {speaker_id} has only {len(clips)} test clips; "
                "need at least 2 to split into enroll/recognize."
            )

        rng.shuffle(clips)
        n_enroll = max(1, int(len(clips) * enroll_ratio))
        n_enroll = min(n_enroll, len(clips) - 1)  # ensure at least one for recognition

        enroll_map[speaker_id] = [Path(p) for p in clips[:n_enroll]]
        for clip in clips[n_enroll:]:
            eval_list.append((speaker_id, Path(clip)))

    return enroll_map, eval_list


def prepare_store(store_path: Path, overwrite: bool) -> None:
    """Ensure the enrollment store is ready for a fresh run."""
    if store_path.exists():
        if overwrite:
            store_path.unlink()
        else:
            raise SystemExit(
                f"Enrollment store already exists at {store_path}. "
                "Use --overwrite-store to replace it."
            )
    store_path.parent.mkdir(parents=True, exist_ok=True)


def enroll_all_speakers(
    enroll_map: Dict[str, List[Path]],
    *,
    model_path: Path,
    store_path: Path,
    sr: int,
    n_mels: int,
    duration: float,
    n_fft: int,
    hop_length: int,
    apply_filter: bool,
    filter_top_db: float,
    chunk_duration: float,
    chunk_overlap: float,
) -> None:
    """Enroll every speaker using the provided clips."""
    for speaker_id, audio_files in enroll_map.items():
        print(f"Enrolling speaker {speaker_id} with {len(audio_files)} file(s)...")
        enroll_speaker(
            speaker_id=speaker_id,
            audio_files=audio_files,
            model_path=model_path,
            store_path=store_path,
            sr=sr,
            n_mels=n_mels,
            duration=duration,
            n_fft=n_fft,
            hop_length=hop_length,
            apply_filter=apply_filter,
            filter_top_db=filter_top_db,
            chunk_duration=chunk_duration,
            chunk_overlap=chunk_overlap,
        )


def evaluate_recognition(
    eval_samples: List[Tuple[str, Path]],
    *,
    model,
    enrolled: Dict[str, torch.Tensor],
    sr: int,
    n_mels: int,
    duration: float,
    n_fft: int,
    hop_length: int,
    apply_filter: bool,
    filter_top_db: float,
    chunk_duration: float,
    chunk_overlap: float,
    threshold: float,
) -> Tuple[float, List[Tuple[Path, str, str | None, float]]]:
    """Run recognition against enrolled speakers and compute accuracy."""
    if not eval_samples:
        return 0.0, []

    device = next(model.parameters()).device
    correct = 0
    results: List[Tuple[Path, str, str | None, float]] = []

    for true_speaker, audio_path in eval_samples:
        query_emb = compute_embedding_from_file(
            audio_path,
            model,
            device,
            sr=sr,
            n_mels=n_mels,
            duration=duration,
            n_fft=n_fft,
            hop_length=hop_length,
            apply_filter=apply_filter,
            filter_top_db=filter_top_db,
            chunk_duration=chunk_duration,
            chunk_overlap=chunk_overlap,
        )

        best_speaker = None
        best_score = float("-inf")
        for spk, emb in enrolled.items():
            score = float(F.cosine_similarity(query_emb, emb, dim=0))
            if score > best_score:
                best_score = score
                best_speaker = spk

        predicted = best_speaker if best_score >= threshold else None
        correct += int(predicted == true_speaker)
        results.append((audio_path, true_speaker, predicted, best_score))

    accuracy = 100.0 * correct / len(eval_samples)
    return accuracy, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enroll and evaluate the LibriSpeech test split (8:2 enroll/recognize)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/d/Projects/Nemo_SR/data/speakerdataset/LibriSpeech"),
        help="Path to the LibriSpeech root.",
    )
    parser.add_argument("--num-speakers", type=int, default=40, help="Speakers to sample.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train speaker ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation speaker ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test speaker ratio.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum clips per speaker to keep.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=15,
        help="Cap clips per speaker (set <=0 for no cap).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection.")
    parser.add_argument(
        "--enroll-ratio",
        type=float,
        default=0.6,
        help="Portion of each speaker's test clips used for enrollment.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ECAPATDNN_protonet_model.pth"),
        help="Trained ECAPA-TDNN checkpoint.",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path("test_enrolled_speakers.pt"),
        help="Where to save the temporary enrollment store.",
    )
    parser.add_argument(
        "--overwrite-store",
        action="store_true",
        help="Replace an existing store at --store-path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Cosine similarity threshold for acceptance.",
    )
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bins.")
    parser.add_argument("--duration", type=float, default=3.0, help="Mel duration in seconds.")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT size.")
    parser.add_argument("--hop-length", type=int, default=256, help="Hop length.")
    parser.add_argument(
        "--filter-top-db",
        type=float,
        default=10.0,
        help="Silence removal aggressiveness (higher keeps more).",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=10.0,
        help="Chunk length in seconds when generating embeddings.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=0.5,
        help="Chunk overlap in seconds.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable silence removal before embedding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_list, _ = build_prototypical_dataset(
        root=args.data_root,
        num_speakers=args.num_speakers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_samples_per_speaker=args.min_samples,
        max_samples_per_speaker=args.max_samples if args.max_samples > 0 else None,
        seed=args.seed,
    )

    test_samples = [item for item in dataset_list if item["split"] == "test"]
    test_speakers = sorted({item["label"] for item in test_samples})
    print(f"Found {len(test_samples)} test clips across {len(test_speakers)} speakers.")

    enroll_map, eval_samples = split_test_samples(
        test_samples,
        enroll_ratio=args.enroll_ratio,
        seed=args.seed,
    )
    total_enroll = sum(len(v) for v in enroll_map.values())
    print(
        f"Split test set -> enroll clips: {total_enroll}, "
        f"recognition clips: {len(eval_samples)} (ratio {args.enroll_ratio:.2f}:{1-args.enroll_ratio:.2f})"
    )

    prepare_store(args.store_path, overwrite=args.overwrite_store)
    enroll_all_speakers(
        enroll_map,
        model_path=args.model_path,
        store_path=args.store_path,
        sr=args.sr,
        n_mels=args.n_mels,
        duration=args.duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        apply_filter=not args.no_filter,
        filter_top_db=args.filter_top_db,
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        args.model_path,
        device=device,
        n_mels=args.n_mels,
        emb_dim=64,
        channels=512,
    )
    enrolled = load_enrolled_speakers(args.store_path)
    accuracy, results = evaluate_recognition(
        eval_samples,
        model=model,
        enrolled=enrolled,
        sr=args.sr,
        n_mels=args.n_mels,
        duration=args.duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        apply_filter=not args.no_filter,
        filter_top_db=args.filter_top_db,
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap,
        threshold=args.threshold,
    )

    print(f"\nRecognition accuracy: {accuracy:.2f}% on {len(results)} clip(s) (threshold {args.threshold}).")
    failures = [r for r in results if r[1] != r[2]]
    if failures:
        print(f"Failures: {len(failures)}")
        for audio_path, true_spk, pred_spk, score in failures[:10]:
            print(
                f"  {audio_path} | true={true_spk} pred={pred_spk} score={score:.4f}"
            )
    else:
        print("All recognition samples matched their enrolled speaker.")


if __name__ == "__main__":
    main()
