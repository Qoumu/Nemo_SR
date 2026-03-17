from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F

from utils.data_preprocessing import audio_chunking, audio_to_mel_spectrogram
from utils.model_functions import load_model


def load_enrolled_speakers(store_path: Path) -> dict[str, torch.Tensor]:
    """Load enrolled speaker embeddings from disk."""
    if not store_path.exists():
        raise FileNotFoundError(f"Enrollment store not found: {store_path}")

    store = torch.load(store_path, map_location="cpu")
    if not isinstance(store, dict):
        raise ValueError(f"Enrollment store must be a dict, got {type(store)}")

    normalized = {}
    for spk, emb in store.items():
        emb_tensor = torch.as_tensor(emb, dtype=torch.float32)
        normalized[spk] = F.normalize(emb_tensor, p=2, dim=0)
    return normalized


def compute_embedding_from_file(
    audio_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    *,
    sr: int = 16000,
    n_mels: int = 80,
    duration: float = 3.0,
    n_fft: int = 1024,
    hop_length: int = 256,
    apply_filter: bool = True,
    filter_top_db: float = 20.0,
    chunk_duration: float | None = None,
    chunk_overlap: float = 0.5,
) -> torch.Tensor:
    """Chunk the audio, embed each chunk, and average to one normalized vector."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    _, y, sr_loaded = audio_to_mel_spectrogram(
        audio_path=audio_path,
        sr=sr,
        duration=None,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        to_db=True,
    )

    mel_chunks = audio_chunking(
        y,
        sr_loaded,
        chunk_duration=chunk_duration or duration,
        overlap_duration=chunk_overlap,
        return_mels=True,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        target_duration=duration,
        apply_filter=apply_filter,
        filter_top_db=filter_top_db,
    )

    if not mel_chunks:
        raise ValueError(f"No usable chunks produced from {audio_path}")

    embeds = []
    model.eval()
    with torch.no_grad():
        for mel in mel_chunks:
            emb = model(mel.to(device)).squeeze(0).cpu()
            embeds.append(emb)

    mean_emb = torch.stack(embeds, dim=0).mean(dim=0)
    return F.normalize(mean_emb, p=2, dim=0)


def recognize_speaker(
    audio_path: Path,
    *,
    model_path: Path = Path("ECAPATDNN_protonet_model.pth"),
    store_path: Path = Path("enrolled_speakers.pt"),
    sr: int = 16000,
    n_mels: int = 80,
    duration: float = 3.0,
    n_fft: int = 1024,
    hop_length: int = 256,
    apply_filter: bool = True,
    filter_top_db: float = 20.0,
    chunk_duration: float | None = None,
    chunk_overlap: float = 0.5,
    threshold: float = 0.6,
) -> Tuple[str | None, float]:
    """
    Predict if the audio matches any enrolled speaker. Returns (speaker_id or None, score).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_path,
        device=device,
        n_mels=n_mels,
        emb_dim=64,
        channels=512,
    )

    enrolled = load_enrolled_speakers(store_path)
    if not enrolled:
        raise ValueError("Enrollment store is empty.")

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

    decision = best_speaker if best_score >= threshold else None
    return decision, best_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize whether an audio file matches an enrolled speaker."
    )
    parser.add_argument("--audio-file", required=True, type=Path, help="Path to audio to test.")
    parser.add_argument("--model-path", type=Path, default=Path("ECAPATDNN_protonet_model.pth"))
    parser.add_argument("--store-path", type=Path, default=Path("enrolled_speakers.pt"))
    parser.add_argument("--threshold", type=float, default=0.8, help="Cosine similarity decision threshold.")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--no-filter", action="store_true", help="Disable silence removal.")
    parser.add_argument("--filter-top-db", type=float, default=20.0)
    parser.add_argument("--chunk-duration", type=float, default=10, help="Chunk length (sec). Default: duration.")
    parser.add_argument("--chunk-overlap", type=float, default=0.5, help="Chunk overlap (sec).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    speaker, score = recognize_speaker(
        audio_path=args.audio_file,
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
        threshold=args.threshold,
    )

    if speaker is None:
        print(f"Rejected: best similarity {score:.4f} below threshold {args.threshold}")
    else:
        print(f"Matched speaker '{speaker}' with similarity {score:.4f} (threshold {args.threshold})")

