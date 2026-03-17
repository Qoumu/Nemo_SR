from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from utils.data_preprocessing import audio_chunking, audio_to_mel_spectrogram
from utils.model_functions import load_model


def enroll_speaker(
    speaker_id: str,
    audio_files: Iterable[Path],
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
) -> torch.Tensor:
    """
    Generate an embedding for a new speaker from chunked audio and append/update
    the store.

    Returns the normalized speaker embedding tensor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_path,
        device=device,
        n_mels=n_mels,
        emb_dim=64,
        channels=512,
    )

    embeddings: list[torch.Tensor] = []
    chunk_dur = chunk_duration or duration

    for audio_path in audio_files:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        # Load waveform once, then chunk to generate multiple embeddings
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
            chunk_duration=chunk_dur,
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

        for mel in mel_chunks:
            with torch.no_grad():
                emb = model(mel.to(device)).squeeze(0).cpu()
            embeddings.append(emb)

    if not embeddings:
        raise ValueError("No audio files provided for enrollment.")

    stacked = torch.stack(embeddings, dim=0)
    speaker_embedding = F.normalize(stacked.mean(dim=0), p=2, dim=0).cpu()

    if store_path.exists():
        store = torch.load(store_path, map_location="cpu")
        if not isinstance(store, dict):
            raise ValueError(f"Existing store {store_path} is not a dict.")
    else:
        store = {}

    store[str(speaker_id)] = speaker_embedding
    store_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, store_path)

    print(f"Enrolled speaker '{speaker_id}' with {len(embeddings)} sample(s).")
    print(f"Updated embedding store at: {store_path}")
    return speaker_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll a new speaker.")
    parser.add_argument(
        "--speaker-id",
        required=True,
        help="Identifier/name for the new speaker.",
    )
    parser.add_argument(
        "--audio-files",
        nargs="+",
        required=True,
        type=Path,
        help="Paths to audio files for enrollment.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ECAPATDNN_protonet_model.pth"),
        help="Path to the trained ECAPA-TDNN model weights.",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path("enrolled_speakers.pt"),
        help="Where to save the speaker embedding store.",
    )
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bins.")
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Target duration (seconds) for mel padding/cropping.",
    )
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT size.")
    parser.add_argument("--hop-length", type=int, default=256, help="Hop length.")
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable silence removal before computing the mel spectrogram.",
    )
    parser.add_argument(
        "--filter-top-db",
        type=float,
        default=20.0,
        help="Top dB threshold for silence removal (ignored when --no-filter).",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=10,
        help="Chunk length in seconds (default: use --duration).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=0.5,
        help="Overlap in seconds between successive chunks.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    enroll_speaker(
        speaker_id=args.speaker_id,
        audio_files=args.audio_files,
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
