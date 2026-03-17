"""
Speaker Enrollment (lightweight version).
Creates .npy embeddings + .json metadata files compatible with recognition_rknn_lite.py

This is designed to run on a host machine (with torch) to prepare embeddings
that will then be transferred to the embedded device.

Dependencies: argparse, json, numpy, pathlib, torch (host only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def enroll_speaker_torch(
    speaker_id: str,
    audio_files: Iterable[Path],
    *,
    model_path: Path = Path("ECAPATDNN_protonet_model.pth"),
    sr: int = 16000,
    n_mels: int = 80,
    duration: float = 3.0,
    n_fft: int = 1024,
    hop_length: int = 256,
    apply_filter: bool = True,
    filter_top_db: float = 20.0,
    chunk_duration: Optional[float] = None,
    chunk_overlap: float = 0.5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate embedding using PyTorch model (host-side enrollment).
    
    Args:
        speaker_id: Unique speaker identifier
        audio_files: Iterable of audio file paths
        model_path: Path to pretrained model .pth
        ... (standard audio parameters)
        verbose: Print debug info
        
    Returns:
        Speaker embedding as float32 numpy array
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for host-side enrollment")
    
    import torch.nn.functional as F
    from utils.data_preprocessing import audio_chunking, audio_to_mel_spectrogram
    from utils.model_functions import load_model
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    
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
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if verbose:
            print(f"  Processing: {audio_path.name}")
        
        # Load audio and compute mel
        _, y, sr_loaded = audio_to_mel_spectrogram(
            audio_path=audio_path,
            sr=sr,
            duration=None,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            to_db=True,
        )
        
        # Chunk and embed
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
            if verbose:
                print(f"    Warning: No usable chunks from {audio_path.name}")
            continue
        
        for mel in mel_chunks:
            with torch.no_grad():
                emb = model(mel.to(device)).squeeze(0).cpu()
            embeddings.append(emb)
    
    if not embeddings:
        raise ValueError(f"No usable embeddings for speaker {speaker_id}")
    
    # Average and normalize
    stacked = torch.stack(embeddings, dim=0)
    speaker_embedding = F.normalize(stacked.mean(dim=0), p=2, dim=0).cpu()
    
    return speaker_embedding.numpy().astype(np.float32)


def save_enrollment(
    speaker_id: str,
    embedding: np.ndarray,
    store_path: Path = Path("enrolled_speakers.json"),
    verbose: bool = False,
) -> None:
    """
    Save speaker embedding as .npy file with .json metadata.
    
    Args:
        speaker_id: Speaker identifier
        embedding: Embedding vector (1D float32)
        store_path: Path to .json metadata file (base name)
        verbose: Print debug info
    """
    store_path = Path(store_path)
    store_dir = store_path.parent
    store_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing metadata or create new
    if store_path.exists():
        with open(store_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Generate embedding filename
    emb_filename = f"{speaker_id}_embedding.npy"
    emb_filepath = store_dir / emb_filename
    
    # Save embedding
    np.save(str(emb_filepath), embedding)
    
    # Update metadata
    metadata[speaker_id] = {
        "embedding_file": emb_filename,
        "embedding_dim": int(embedding.shape[0]),
        "dtype": "float32",
    }
    
    # Save metadata
    with open(store_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"Saved enrollment for '{speaker_id}':")
        print(f"  Embedding: {emb_filepath} ({embedding.nbytes} bytes)")
        print(f"  Dimension: {embedding.shape[0]}")
        print(f"  Metadata: {store_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enroll speaker with embeddings for lightweight recognition."
    )
    parser.add_argument(
        "--speaker-id",
        required=True,
        help="Unique speaker identifier"
    )
    parser.add_argument(
        "--audio-files",
        required=True,
        nargs="+",
        type=Path,
        help="Audio files for enrollment (space-separated)"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ECAPATDNN_protonet_model.pth"),
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path("enrolled_speakers.json"),
        help="Path to save enrollment metadata (JSON)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate"
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="Number of mel bins"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Target duration in seconds"
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=1024,
        help="FFT size"
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
        help="Hop length"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable silence removal"
    )
    parser.add_argument(
        "--filter-top-db",
        type=float,
        default=20.0,
        help="Silence removal threshold"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help="Chunk duration in seconds (default: use target duration)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=0.5,
        help="Chunk overlap in seconds"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.verbose:
        print(f"Enrolling speaker: {args.speaker_id}")
        print(f"Audio files: {len(args.audio_files)}")
    
    try:
        # Generate embedding
        embedding = enroll_speaker_torch(
            speaker_id=args.speaker_id,
            audio_files=args.audio_files,
            model_path=args.model_path,
            sr=args.sr,
            n_mels=args.n_mels,
            duration=args.duration,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            apply_filter=not args.no_filter,
            filter_top_db=args.filter_top_db,
            chunk_duration=args.chunk_duration,
            chunk_overlap=args.chunk_overlap,
            verbose=args.verbose,
        )
        
        # Save enrollment
        save_enrollment(
            speaker_id=args.speaker_id,
            embedding=embedding,
            store_path=args.store_path,
            verbose=args.verbose,
        )
        
        print(f"✓ Successfully enrolled speaker '{args.speaker_id}'")
        
    except Exception as e:
        print(f"✗ Enrollment failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
