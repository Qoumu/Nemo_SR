"""
Speaker Recognition using RKNN Runtime (ctypes-based).
Designed for Luckfox Pico Pro Max RV1106G3 (no torch, minimal dependencies).

Dependencies: argparse, json, ctypes, numpy, soundfile
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import soundfile as sf

from rknn_ctypes import RKNNLite, RKNNError


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2 normalize a vector."""
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm


def load_enrolled_speakers(store_path: Path) -> Dict[str, np.ndarray]:
    """Load enrolled speaker embeddings from JSON metadata + .npy files."""
    metadata_path = store_path.with_suffix('.json')
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Enrollment metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a dict, got {type(metadata)}")
    
    store_dir = metadata_path.parent
    normalized: Dict[str, np.ndarray] = {}
    
    for speaker_id, info in metadata.items():
        emb_file = store_dir / info['embedding_file']
        if not emb_file.exists():
            raise FileNotFoundError(f"Embedding file not found: {emb_file}")
        
        emb_np = np.load(str(emb_file)).astype(np.float32).reshape(-1)
        normalized[str(speaker_id)] = _l2_normalize(emb_np)
    
    return normalized


def audio_to_mel_spectrogram(
    audio_path: Path,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> np.ndarray:
    """
    Load audio and convert to mel spectrogram.
    
    Returns:
        mel: (n_mels, time) float32 array
    """
    # Load audio
    y, sr_loaded = sf.read(str(audio_path), dtype='float32')
    
    if y.ndim > 1:
        y = y[:, 0]  # Take first channel if stereo
    
    # Resample if needed
    if sr_loaded != sr:
        # Simple nearest-neighbor resampling
        ratio = sr / sr_loaded
        new_length = int(len(y) * ratio)
        y = np.interp(
            np.linspace(0, len(y) - 1, new_length),
            np.arange(len(y)),
            y
        )
        sr_loaded = sr
    
    # Compute STFT
    D = np.abs(np.fft.rfft(y[None, :] * np.hanning(len(y)), n=n_fft, axis=1))
    
    # Mel-scale filterbank
    mel_freqs = np.linspace(0, sr_loaded // 2, n_mels * 2 + 2)
    mel_freqs_hz = 2595 * np.log10(1 + mel_freqs / 700)
    
    # Simple mel spectrogram (linear interpolation)
    freq_bins = np.linspace(0, sr_loaded // 2, D.shape[1])
    mel = np.zeros((n_mels, D.shape[1]))
    for i in range(n_mels):
        f_center = mel_freqs[i + 1]
        f_left = mel_freqs[i]
        f_right = mel_freqs[i + 2]
        
        left_slope = (freq_bins - f_left) / (f_center - f_left + 1e-8)
        right_slope = (f_right - freq_bins) / (f_right - f_center + 1e-8)
        
        triangle = np.maximum(0, np.minimum(left_slope, right_slope))
        mel[i] = np.max(D[0] * triangle)
    
    # Convert to dB scale
    mel_db = 10 * np.log10(np.maximum(mel, 1e-5))
    
    # Normalize
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
    
    return mel_db.astype(np.float32)


def compute_embedding_from_file(
    audio_path: Path,
    embedder: RKNNLite,
    *,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    chunk_duration: float = 3.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Process audio file and compute normalized embedding using RKNN.
    
    Args:
        audio_path: Path to audio file
        embedder: RKNNLite instance
        sr: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT size
        hop_length: Hop length
        chunk_duration: Chunk duration in seconds
        verbose: Print debug info
        
    Returns:
        Normalized embedding vector
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if verbose:
        print(f"Loading audio: {audio_path}")
    
    # Load and compute mel spectrogram
    mel = audio_to_mel_spectrogram(
        audio_path,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    
    # Chunk into fixed-size segments
    chunk_samples = int(chunk_duration * sr / hop_length)
    n_chunks = max(1, (mel.shape[1] - chunk_samples) // (chunk_samples // 2) + 1)
    
    embeds = []
    for i in range(n_chunks):
        start = i * (chunk_samples // 2)
        end = start + chunk_samples
        
        if end > mel.shape[1]:
            end = mel.shape[1]
            start = max(0, end - chunk_samples)
        
        mel_chunk = mel[:, start:end]
        
        # Pad if necessary
        if mel_chunk.shape[1] < chunk_samples:
            pad_width = ((0, 0), (0, chunk_samples - mel_chunk.shape[1]))
            mel_chunk = np.pad(mel_chunk, pad_width, mode='constant')
        
        # Prepare input: add batch dimension [1, n_mels, time]
        mel_input = mel_chunk[np.newaxis, :, :].astype(np.float32)
        
        if verbose:
            print(f"  Chunk {i+1}/{n_chunks}: shape {mel_input.shape}")
        
        # Run inference
        try:
            outputs = embedder.inference(inputs=[mel_input])
            if outputs and len(outputs) > 0:
                emb = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
                embeds.append(_l2_normalize(emb))
            else:
                if verbose:
                    print(f"  Warning: No output from chunk {i+1}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Inference failed for chunk {i+1}: {e}")
            continue
    
    if not embeds:
        raise ValueError(f"Failed to extract embeddings from {audio_path}")
    
    # Average embeddings
    mean_emb = np.mean(np.stack(embeds, axis=0), axis=0).astype(np.float32)
    return _l2_normalize(mean_emb)


def recognize_speaker(
    query_emb: np.ndarray,
    enrolled: Dict[str, np.ndarray],
    threshold: float = 0.8,
) -> Tuple[Optional[str], float]:
    """
    Match query embedding against enrolled speakers.
    
    Args:
        query_emb: Query speaker embedding
        enrolled: Dict of speaker_id -> normalized embedding
        threshold: Similarity threshold for acceptance
        
    Returns:
        (matched_speaker_id, similarity_score) or (None, best_score)
    """
    best_speaker = None
    best_score = float("-inf")
    
    for speaker_id, emb in enrolled.items():
        score = float(np.dot(query_emb, emb))
        if score > best_score:
            best_score = score
            best_speaker = speaker_id
    
    decision = best_speaker if best_score >= threshold else None
    return decision, best_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speaker recognition using RKNN runtime on embedded device (no torch)."
    )
    parser.add_argument(
        "--audio-file",
        required=True,
        type=Path,
        help="Audio file to recognize"
    )
    parser.add_argument(
        "--rknn-model",
        required=True,
        type=Path,
        help="Path to .rknn model file"
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path("enrolled_speakers.json"),
        help="Path to enrollment metadata (JSON)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for acceptance"
    )
    parser.add_argument(
        "--core-mask",
        type=str,
        default="auto",
        choices=["auto", "0", "1", "2", "0_1_2"],
        help="NPU core mask"
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
        "--chunk-duration",
        type=float,
        default=3.0,
        help="Chunk duration in seconds"
    )
    parser.add_argument(
        "--rknn-lib",
        type=str,
        default=None,
        help="Path to librknn_runtime.so (auto-detect if not specified)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Validate inputs
    if not args.rknn_model.exists():
        raise FileNotFoundError(f"RKNN model not found: {args.rknn_model}")
    
    if not args.audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_file}")
    
    # Load enrolled speakers
    if args.verbose:
        print("Loading enrolled speakers...")
    enrolled = load_enrolled_speakers(args.store_path)
    if not enrolled:
        raise ValueError("No enrolled speakers found")
    if args.verbose:
        print(f"Loaded {len(enrolled)} enrolled speaker(s)")
    
    # Resolve core mask
    core_mask_map = {
        "auto": RKNNLite.NPU_CORE_AUTO,
        "0": RKNNLite.NPU_CORE_0,
        "1": RKNNLite.NPU_CORE_1,
        "2": RKNNLite.NPU_CORE_2,
        "0_1_2": RKNNLite.NPU_CORE_0_1_2,
    }
    core_mask = core_mask_map.get(args.core_mask, RKNNLite.NPU_CORE_AUTO)
    
    # Initialize RKNN
    if args.verbose:
        print("Initializing RKNN runtime...")
    try:
        embedder = RKNNLite(verbose=args.verbose, lib_path=args.rknn_lib)
        ret = embedder.load_rknn(str(args.rknn_model))
        if ret != embedder.RKNN_SUCC:
            raise RKNNError(f"Failed to load model: code {ret}")
        
        ret = embedder.init_runtime(core_mask=core_mask)
        if ret != embedder.RKNN_SUCC:
            raise RKNNError(f"Failed to init runtime: code {ret}")
    except RKNNError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Compute embedding for query audio
    if args.verbose:
        print("Processing query audio...")
    t_start = time.time()
    try:
        query_emb = compute_embedding_from_file(
            audio_path=args.audio_file,
            embedder=embedder,
            sr=args.sr,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            chunk_duration=args.chunk_duration,
            verbose=args.verbose,
        )
    finally:
        embedder.release()
    
    elapsed = time.time() - t_start
    if args.verbose:
        print(f"Embedding computed in {elapsed:.2f}s")
    
    # Recognize speaker
    matched_speaker, score = recognize_speaker(
        query_emb,
        enrolled,
        threshold=args.threshold
    )
    
    # Output results
    if matched_speaker is None:
        print(
            f"REJECTED: best similarity {score:.4f} below threshold {args.threshold}"
        )
        sys.exit(1)
    else:
        print(
            f"MATCHED: speaker '{matched_speaker}' with similarity "
            f"{score:.4f} (threshold {args.threshold})"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
