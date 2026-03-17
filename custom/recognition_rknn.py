from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from utils.data_preprocessing import audio_chunking, audio_to_mel_spectrogram


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm


def load_enrolled_speakers(store_path: Path) -> Dict[str, np.ndarray]:
    """Load enrolled speaker embeddings from .pt and return normalized numpy vectors."""
    if not store_path.exists():
        raise FileNotFoundError(f"Enrollment store not found: {store_path}")

    store = torch.load(store_path, map_location="cpu")
    if not isinstance(store, dict):
        raise ValueError(f"Enrollment store must be a dict, got {type(store)}")

    normalized: Dict[str, np.ndarray] = {}
    for speaker_id, emb in store.items():
        emb_np = torch.as_tensor(emb, dtype=torch.float32).cpu().numpy().reshape(-1)
        normalized[str(speaker_id)] = _l2_normalize(emb_np.astype(np.float32))
    return normalized


class RKNNEmbedder:
    """Minimal RKNN wrapper for embedding inference on NPU."""

    def __init__(self, model_path: Path, core_mask: str = "auto", verbose: bool = False):
        self.model_path = model_path
        self.core_mask = core_mask.lower()
        self.verbose = verbose
        self._rknn = None
        self._supports_data_type = True

    def __enter__(self) -> "RKNNEmbedder":
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise ImportError(
                "rknn-toolkit-lite2 is not installed. Install it on the target device first."
            ) from exc

        rknn = RKNNLite(verbose=self.verbose)
        ret = rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed with code: {ret}")

        mask = self._resolve_core_mask(RKNNLite)
        if mask is None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(core_mask=mask)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed with code: {ret}")

        self._rknn = rknn
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def _resolve_core_mask(self, rknn_cls):
        masks = {
            "auto": "NPU_CORE_AUTO",
            "0": "NPU_CORE_0",
            "1": "NPU_CORE_1",
            "2": "NPU_CORE_2",
            "0_1_2": "NPU_CORE_0_1_2",
        }
        attr_name = masks.get(self.core_mask)
        if attr_name is None:
            raise ValueError(
                f"Unsupported core mask '{self.core_mask}'. Use one of: "
                "auto, 0, 1, 2, 0_1_2."
            )
        return getattr(rknn_cls, attr_name, None)

    def embed(self, mel: np.ndarray) -> np.ndarray:
        """
        Run a single mel chunk through RKNN.

        Args:
            mel: np.ndarray shaped [1, n_mels, T], float32
        """
        if self._rknn is None:
            raise RuntimeError("RKNN runtime is not initialized.")

        mel_input = mel.astype(np.float32)
        if self._supports_data_type:
            try:
                outputs = self._rknn.inference(inputs=[mel_input], data_type="float32")
            except TypeError:
                self._supports_data_type = False
                outputs = self._rknn.inference(inputs=[mel_input])
        else:
            outputs = self._rknn.inference(inputs=[mel_input])
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("RKNN inference returned no outputs.")

        emb = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        return _l2_normalize(emb)


def compute_embedding_from_file(
    audio_path: Path,
    embedder: RKNNEmbedder,
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
) -> np.ndarray:
    """Chunk audio, infer each chunk with RKNN, and return one normalized embedding."""
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
    for mel in mel_chunks:
        mel_np = mel.cpu().numpy().astype(np.float32)  # [1, n_mels, T]
        embeds.append(embedder.embed(mel_np))

    mean_emb = np.mean(np.stack(embeds, axis=0), axis=0).astype(np.float32)
    return _l2_normalize(mean_emb)


def recognize_speaker(
    query_emb: np.ndarray,
    enrolled: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[str | None, float]:
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
        description="Run speaker recognition using RKNN model + enrolled_speakers .pt on NPU device."
    )
    parser.add_argument("--audio-file", required=True, type=Path, help="Audio file to recognize.")
    parser.add_argument("--rknn-model", required=True, type=Path, help="Path to .rknn model file.")
    parser.add_argument("--store-path", type=Path, default=Path("enrolled_speakers.pt"))
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--core-mask", type=str, default="auto", help="NPU core mask: auto, 0, 1, 2, 0_1_2")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--no-filter", action="store_true", help="Disable silence removal.")
    parser.add_argument("--filter-top-db", type=float, default=20.0)
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=10.0,
        help="Chunk length in seconds. Default uses 10s windows.",
    )
    parser.add_argument("--chunk-overlap", type=float, default=0.5, help="Chunk overlap in seconds.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose RKNN logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.rknn_model.exists():
        raise FileNotFoundError(f"RKNN model not found: {args.rknn_model}")

    enrolled = load_enrolled_speakers(args.store_path)
    if not enrolled:
        raise ValueError("Enrollment store is empty.")

    with RKNNEmbedder(args.rknn_model, core_mask=args.core_mask, verbose=args.verbose) as embedder:
        query_emb = compute_embedding_from_file(
            audio_path=args.audio_file,
            embedder=embedder,
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

    speaker, score = recognize_speaker(query_emb, enrolled, threshold=args.threshold)
    if speaker is None:
        print(f"Rejected: best similarity {score:.4f} below threshold {args.threshold}")
    else:
        print(f"Matched speaker '{speaker}' with similarity {score:.4f} (threshold {args.threshold})")


if __name__ == "__main__":
    main()
