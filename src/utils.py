import math
import numpy as np
import librosa, faiss
import time
from contextlib import contextmanager

@contextmanager
def timed(label):
    start = time.perf_counter()
    yield
    print(f"{label}: {(time.perf_counter() - start):.2f}s")

def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-10
    return v / n

def _centroid(vectors) -> np.ndarray:
    '''
    Accepts either an iterable of embeddings or an ndarray shaped [N, D].
    Returns an L2-normalized centroid.
    '''
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("vectors must contain at least one embedding.")
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[0] == 0:
        raise ValueError("vectors must contain at least one embedding.")
    normalized = np.apply_along_axis(_l2norm, 1, arr)
    centroid = normalized.mean(axis=0)
    return _l2norm(centroid)

def centroid_enrollment(speaker_id: str, centroid: np.ndarray) -> dict[str, list[np.ndarray]]:
    '''
    Args:
        speaker_id: str, identifier for the enrolled speaker
        centroid: np.ndarray, shape [D]
    Returns:
        enroll dict: {speaker_id: [centroid]}
    '''
    if centroid.ndim != 1:
        raise ValueError("centroid must be a 1-D embedding vector.")

    norm_centroid = _l2norm(np.asarray(centroid, dtype=np.float32))
    return {speaker_id: [norm_centroid]}

def z_norm_score(score: float, cohort_scores: list[float] | np.ndarray, eps: float = 1e-6):
    '''
    Perform simple z-normalization using the provided cohort scores.

    Returns:
        norm_score: float
        mean: float
        std: float
    '''
    cohort = np.asarray(cohort_scores, dtype=np.float32)
    if cohort.size == 0:
        return score, float(score), 0.0
    mean = float(cohort.mean())
    std = float(cohort.std())
    if std < eps:
        return score - mean, mean, std
    return (score - mean) / (std + eps), mean, std

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def calibrated_confidence(
    raw_score: float,
    norm_score: float,
    *,
    second_best: float | None,
    threshold: float,
    norm_threshold: float | None,
    pairwise_similarity: float | None = None,
    pairwise_threshold: float | None = None,
    clip_min: float = 1e-6,
    clip_max: float = 1 - 1e-6,
) -> float:
    '''
    Combine raw similarity, z-normalized score, gap to the runner-up, and optional
    pairwise cosine validation into a confidence estimate.
    Returns a value in [clip_min, clip_max], skewed low when scores hover near thresholds or the margin is small.
    '''
    centered_norm = norm_score - (norm_threshold or 0.0)
    norm_term = _sigmoid(0.5 * centered_norm)

    raw_margin = raw_score - threshold
    raw_term = _sigmoid(10.0 * raw_margin)

    if second_best is None:
        gap_term = _sigmoid(5.0 * raw_margin)
    else:
        gap_term = _sigmoid(15.0 * (raw_score - second_best))

    pairwise_term = 1.0
    if pairwise_similarity is not None:
        baseline = pairwise_threshold if pairwise_threshold is not None else 0.5
        pairwise_term = _sigmoid(10.0 * (pairwise_similarity - baseline))

    confidence = norm_term * raw_term * gap_term * pairwise_term
    confidence = max(clip_min, min(clip_max, confidence))
    return float(confidence)
