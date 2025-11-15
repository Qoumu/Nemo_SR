from pathlib import Path
import os
import time

import faiss
import numpy as np

from TitaNet import TitaNet
from parser import load_catalog_centroids, load_waveforms_from_json, save_catalog_json
from utils import timed


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no"}


def enroll_speakers(model, json_path, output_json_path):
    '''
    Args:
        json_path: str, path to input JSON file with waveform paths
        output_json_path: str, path to output JSON file to save catalog
    '''
    # Load waveforms from JSON
    sid, waves, meta, target_sr = load_waveforms_from_json(json_path)
    enroll_dict = {}

    for speaker_id, clips in waves.items():
        print(f"Processing speaker: {speaker_id} with {len(clips)} clips")
        centroid = model.batch_wav_to_centroid(clips, target_sr=target_sr)
        enroll_dict[speaker_id] = centroid

    # Save catalog JSON with centroids
    save_catalog_json(output_json_path, enroll_dict, meta)
    print(f"Enrollment catalog saved to {output_json_path}")


def build_faiss(enroll: dict[str, np.ndarray]):
    labels = list(enroll.keys())
    mat = np.stack([enroll[k] for k in labels]).astype("float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])  # cosine if inputs are L2-normalized
    index.add(mat)
    return index, labels


def main():
    start = time.perf_counter()

    model = TitaNet(
        config_path="configs/lightweight_titanet.yaml",
        device="cpu",
        use_pruned_model=True,
    )
    print(f"load titan et: {(time.perf_counter() - start):.2f}s")

    known_json = Path("data/speakers/known/speaker.json")
    known_centroids_path = Path("data/speakers/known/centroids.json")

    # Ensure known speaker centroids exist (recompute when file missing or empty)
    if not known_centroids_path.exists() or known_centroids_path.stat().st_size == 0:
        enroll_speakers(model, known_json, known_centroids_path)

    enroll = load_catalog_centroids(known_centroids_path)
    if not enroll:
        raise RuntimeError(f"No centroids loaded from {known_centroids_path}.")

    with timed("build faiss index"):
        idx, labels = build_faiss(enroll)

    start = time.perf_counter()
    cohort = min(len(labels), 20)
    result = model.recognize(
        query_wav="data/dataset/Libri-speech/7127/75946/7127-75946-0016.flac",
        target_sr=16000,
        index=idx,
        labels=labels,
        threshold=0.75,
        norm_threshold=0.0,
        cohort_size=cohort,
        reference_catalog=enroll,
        pairwise_threshold=0.75,
    )
    print(
        "Recognition result:",
        f"label={result['label']} (best={result['best_match']} conf={result['confidence']:.3f})",
    )
    print(f"run model recognition: {(time.perf_counter() - start):.2f}s")


if __name__ == "__main__":
    main()
