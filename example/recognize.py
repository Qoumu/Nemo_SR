from pathlib import Path
import os
import time

import faiss
import torch
import numpy as np

from ASR.model.TitaNet import TitaNet
from utils.parser import load_catalog_centroids, load_waveforms_from_json, save_catalog_json
from utils.compute import timed


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


def validate_on_unknown_speakers(
    model, unknown_waves: dict, target_sr: int, idx, labels: list, enroll: dict, threshold: float = 0.75
):
    """
    Validate model on unknown speakers.
    Counts how many queries are correctly rejected (not matched to any known speaker).
    """
    correct_rejections = 0
    total_queries = 0
    
    for speaker_id, clips in unknown_waves.items():
        print(f"\nValidating speaker {speaker_id} ({len(clips)} clips):")
        speaker_rejections = 0
        
        for i, clip_path in enumerate(clips[:10]):  # Validate on first 5 clips per speaker
            try:
                result = model.recognize(
                    query_wav=clip_path,
                    target_sr=target_sr,
                    index=idx,
                    labels=labels,
                    threshold=threshold,
                    norm_threshold=0.0,
                    cohort_size=min(len(labels), 20),
                    reference_catalog=enroll,
                    pairwise_threshold=threshold,
                )
                
                total_queries += 1
                
                # Check if query was correctly rejected (label is None or Unknown)
                if result["best_match"] is None or result["best_match"] != str(speaker_id):
                    correct_rejections += 1
                    speaker_rejections += 1
                    print(f"  Clip {i}: ✗ Unknown speaker correctly rejected")
                else:
                    print(f"  Clip {i}: ✓ Known speaker detected - matched to {result['best_match']} (conf={result['confidence']:.3f})")
            except Exception as e:
                print(f"  Clip {i}: Error - {e}")
                total_queries += 1
        
        print(f"  Speaker {speaker_id} rejection rate: {speaker_rejections}/5")
    
    if total_queries > 0:
        rejection_rate = 100 * correct_rejections / total_queries
        print(f"\n--- Validation Summary ---")
        print(f"Total unknown queries: {total_queries}")
        print(f"Correct rejections: {correct_rejections}")
        print(f"Rejection rate: {rejection_rate:.1f}%")
        print(f"Known speaker rate: {100 - rejection_rate:.1f}%")


def main():
    start = time.perf_counter()

    model = TitaNet(
        config_path="configs/lightweight_titanet.yaml",
        device="cpu",
        use_pruned_model=False,
    )  
    print(f"load titanet: {(time.perf_counter() - start):.2f}s")

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

    # # Load unknown speakers for validation
    # unknown_json = Path("data/speakers/unknown/speaker.json")
    # if unknown_json.exists():
    #     print("\n--- Validation on Unknown Speakers ---")
    #     try:
    #         _, unknown_waves, _, target_sr = load_waveforms_from_json(str(unknown_json))
    #         validate_on_unknown_speakers(model, unknown_waves, target_sr, idx, labels, enroll)
    #     except Exception as e:
    #         print(f"[WARN] Validation failed: {e}")
    
    # Test on known speaker for sanity check
    start = time.perf_counter()
    result = model.recognize(
        query_wav="data/speakerdataset/Libri-speech/1320/122612/1320-122612-0006.flac",
        target_sr=16000,
        threshold=0.75,
        reference_catalog=enroll,
    )
    print(
        "\nRecognition result:",
        f"label={result['label']} (best={result['best_match']} score={result['score']:.3f})",
    )
    print(f"run model recognition: {(time.perf_counter() - start):.2f}s")


if __name__ == "__main__":
    main()
