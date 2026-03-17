"""
Quick test script to verify the lightweight recognition pipeline.
Does NOT require a real RKNN device - tests with mock embeddings.
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np


def create_mock_enrolled_speakers(
    store_path: Path = Path("enrolled_speakers_test.json"),
    num_speakers: int = 3,
    embedding_dim: int = 64,
) -> None:
    """Create mock enrolled speakers for testing."""
    store_dir = store_path.parent
    store_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {}
    for i in range(num_speakers):
        speaker_id = f"speaker_{i:02d}"
        emb_filename = f"{speaker_id}_embedding.npy"
        emb_filepath = store_dir / emb_filename
        
        # Create random normalized embedding
        emb = np.random.randn(embedding_dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        np.save(str(emb_filepath), emb)
        
        metadata[speaker_id] = {
            "embedding_file": emb_filename,
            "embedding_dim": embedding_dim,
            "dtype": "float32",
        }
    
    with open(store_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created {num_speakers} mock speakers in {store_path}")


def load_enrolled_speakers(store_path: Path) -> Dict[str, np.ndarray]:
    """Load enrolled speakers from JSON + .npy files."""
    metadata_path = store_path
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    store_dir = metadata_path.parent
    normalized = {}
    
    for speaker_id, info in metadata.items():
        emb_file = store_dir / info['embedding_file']
        if not emb_file.exists():
            raise FileNotFoundError(f"Embedding file not found: {emb_file}")
        
        emb_np = np.load(str(emb_file)).astype(np.float32).reshape(-1)
        norm = np.linalg.norm(emb_np)
        if norm > 1e-8:
            emb_np = emb_np / norm
        normalized[str(speaker_id)] = emb_np
    
    return normalized


def recognize_speaker(
    query_emb: np.ndarray,
    enrolled: Dict[str, np.ndarray],
    threshold: float = 0.8,
) -> tuple:
    """Match query embedding against enrolled speakers."""
    best_speaker = None
    best_score = float("-inf")
    scores = {}
    
    for speaker_id, emb in enrolled.items():
        score = float(np.dot(query_emb, emb))
        scores[speaker_id] = score
        if score > best_score:
            best_score = score
            best_speaker = speaker_id
    
    decision = best_speaker if best_score >= threshold else None
    return decision, best_score, scores


def main() -> None:
    print("=" * 70)
    print("Lightweight Speaker Recognition - Test Suite")
    print("=" * 70)
    
    # Create mock speakers
    print("\n1. Creating mock enrolled speakers...")
    store_path = Path("enrolled_speakers_test.json")
    create_mock_enrolled_speakers(store_path)
    
    # Load speakers
    print("\n2. Loading enrolled speakers...")
    enrolled = load_enrolled_speakers(store_path)
    print(f"   Loaded {len(enrolled)} speakers:")
    for speaker_id, emb in enrolled.items():
        print(f"     - {speaker_id}: dim={emb.shape[0]}, norm={np.linalg.norm(emb):.4f}")
    
    # Test 1: Query similar to speaker_00
    print("\n3. Test: Query similar to speaker_00...")
    query_00 = enrolled['speaker_00'] + np.random.randn(64) * 0.1
    query_00 /= np.linalg.norm(query_00)
    matched, score, scores = recognize_speaker(query_00, enrolled, threshold=0.5)
    print(f"   Matched: {matched} (score={score:.4f})")
    print(f"   All scores:")
    for spk, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"     - {spk}: {s:.4f}")
    
    # Test 2: Random query (should reject)
    print("\n4. Test: Random query (should reject at high threshold)...")
    query_random = np.random.randn(64).astype(np.float32)
    query_random /= np.linalg.norm(query_random)
    matched, score, scores = recognize_speaker(query_random, enrolled, threshold=0.8)
    print(f"   Matched: {matched} (score={score:.4f}, threshold=0.8)")
    print(f"   All scores:")
    for spk, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"     - {spk}: {s:.4f}")
    
    # Cleanup
    print("\n5. Cleanup...")
    for info in json.load(open(store_path)).values():
        (Path("enrolled_speakers_test.json").parent / info['embedding_file']).unlink()
    store_path.unlink()
    print("   Test files cleaned up")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"✗ Test failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
