import json, os, numpy as np, faiss, librosa
from utils import _l2norm

def load_waveforms_from_json(json_path):
    """
    JSON schema:
    {
      "speakers": [
        {"id": "alice", "clips": ["data/speakers/alice_1.wav", "data/speakers/alice_2.wav"]},
        {"id": "bob",   "clips": ["data/speakers/bob_1.wav",   "data/speakers/bob_2.wav"]}
      ],
      "config": {"target_sr": 16000}
    }

    Returns:
      waves: {speaker_id: [np.ndarray wave1, wave2, ...]}  (all mono @ target_sr)
      meta:  {speaker_id: {"num_samples": int, "sr": int}}
      target_sr: int
    """
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    target_sr = int(cfg.get("config", {}).get("target_sr", 16000))
    waves = {}
    meta  = {}
    sid = None

    for spk in cfg.get("speakers", []):
        sid = spk["id"]
        clips = spk.get("clips", [])
        arrs = []
        
        for p in clips:
            if not os.path.exists(p):
                print(f"[WARN] missing file: {p} (skipped)")
                continue
            
            arrs.append(p)
            
        if not arrs:
            print(f"[WARN] no usable clips for {sid}, skipped")
            continue
        
        waves[sid] = arrs
        meta[sid] = {"num_samples": len(arrs), "sr": target_sr}

    if not waves:
        raise ValueError("No speakers loaded (empty JSON or all files missing).")

    return sid, waves, meta, target_sr

def save_catalog_json(path, enroll_dict, meta, examples=None):
    """
    examples: optional dict {speaker_id: ["path1.wav", "path2.wav", ...]}
    Stores per-speaker centroid for fast reload (no recompute).
    """
    out = {"speakers": []}
    for sid, centroid in enroll_dict.items():
        norm_centroid = _l2norm(np.asarray(centroid, dtype=np.float32))
        item = {
            "id": sid,
            "centroid": norm_centroid.tolist(),
            "num_samples": meta.get(sid, {}).get("num_samples", None),
            "source": meta.get(sid, {}).get("source", None),
        }
        if examples and sid in examples:
            item["examples"] = examples[sid]
        out["speakers"].append(item)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        
def load_catalog_centroids(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    enroll = {}
    for spk in data.get("speakers", []):
        sid = spk["id"]
        c = np.array(spk["centroid"], dtype=np.float32)
        enroll[sid] = _l2norm(c)
    return enroll
