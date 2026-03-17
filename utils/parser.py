import json, os, warnings, numpy as np, faiss, librosa
from utils.compute import _l2norm

def load_waveforms_from_json(json_path):
    """
    Preferred JSON lines schema (one JSON object per line):
      {"audio_filepath": "path.wav", "offset": 0, "duration": 4.16, "label": "speaker_id"}

    The legacy nested schema with a top-level "speakers" list is still accepted for
    backward compatibility, but manifests should migrate to the JSONL format above.

    Returns a tuple: (first_speaker_id, waves_dict, meta_dict, target_sr)
      - waves: {speaker_id: [clip_path1, clip_path2, ...]}
      - meta:  {speaker_id: {"num_samples": int, "sr": int}}
      - target_sr: int
    """
    def _parse_manifest_lines(raw: str, default_sr: int = 16000):
        waves = {}
        meta = {}
        target_sr = default_sr
        missing = []

        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {json_path}: {e}") from e

            path = obj.get("audio_filepath")
            label = obj.get("label") or obj.get("id") or obj.get("speaker")
            if path is None or label is None:
                raise ValueError(
                    f"Manifest line {line_no} in {json_path} is missing required "
                    "audio_filepath or label fields."
                )

            path = str(path)
            label = str(label)

            sr_val = obj.get("sample_rate")
            if sr_val is not None and target_sr == default_sr:
                try:
                    target_sr = int(sr_val)
                except (TypeError, ValueError):
                    pass

            if not os.path.exists(path):
                missing.append(path)
                continue

            clips = waves.setdefault(label, [])
            clips.append(path)
            meta[label] = {"num_samples": len(clips), "sr": target_sr}

        if missing:
            warnings.warn(
                f"{len(missing)} audio files referenced in {json_path} are missing. "
                f"First missing: {missing[:3]}"
            )

        if not waves:
            raise ValueError("No speakers loaded (empty JSON or all files missing).")

        return waves, meta, target_sr

    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        cfg = None

    # JSONL manifest format
    if not (isinstance(cfg, dict) and "speakers" in cfg):
        waves, meta, target_sr = _parse_manifest_lines(raw)
        sid = next(iter(waves.keys()))
        return sid, waves, meta, target_sr

    # Legacy nested format
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

def get_speaker_count_and_clips(json_path):
    """Get statistics about speakers and clips in a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        cfg = None
    
    # Legacy nested format
    if isinstance(cfg, dict) and "speakers" in cfg:
        speakers = cfg.get("speakers", [])
        num_speakers = len(speakers)
        num_clips = sum(len(s.get("clips", [])) for s in speakers)
        return num_speakers, num_clips

    # JSONL manifest format
    speaker_counts = {}
    total_clips = 0
    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {line_no} of {json_path}: {e}") from e

        label = obj.get("label") or obj.get("id") or obj.get("speaker")
        if label is None:
            raise ValueError(f"Manifest line {line_no} in {json_path} is missing a label.")
        label = str(label)
        speaker_counts[label] = speaker_counts.get(label, 0) + 1
        total_clips += 1
    
    return len(speaker_counts), total_clips
