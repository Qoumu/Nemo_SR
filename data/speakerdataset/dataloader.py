import json
import warnings
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Prefer the sox backend to avoid torchcodec dependency.
try:
    torchaudio.set_audio_backend("sox_io")
except Exception:
    pass


def collate_fn(batch):
    """
    batch: list of (waveform, length, label) tuples.
    waveform: [T]
    """
    waveforms, lengths, labels = zip(*batch)

    # Pad to max length in batch
    padded = pad_sequence(waveforms, batch_first=True)  # [B, T_max]
    lengths = torch.tensor(lengths, dtype=torch.long)   # [B]
    labels = torch.tensor(labels, dtype=torch.long)     # [B]

    # final batch = (inputs, input_lengths, targets)
    return padded, lengths, labels


class Dataloader(Dataset):
    def __init__(self, manifest_path, sample_rate=16000):
        self.items = []
        self.target_sr = sample_rate
        self.label_to_index: dict[str, int] = {}

        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = f.read()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict) and "speakers" in data:
            warnings.warn(
                "Detected legacy nested manifest format. Please migrate to JSONL with "
                '{"audio_filepath": ..., "offset": 0, "duration": ..., "label": ...}.',
                stacklevel=2,
            )
            self.target_sr = int(data.get("config", {}).get("target_sr", sample_rate))
            for spk in data.get("speakers", []):
                label = str(spk["id"])
                label_idx = self.label_to_index.setdefault(label, len(self.label_to_index))
                for clip_path in spk.get("clips", []):
                    self.items.append(
                        {
                            "audio_filepath": clip_path,
                            "label": label_idx,
                            "label_str": label,
                            "offset": 0.0,
                            "duration": None,
                        }
                    )
        else:
            for line_no, line in enumerate(raw.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no} of {manifest_path}: {e}") from e

                audio_path = obj.get("audio_filepath")
                label_str = obj.get("label") or obj.get("id") or obj.get("speaker")
                if audio_path is None or label_str is None:
                    raise ValueError(
                        f"Manifest line {line_no} in {manifest_path} is missing audio_filepath or label."
                    )

                label_str = str(label_str)
                label_idx = self.label_to_index.setdefault(label_str, len(self.label_to_index))

                # Pick up sample rate hints if present; fall back to provided default.
                if self.target_sr == sample_rate and obj.get("sample_rate") is not None:
                    try:
                        self.target_sr = int(obj["sample_rate"])
                    except (TypeError, ValueError):
                        pass

                duration = obj.get("duration")
                if duration is not None:
                    try:
                        duration = float(duration)
                    except (TypeError, ValueError):
                        duration = None

                self.items.append(
                    {
                        "audio_filepath": str(audio_path),
                        "label": label_idx,
                        "label_str": label_str,
                        "offset": float(obj.get("offset", 0.0) or 0.0),
                        "duration": duration,
                    }
                )

        if not self.items:
            raise ValueError(f"No audio items found in {manifest_path}")

        # Filter out missing audio files early to avoid cryptic decoder errors.
        existing = []
        missing = []
        for item in self.items:
            wav_path = item["audio_filepath"]
            if Path(wav_path).exists():
                existing.append(item)
            else:
                missing.append(wav_path)
        self.items = existing

        if missing:
            warnings.warn(
                f"{len(missing)} audio files referenced in {manifest_path} are missing. "
                f"First missing: {missing[:3]}"
            )
        if not self.items:
            raise FileNotFoundError(
                f"All audio files listed in {manifest_path} are missing. "
                "Download the dataset or fix the manifest paths."
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav_path = item["audio_filepath"]
        speaker_id = item["label"]
        offset = float(item.get("offset", 0.0) or 0.0)
        duration = item.get("duration")

        # Try torchaudio first; fall back to soundfile if torchcodec is unavailable.
        try:
            waveform, sr = torchaudio.load(wav_path)  # waveform: [C, T]
        except Exception:
            data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            waveform = torch.tensor(data, dtype=torch.float32)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # [1, T]
            else:
                waveform = waveform.transpose(0, 1)  # [C, T]

        # Apply offset/duration cropping before any resampling.
        start = int(round(offset * sr))
        end = None
        if duration is not None:
            try:
                end = start + int(round(float(duration) * sr))
            except (TypeError, ValueError):
                end = None

        if start > 0 or end is not None:
            if start >= waveform.shape[-1]:
                waveform = waveform[..., :0]
            else:
                waveform = waveform[..., start:end]

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_sr
            )

        waveform = waveform.mean(dim=0) if waveform.dim() > 1 else waveform.squeeze(0)  # [T]
        length = waveform.shape[-1]

        return waveform, length, int(speaker_id)
