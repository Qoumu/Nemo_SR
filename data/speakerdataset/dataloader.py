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

        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = f.read()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict) and "speakers" in data:
            self.target_sr = int(data.get("config", {}).get("target_sr", sample_rate))
            for spk in data.get("speakers", []):
                sid = int(spk["id"])
                for clip_path in spk.get("clips", []):
                    self.items.append((clip_path, sid))
        else:
            for line in raw.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append((obj["audio_filepath"], int(obj["label"])))

        if not self.items:
            raise ValueError(f"No audio items found in {manifest_path}")

        # Filter out missing audio files early to avoid cryptic decoder errors.
        existing = []
        missing = []
        for wav_path, speaker_id in self.items:
            if Path(wav_path).exists():
                existing.append((wav_path, speaker_id))
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
        wav_path, speaker_id = self.items[idx]

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

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_sr
            )

        waveform = waveform.mean(dim=0) if waveform.dim() > 1 else waveform.squeeze(0)  # [T]
        length = waveform.shape[-1]

        return waveform, length, int(speaker_id)
