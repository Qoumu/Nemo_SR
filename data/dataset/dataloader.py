import json
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    batch: list of (mel, label) pairs.
    mel:   [n_mels, T_i]
    """
    mels, labels = zip(*batch)  # tuples of length B

    # transpose to [T_i, n_mels] so pad_sequence pads along time
    mels = [m.T for m in mels]  # each: [T_i, n_mels]

    # pad to max T in batch: result [B, max_T, n_mels]
    mels_padded = pad_sequence(mels, batch_first=True)  # [B, T_max, n_mels]

    # transpose back if your model expects [B, n_mels, T]
    mels_padded = mels_padded.permute(0, 2, 1)          # [B, n_mels, T_max]

    labels = torch.stack(labels)                        # [B]

    # final batch = (inputs, targets)
    return mels_padded, labels


class SpeakerDataset(Dataset):
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
                sid = str(spk["id"])
                for clip_path in spk.get("clips", []):
                    self.items.append((clip_path, sid))
        else:
            for line in raw.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append((obj["audio_filepath"], obj["label"]))

        if not self.items:
            raise ValueError(f"No audio items found in {manifest_path}")

        for _, label in self.items:
            if label not in self.label2id:
                self.label2id[label] = len(self.label2id)

        # simple transform: waveform -> log-mel
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_mels=80
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, label_str = self.items[idx]
        speaker_id = self.label2id[label_str]

        # Load audio clip
        waveform, sr = torchaudio.load(wav_path)  # waveform: [C, T]
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_sr
            )

        # Convert waveform to log-mel spectrogram (speaker clip representation)
        with torch.no_grad():
            speaker_clip = self.mel_transform(waveform)      # [C, n_mels, time]
            speaker_clip = torch.log(speaker_clip + 1e-6)
            speaker_clip = speaker_clip.squeeze(0)           # [n_mels, time] (mono)

        # return (speaker_clip, speaker_id)
        return speaker_clip, torch.tensor(speaker_id, dtype=torch.long)
