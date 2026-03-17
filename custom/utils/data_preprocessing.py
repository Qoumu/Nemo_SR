import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import librosa

def audio_to_mel_spectrogram(
    audio_path: str | Path | None = None,
    *,
    y: Optional[np.ndarray] = None,
    sr: int = 16000,
    offset: float = 0.0,
    duration: Optional[float] = None,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
    to_db: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load an audio file segment and convert it to a Mel spectrogram.

    Returns:
        mel: (n_mels, time) float32 ndarray (dB if to_db=True else power mel)
        y: waveform float32 (n_samples,)
        sr: sample rate (int)
    """
    if y is None:
        if audio_path is None:
            raise ValueError("Provide either audio_path or y.")
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr_loaded = librosa.load(
            str(audio_path),
            sr=sr,               # resample to target sr
            mono=True,
            offset=offset,
            duration=duration,
        )
    else:
        if y.ndim != 1:
            raise ValueError("Waveform y must be 1D.")
        sr_loaded = sr

    if y.size == 0:
        raise ValueError("Loaded audio is empty. Check offset/duration and file content.")

    if win_length is None:
        win_length = n_fft
    if fmax is None:
        fmax = sr_loaded / 2.0

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr_loaded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )

    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)

    return mel.astype(np.float32), y.astype(np.float32), sr_loaded

def filter_audio(
    y: np.ndarray,
    sr: int,
    *,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    pad_ms: float = 80.0,
    min_chunk_ms: float = 200.0,
    merge_gap_ms: float = 120.0,
) -> np.ndarray:
    """
    Remove low-energy (quiet) parts of an audio signal and concatenate the rest.

    Args:
        y: 1D waveform (n_samples,)
        sr: sample rate
        top_db: silence threshold (lower = more aggressive)
        frame_length, hop_length: analysis parameters
        pad_ms: padding added around kept regions
        min_chunk_ms: discard very short kept chunks
        merge_gap_ms: merge kept chunks if gap is small

    Returns:
        y_filtered: concatenated waveform containing only informative audio
    """
    if y.ndim != 1 or y.size == 0:
        return np.zeros(0, dtype=y.dtype)

    intervals = librosa.effects.split(
        y,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    if len(intervals) == 0:
        return np.zeros(0, dtype=y.dtype)

    pad = int(sr * pad_ms / 1000)
    min_len = int(sr * min_chunk_ms / 1000)
    merge_gap = int(sr * merge_gap_ms / 1000)

    # pad + filter tiny chunks
    padded = []
    for s, e in intervals:
        s = max(0, s - pad)
        e = min(len(y), e + pad)
        if e - s >= min_len:
            padded.append([s, e])

    if not padded:
        return np.zeros(0, dtype=y.dtype)

    # merge close chunks
    padded.sort(key=lambda x: x[0])
    merged = [padded[0]]
    for s, e in padded[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # concatenate
    return np.concatenate([y[s:e] for s, e in merged], axis=0)

class SpeakerDataset(Dataset):
    def __init__(
        self,
        dataset_list: List[dict],
        sr: int = 16000,
        n_mels: int = 80,
        duration: float = 3.0,
        augment: bool = False,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        apply_filter: bool = True,
        filter_top_db: float = 30.0,
    ):
        """
        Args:
            dataset_list: list of dicts with keys: 'audio_filepath', 'label', 'label_id', 'split'
            sr: audio sample rate
            n_mels: number of mel filterbanks
            duration: fixed duration in seconds (will pad/crop)
            augment: whether to apply augmentation
            n_fft: FFT window size
            hop_length: hop length for STFT
            win_length: window length (defaults to n_fft)
            fmin: minimum frequency
            fmax: maximum frequency (defaults to sr/2)
            apply_filter: whether to apply silence removal filter
            filter_top_db: silence threshold for filter (lower = more aggressive)
        """
        self.dataset_list = dataset_list
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.augment = augment
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr / 2.0
        self.apply_filter = apply_filter
        self.filter_top_db = filter_top_db

    def __len__(self):
        return len(self.dataset_list)

    def _augment_waveform(self, y: np.ndarray) -> np.ndarray:
        """Apply time-domain augmentation to waveform"""
        if not self.augment:
            return y

        # Random gain (volume)
        if np.random.rand() > 0.5:
            gain = np.random.uniform(0.8, 1.2)
            y = y * gain

        # Add small noise
        if np.random.rand() > 0.5:
            noise = np.random.randn(len(y)) * 0.005
            y = y + noise

        # Time shift
        if np.random.rand() > 0.5:
            shift = np.random.randint(-self.sr // 10, self.sr // 10)
            y = np.roll(y, shift)

        return y

    def _pad_or_crop_mel(self, mel: np.ndarray) -> np.ndarray:
        """
        Pad or crop mel spectrogram to fixed time dimension

        Args:
            mel: (n_mels, time) array

        Returns:
            mel: (n_mels, target_time) array
        """
        target_frames = int(self.duration * self.sr / self.hop_length)
        current_frames = mel.shape[1]

        if current_frames > target_frames:
            # Random crop for augmentation, center crop for validation
            if self.augment:
                start = np.random.randint(0, current_frames - target_frames + 1)
            else:
                start = (current_frames - target_frames) // 2
            mel = mel[:, start:start + target_frames]
        elif current_frames < target_frames:
            # Pad with minimum value (for dB scale)
            pad_width = target_frames - current_frames
            pad_value = mel.min()
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=pad_value)

        return mel

    def __getitem__(self, idx):
        item = self.dataset_list[idx]
        audio_path = item['audio_filepath']
        label_id = item['label_id']

        # Load audio and convert to mel spectrogram using your function
        mel, y, sr = audio_to_mel_spectrogram(
            audio_path=audio_path,
            sr=self.sr,
            duration=None,  # Load entire file first
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            to_db=True,
        )

        # Apply silence removal filter (removes quiet/non-speech parts)
        if self.apply_filter:
            y = filter_audio(
                y,
                sr,
                top_db=self.filter_top_db,
                frame_length=2048,
                hop_length=512,
                pad_ms=80.0,
                min_chunk_ms=200.0,
                merge_gap_ms=120.0,
            )

            # If filtering removed everything, skip filtering
            if y.size == 0:
                # Reload without filtering
                mel, y, sr = audio_to_mel_spectrogram(
                    audio_path=audio_path,
                    sr=self.sr,
                    duration=None,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    n_mels=self.n_mels,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    to_db=True,
                )

        # Apply waveform augmentation and recompute mel if needed
        if self.augment:
            y = self._augment_waveform(y)

        # Recompute mel spectrogram with filtered/augmented waveform
        mel, _, _ = audio_to_mel_spectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            to_db=True,
        )

        # Pad or crop to fixed duration
        mel = self._pad_or_crop_mel(mel)

        # Normalize (per-sample normalization)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        # Convert to torch tensor: (n_mels, time)
        mel_tensor = torch.from_numpy(mel).float()

        return mel_tensor, label_id
    
def _pad_or_crop_mel_chunk(
    mel: np.ndarray, *, target_frames: int, pad_value: float
) -> np.ndarray:
    """Pad or center-crop mel spectrogram to the target number of frames."""
    current_frames = mel.shape[1]
    if current_frames > target_frames:
        start = (current_frames - target_frames) // 2
        mel = mel[:, start : start + target_frames]
    elif current_frames < target_frames:
        pad_width = target_frames - current_frames
        mel = np.pad(
            mel,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=pad_value,
        )
    return mel


def audio_chunking(
    y: np.ndarray,
    sr: int,
    chunk_duration: float = 10.0,
    overlap_duration: float = 0.5,
    *,
    return_mels: bool = False,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    target_duration: float | None = None,
    apply_filter: bool = False,
    filter_top_db: float = 20.0,
) -> List[np.ndarray | torch.Tensor]:
    """
    Split audio waveform into overlapping chunks. Optionally convert each chunk
    into a model-ready mel tensor shaped [1, n_mels, T].

    Args:
        y: 1D waveform (n_samples,)
        sr: sample rate
        chunk_duration: duration of each chunk in seconds
        overlap_duration: overlap between chunks in seconds
        return_mels: when True, return normalized mel tensors instead of waveforms
        n_mels, n_fft, hop_length: mel/STFT parameters (used when return_mels=True)
        target_duration: duration (sec) to pad/crop mel chunks to. Defaults to
            chunk_duration when None.
        apply_filter: remove silence before chunking using filter_audio
        filter_top_db: silence threshold passed to filter_audio

    Returns:
        List of waveform arrays or mel tensors (each [1, n_mels, T]).
    """
    if apply_filter:
        y = filter_audio(
            y,
            sr,
            top_db=filter_top_db,
            frame_length=2048,
            hop_length=512,
            pad_ms=80.0,
            min_chunk_ms=200.0,
            merge_gap_ms=120.0,
        )
    if y.size == 0:
        return []

    chunk_size = int(chunk_duration * sr)
    overlap_size = int(overlap_duration * sr)
    step_size = max(1, chunk_size - overlap_size)

    wave_chunks: list[np.ndarray] = []
    for start in range(0, len(y), step_size):
        end = start + chunk_size
        chunk = y[start:end]
        if len(chunk) < chunk_size:
            pad_width = chunk_size - len(chunk)
            chunk = np.pad(chunk, (0, pad_width), mode="constant", constant_values=0)
        wave_chunks.append(chunk)
        if end >= len(y):
            break

    if not return_mels:
        return wave_chunks

    mels: list[torch.Tensor] = []
    target_dur = target_duration if target_duration is not None else chunk_duration
    target_frames = int(target_dur * sr / hop_length)

    for chunk in wave_chunks:
        mel, _, _ = audio_to_mel_spectrogram(
            y=chunk,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            to_db=True,
        )
        mel = _pad_or_crop_mel_chunk(mel, target_frames=target_frames, pad_value=mel.min())
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mels.append(torch.from_numpy(mel).float().unsqueeze(0))  # [1, n_mels, T]

    return mels
