# Lightweight Speaker Recognition for Embedded Devices

This is a redesigned speaker recognition system optimized for embedded devices like the **Luckfox Pico Pro Max RV1106G3** with minimal dependencies and no PyTorch runtime.

## Architecture Overview

```
Host Machine (with PyTorch)          Embedded Device (RV1106G3)
                                     
enroll_lite.py                       recognition_rknn_lite.py
├─ Load model (torch)                ├─ Load embeddings (JSON + .npy)
├─ Generate embeddings               ├─ Load audio (soundfile)
└─ Save as .npy + .json             ├─ Call RKNN via ctypes
                                     └─ Match speakers (numpy)

enrolled_speakers.json (transferred)
├─ speaker_00_embedding.npy
├─ speaker_01_embedding.npy
└─ ...
```

## Lightweight Programs

### 1. `enroll_lite.py` - Host-side Enrollment
Runs on your development machine (with PyTorch) to generate and save embeddings.

**Dependencies:**
- `torch` (host only)
- `numpy`
- `soundfile` (existing utilities)

**Usage:**
```bash
# Enroll a single speaker from multiple audio files
python enroll_lite.py \
    --speaker-id alice \
    --audio-files alice_sample1.wav alice_sample2.wav alice_sample3.wav \
    --model-path ECAPATDNN_protonet_model.pth \
    --store-path enrolled_speakers.json \
    --verbose
```

**Output:**
- `enrolled_speakers.json` - Metadata file with speaker info
- `alice_embedding.npy` - Embedding file (binary, ~256 bytes for 64-dim embeddings)

### 2. `recognition_rknn_lite.py` - Embedded Device Recognition
Runs on the embedded device (no PyTorch) using RKNN runtime via ctypes.

**Dependencies (on device):**
- `argparse` (stdlib)
- `json` (stdlib)
- `ctypes` (stdlib)
- `numpy` (~10MB compiled)
- `soundfile` (~1MB)

**No dependencies required:**
- ✗ PyTorch
- ✗ rknn-toolkit-lite2
- ✓ Only RKNN runtime library (`librknn_runtime.so`)

**Usage:**
```bash
# Copy enrollment files to device first
scp enrolled_speakers.json deployed_model.rknn device:/home/root/

# On device:
python recognition_rknn_lite.py \
    --audio-file query.wav \
    --rknn-model deployed_model.rknn \
    --store-path enrolled_speakers.json \
    --threshold 0.8 \
    --verbose
```

**Output:**
```
MATCHED: speaker 'alice' with similarity 0.9234 (threshold 0.8)
```

### 3. `rknn_ctypes.py` - RKNN Runtime Wrapper
Direct ctypes binding to `librknn_runtime.so` for NPU inference.

**Features:**
- No rknn-toolkit-lite2 required (only runtime library)
- Direct C API calls via ctypes
- Support for multiple NPU cores
- Auto-detection of librknn library paths

## Data Format

### Enrollment Metadata (`enrolled_speakers.json`)
```json
{
  "alice": {
    "embedding_file": "alice_embedding.npy",
    "embedding_dim": 64,
    "dtype": "float32"
  },
  "bob": {
    "embedding_file": "bob_embedding.npy",
    "embedding_dim": 64,
    "dtype": "float32"
  }
}
```

### Embeddings (`.npy` files)
Binary NumPy format:
- Shape: `(64,)` - 1D array of 64 float32 values
- L2-normalized (norm = 1.0)
- Size: ~256 bytes per speaker

**Space efficiency:**
- Old PyTorch `.pt` format: ~1KB+ per speaker
- New `.npy` format: ~256 bytes per speaker
- JSON metadata: ~300 bytes total
- **Total for 100 speakers: ~26KB** (vs 100KB+ with .pt)

## Setup Instructions

### 1. Host Machine Setup (Enrollment)
```bash
# Install dependencies
pip install torch numpy soundfile

# Prepare your audio files (16kHz, WAV format recommended)
ls ~/data/speaker_audio/

# Create enrollments
python enroll_lite.py \
    --speaker-id alice \
    --audio-files ~/data/speaker_audio/alice/*.wav \
    --store-path enrolled_speakers.json \
    --verbose

# Repeat for each speaker
```

### 2. Device Preparation
```bash
# Install minimal dependencies on device
opkg install python3-numpy python3-soundfile

# Copy enrollment files to device
scp enrolled_speakers.json user@device:/home/root/
scp alice_embedding.npy user@device:/home/root/
# ... copy all embeddings

# Copy RKNN model (already exported)
scp model.rknn user@device:/home/root/

# Copy recognition script
scp recognition_rknn_lite.py user@device:/home/root/
scp rknn_ctypes.py user@device:/home/root/
```

### 3. Run Recognition on Device
```bash
# SSH into device
ssh user@device

cd /home/root

# Test with sample audio
python recognition_rknn_lite.py \
    --audio-file test_speaker.wav \
    --rknn-model model.rknn \
    --store-path enrolled_speakers.json \
    --threshold 0.8
```

## Compatibility

### Tested Devices
- ✓ Luckfox Pico Pro Max RV1106G3 (32-bit Linux)
- ✓ RV1109/RV1126 boards
- ✓ Other Rockchip NPU devices

### Python Versions
- Python 3.8+
- Both 32-bit and 64-bit architectures

### Audio Format
- WAV files (recommended)
- 16kHz sample rate
- Mono or stereo (auto-converted to mono)

## Performance Notes

### Inference Speed
- Per audio chunk: ~100-200ms on RV1106G3 (depending on chunk size)
- Pre-processing (mel-spectrogram): ~50-100ms
- Total for 3-second audio: ~500-1000ms

### Memory Usage
- Runtime script: ~30-50MB (Python + numpy)
- Loaded model: ~10-20MB (RKNN runtime)
- Total system: **under 100MB RAM**

## Troubleshooting

### "librknn_runtime.so not found"
```bash
# Find the library location
find /usr -name "librknn_runtime.so" 2>/dev/null

# Use explicit path
python recognition_rknn_lite.py \
    --rknn-lib /usr/lib/librknn_runtime.so \
    ...
```

### "No module named 'soundfile'"
```bash
pip install soundfile
# Or on device:
opkg install python3-soundfile
```

### RKNN inference returns no outputs
- Check RKNN model is valid: `file model.rknn`
- Verify core mask option: try `--core-mask auto`
- Check NPU is not busy: `ps aux | grep rknn`

## Testing

Run the test suite to verify the lightweight system:
```bash
python test_recognition_lite.py
```

This creates mock speakers and tests the recognition pipeline without requiring:
- Real audio files
- Real RKNN device
- PyTorch

## Migration from Old System

### Old System (original recognition.py)
```
- Uses torch.load() to load .pt files
- Requires PyTorch at runtime on device
- Large model/embedding files
- GPU-dependent (no NPU support)
```

### New System (recognition_rknn_lite.py)
```
- Uses json.load() + numpy.load()
- No PyTorch at runtime
- Minimal file sizes
- NPU-optimized via RKNN
```

### Migration Steps
1. Run `enroll_lite.py` on host with existing model
2. This generates `.npy` + `.json` files
3. Copy to device
4. Use `recognition_rknn_lite.py` for inference
5. Old `.pt` files can be deleted

## API Reference

### `recognition_rknn_lite.py --help`
```
positional arguments:
  None

optional arguments:
  --audio-file PATH          Audio file to recognize (required)
  --rknn-model PATH          Path to .rknn model (required)
  --store-path PATH          Path to enrollment metadata (default: enrolled_speakers.json)
  --threshold FLOAT          Similarity threshold 0-1 (default: 0.8)
  --core-mask {auto,0,1,2,0_1_2}  NPU cores to use (default: auto)
  --sr INT                   Sample rate (default: 16000)
  --n-mels INT               Mel bins (default: 80)
  --n-fft INT                FFT size (default: 1024)
  --hop-length INT           Hop length (default: 256)
  --chunk-duration FLOAT     Seconds per chunk (default: 3.0)
  --rknn-lib PATH            Path to librknn_runtime.so (auto-detect if omitted)
  --verbose                  Enable debug output
```

## License & Credits

This lightweight version is designed for the Luckfox Pico Pro Max RV1106G3 platform.

Original system based on ECAPA-TDNN speaker embedder + Prototypical Networks.
Lightweight version uses minimal dependencies (numpy, soundfile, ctypes only).

## Support

For issues or questions:
1. Check troubleshooting section above
2. Enable verbose output: `--verbose`
3. Check RKNN library installation on device
4. Verify all `.json` + `.npy` files are present
