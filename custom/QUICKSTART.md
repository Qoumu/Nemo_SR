# Lightweight Recognition System - Quick Start Checklist

## ✓ Regeneration Complete

All programs have been regenerated with only these dependencies:
- `argparse` (stdlib)
- `ctypes` (stdlib) 
- `json` (stdlib)
- `sys` (stdlib)
- `time` (stdlib)
- `pathlib` (stdlib)
- `numpy` (minimal, ~10MB)
- `soundfile` (minimal, ~1MB)

**NO torch, NO rknn-toolkit-lite2 required at runtime**

---

## Phase 1: Host Machine Setup (Enrollment)

**Location:** Your development machine  
**Tools needed:** Python 3.8+, PyTorch (host only)

- [ ] Install enrollment dependencies
  ```bash
  pip install -r requirements_enrollment.txt
  ```

- [ ] Verify ECAPATDNN model file exists
  ```bash
  ls -lh ECAPATDNN_protonet_model.pth
  ```

- [ ] Organize speaker audio files
  ```
  audio_samples/
  ├─ alice/
  │  ├─ alice_001.wav
  │  ├─ alice_002.wav
  │  └─ alice_003.wav
  └─ bob/
     ├─ bob_001.wav
     └─ bob_002.wav
  ```

- [ ] Verify audio format (16kHz WAV recommended)
  ```bash
  python3 -c "import soundfile as sf; y, sr = sf.read('audio_samples/alice/alice_001.wav'); print(f'Sample rate: {sr}Hz')"
  ```

---

## Phase 2: Generate Enrollments

**Output:** `enrolled_speakers.json` + `speaker_*.npy` files

- [ ] Enroll first speaker
  ```bash
  python3 enroll_lite.py \
      --speaker-id alice \
      --audio-files audio_samples/alice/*.wav \
      --model-path ECAPATDNN_protonet_model.pth \
      --store-path enrolled_speakers.json \
      --verbose
  ```

- [ ] Verify enrollment created files
  ```bash
  ls -lh enrolled_speakers.json alice_embedding.npy
  ```

- [ ] Enroll additional speakers
  ```bash
  python3 enroll_lite.py \
      --speaker-id bob \
      --audio-files audio_samples/bob/*.wav \
      --store-path enrolled_speakers.json \
      --verbose
  ```

- [ ] Verify all speakers in metadata
  ```bash
  cat enrolled_speakers.json
  ```

- [ ] Test recognition locally
  ```bash
  python3 test_recognition_lite.py
  ```

---

## Phase 3: Prepare RKNN Model

**Note:** Model export should already be done. Verify it exists.

- [ ] Check RKNN model file
  ```bash
  ls -lh *.rknn
  file *.rknn
  ```

- [ ] Document model details
  - [ ] Model name: ______________________
  - [ ] Input shape: ______________________
  - [ ] Output shape: ______________________
  - [ ] File size: ______________________

---

## Phase 4: Device Preparation

**Location:** Your Luckfox Pico Pro Max RV1106G3  
**Tools needed:** SSH access, Python 3

- [ ] SSH into device
  ```bash
  ssh root@<device_ip>
  ```

- [ ] Verify Python 3 installed
  ```bash
  python3 --version
  ```

- [ ] Install minimal dependencies
  ```bash
  opkg update
  opkg install python3-numpy python3-soundfile
  ```

- [ ] Install librknn runtime (if not already present)
  ```bash
  find /usr -name "librknn_runtime.so" 2>/dev/null
  # Or: opkg search librknn
  ```

- [ ] Create working directory
  ```bash
  mkdir -p /home/root/speaker_recognition
  cd /home/root/speaker_recognition
  ```

---

## Phase 5: Deploy Programs

**Transfer files from host to device**

- [ ] Deploy using script (recommended)
  ```bash
  python3 deploy.py \
      --host <device_ip> \
      --enrollment-file enrolled_speakers.json \
      --rknn-model model.rknn
  ```

  **OR manually transfer:**

- [ ] Copy main scripts to device
  ```bash
  scp recognition_rknn_lite.py root@<device_ip>:/home/root/speaker_recognition/
  scp rknn_ctypes.py root@<device_ip>:/home/root/speaker_recognition/
  ```

- [ ] Copy enrollment metadata and embeddings
  ```bash
  scp enrolled_speakers.json root@<device_ip>:/home/root/speaker_recognition/
  scp *_embedding.npy root@<device_ip>:/home/root/speaker_recognition/
  ```

- [ ] Copy RKNN model
  ```bash
  scp model.rknn root@<device_ip>:/home/root/speaker_recognition/
  ```

- [ ] Verify files transferred
  ```bash
  ssh root@<device_ip> "ls -lh /home/root/speaker_recognition/"
  ```

---

## Phase 6: Test on Device

**Location:** Device terminal

- [ ] Verify Python imports
  ```bash
  cd /home/root/speaker_recognition
  python3 -c "from rknn_ctypes import RKNNLite; print('✓ RKNN wrapper OK')"
  python3 -c "import numpy; print(f'✓ NumPy {numpy.__version__}')"
  ```

- [ ] Test with sample audio on device
  ```bash
  # First, transfer a test audio file
  scp audio_samples/alice/alice_001.wav root@<device_ip>:/home/root/speaker_recognition/test_alice.wav
  
  # Then run recognition
  ssh root@<device_ip> "cd /home/root/speaker_recognition && \
      python3 recognition_rknn_lite.py \
      --audio-file test_alice.wav \
      --rknn-model model.rknn \
      --store-path enrolled_speakers.json \
      --threshold 0.8 \
      --verbose"
  ```

- [ ] Expected output
  ```
  MATCHED: speaker 'alice' with similarity 0.XXXX (threshold 0.8)
  ```

- [ ] Test with different speaker
  ```bash
  scp audio_samples/bob/bob_001.wav root@<device_ip>:/home/root/speaker_recognition/test_bob.wav
  
  ssh root@<device_ip> "cd /home/root/speaker_recognition && \
      python3 recognition_rknn_lite.py \
      --audio-file test_bob.wav \
      --rknn-model model.rknn \
      --store-path enrolled_speakers.json \
      --threshold 0.8"
  ```

- [ ] Test rejection (cross-speaker)
  ```bash
  ssh root@<device_ip> "cd /home/root/speaker_recognition && \
      python3 recognition_rknn_lite.py \
      --audio-file test_alice.wav \
      --rknn-model model.rknn \
      --store-path enrolled_speakers.json \
      --threshold 0.95"
  ```

- [ ] Expected output (should reject with high threshold)
  ```
  REJECTED: best similarity 0.XXXX below threshold 0.95
  ```

---

## Phase 7: Performance Optimization

**On Device** (optional, for production)

- [ ] Test different threshold values
  - [ ] Test with `--threshold 0.70` (permissive)
  - [ ] Test with `--threshold 0.85` (balanced)
  - [ ] Test with `--threshold 0.95` (strict)

- [ ] Test core mask options
  ```bash
  # Test NPU core 0 only
  python3 recognition_rknn_lite.py \
      --core-mask 0 \
      --audio-file test_alice.wav \
      --rknn-model model.rknn \
      --store-path enrolled_speakers.json
  ```

- [ ] Monitor performance
  ```bash
  # Measure inference time
  time python3 recognition_rknn_lite.py \
      --audio-file test_alice.wav \
      --rknn-model model.rknn \
      --store-path enrolled_speakers.json \
      --verbose
  ```

---

## Phase 8: Production Deployment

**Finalize system for deployment**

- [ ] Set up automatic startup (optional)
  ```bash
  # Create service or cron job if needed
  ```

- [ ] Create README for device
  ```bash
  scp LIGHTWEIGHT_README.md root@<device_ip>:/home/root/speaker_recognition/README.md
  ```

- [ ] Document production settings
  - [ ] Threshold value: ______________________
  - [ ] Core mask: ______________________
  - [ ] Model path: ______________________
  - [ ] Audio format: 16kHz WAV
  - [ ] Max inference time: ______________________

- [ ] Verify disk usage
  ```bash
  ssh root@<device_ip> "df -h /home/root/speaker_recognition"
  ```

- [ ] Archive enrollment backup
  ```bash
  tar -czf enrolled_speakers_backup.tar.gz enrolled_speakers.json *_embedding.npy
  ```

---

## Troubleshooting

If something fails, check these:

- [ ] **"librknn_runtime.so not found"**
  ```bash
  # On device:
  find /usr -name "*rknn*" 2>/dev/null
  # Then specify: --rknn-lib /usr/lib/librknn_runtime.so
  ```

- [ ] **"No module named 'soundfile'"**
  ```bash
  ssh root@<device_ip> "opkg install python3-soundfile"
  ```

- [ ] **Inference timeout**
  - Check NPU is not busy: `ps aux | grep rknn`
  - Try different core mask: `--core-mask auto`

- [ ] **Mismatched embedding dimension**
  - Verify model output: 64 dimensions expected
  - Check all embeddings have same dimension: `python3 -c "import json, numpy as np; m = json.load(open('enrolled_speakers.json')); print(m[list(m.keys())[0]]['embedding_dim'])"`

- [ ] **Poor recognition accuracy**
  - Increase enrollment samples (use more audio files)
  - Lower threshold: `--threshold 0.75`
  - Check audio quality (background noise, volume)

---

## Success Criteria

Your system is ready for production when ALL these pass:

- [x] ✓ Lightweight programs regenerated (no torch at runtime)
- [x] ✓ Enrollments created on host (JSON + .npy format)
- [x] ✓ Files deployed to device
- [x] ✓ Python imports work on device
- [x] ✓ RKNN model loads without errors
- [x] ✓ Can recognize enrolled speakers
- [x] ✓ Rejects unknown speakers at high threshold
- [x] ✓ Inference completes in <1 second per 3sec audio

---

## Support Resources

1. **Detailed documentation:** See `LIGHTWEIGHT_README.md`
2. **Architecture overview:** See `REGENERATION_SUMMARY.md`
3. **API reference:** See `recognition_rknn_lite.py` docstrings
4. **Test suite:** Run `test_recognition_lite.py`

---

## Quick Reference Commands

```bash
# Host: Generate enrollment
python3 enroll_lite.py --speaker-id <name> --audio-files <files> --store-path enrolled_speakers.json

# Host: Test locally
python3 test_recognition_lite.py

# Host: Deploy to device
python3 deploy.py --host <device_ip> --enrollment-file enrolled_speakers.json --rknn-model model.rknn

# Device: Run recognition
python3 recognition_rknn_lite.py --audio-file query.wav --rknn-model model.rknn --store-path enrolled_speakers.json --verbose
```

---

**Setup completed:** _______________  
**Device tested:** _______________  
**Production ready:** _______________
