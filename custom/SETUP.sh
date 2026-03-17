#!/usr/bin/env bash
# Quick setup and deployment guide for lightweight speaker recognition

set -e

echo "================================"
echo "Lightweight Speaker Recognition"
echo "Setup Guide for Luckfox Pico Pro Max RV1106G3"
echo "================================"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    OS="unknown"
fi

echo ""
echo "=== STEP 1: Host Setup (Enrollment) ==="

# Check Python
echo -n "Checking Python 3... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ $PYTHON_VERSION"
else
    echo "✗ Python 3 not found"
    exit 1
fi

# Install enrollment dependencies
echo "Installing PyTorch and dependencies for enrollment..."
echo "This may take a few minutes..."

if [ -f "requirements_enrollment.txt" ]; then
    pip install -q -r requirements_enrollment.txt
    echo "✓ Dependencies installed"
else
    echo "✗ requirements_enrollment.txt not found"
    exit 1
fi

echo ""
echo "=== STEP 2: Generate Enrollments ==="

if [ ! -d "audio_samples" ]; then
    echo "ℹ No audio_samples directory found"
    echo "  Create one with subdirectories for each speaker:"
    echo "    audio_samples/"
    echo "    ├─ speaker1/"
    echo "    │  ├─ sample1.wav"
    echo "    │  └─ sample2.wav"
    echo "    └─ speaker2/"
    echo "       └─ sample1.wav"
else
    echo "Found audio samples directory, ready to enroll speakers"
fi

echo ""
echo "Example enrollment command:"
echo ""
echo "python3 enroll_lite.py \\"
echo "    --speaker-id alice \\"
echo '    --audio-files audio_samples/alice/*.wav \'
echo "    --model-path ECAPATDNN_protonet_model.pth \\"
echo "    --store-path enrolled_speakers.json \\"
echo "    --verbose"
echo ""

read -p "Generate enrollments now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "ECAPATDNN_protonet_model.pth" ]; then
        python3 enroll_lite.py \
            --speaker-id test_speaker \
            --audio-files audio_samples/*/*.wav 2>/dev/null || \
            echo "✗ Enrollment failed - check audio_samples directory"
    else
        echo "✗ Model file not found: ECAPATDNN_protonet_model.pth"
    fi
fi

echo ""
echo "=== STEP 3: Prepare Device Deployment ==="

# Check for required files
MISSING=0
if [ ! -f "enrolled_speakers.json" ]; then
    echo "⚠ enrolled_speakers.json not found (run enrollment first)"
    MISSING=1
fi

if [ ! -f "model.rknn" ]; then
    echo "⚠ model.rknn not found (export model first)"
    MISSING=1
fi

if [ $MISSING -eq 0 ]; then
    echo "✓ All files ready for deployment"
    
    echo ""
    echo "To deploy to device:"
    echo ""
    echo "python3 deploy.py \\"
    echo "    --host <device_ip_or_hostname> \\"
    echo "    --user root \\"
    echo "    --enrollment-file enrolled_speakers.json \\"
    echo "    --rknn-model model.rknn"
    echo ""
fi

echo ""
echo "=== Device-side Quick Start ==="
echo ""
echo "Once deployed to device, run:"
echo ""
echo "python3 recognition_rknn_lite.py \\"
echo "    --audio-file query.wav \\"
echo "    --rknn-model model.rknn \\"
echo "    --store-path enrolled_speakers.json \\"
echo "    --threshold 0.8 \\"
echo "    --verbose"
echo ""

echo ""
echo "For more details, see: LIGHTWEIGHT_README.md"
echo ""
