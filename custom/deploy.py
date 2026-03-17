#!/usr/bin/env python3
"""
Deployment helper for transferring speaker recognition system to embedded device.
Handles file sync, verification, and setup.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_cmd(cmd: list, check: bool = True, quiet: bool = False) -> str:
    """Run shell command and return output."""
    if not quiet:
        print(f"$ {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return result.stdout.strip()


def check_device_connectivity(host: str, user: str = "root") -> bool:
    """Test SSH connectivity to device."""
    print(f"Checking connectivity to {user}@{host}...", end=" ")
    try:
        run_cmd(["ssh", f"{user}@{host}", "echo", "OK"], quiet=True)
        print("✓ Connected")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed")
        return False


def check_python_on_device(host: str, user: str = "root") -> Optional[str]:
    """Check if Python 3 is available on device."""
    print(f"Checking Python on device...", end=" ")
    try:
        version = run_cmd(
            ["ssh", f"{user}@{host}", "python3", "--version"],
            quiet=True
        )
        print(f"✓ {version}")
        return version
    except subprocess.CalledProcessError:
        print("✗ Python 3 not found")
        return None


def check_dependencies_on_device(host: str, user: str = "root") -> bool:
    """Check if numpy and soundfile are available on device."""
    print("Checking dependencies on device...")
    
    deps = ["numpy", "soundfile"]
    missing = []
    
    for dep in deps:
        print(f"  {dep}...", end=" ")
        try:
            run_cmd(
                ["ssh", f"{user}@{host}", "python3", "-c", f"import {dep}"],
                quiet=True
            )
            print("✓")
        except subprocess.CalledProcessError:
            print("✗")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠ Missing: {', '.join(missing)}")
        print(f"  Install with: opkg install {' '.join(['python3-' + d for d in missing])}")
        return False
    
    return True


def transfer_file(
    local_path: Path,
    device_path: str,
    host: str,
    user: str = "root",
    force: bool = False,
) -> bool:
    """Transfer file to device via SCP."""
    remote = f"{user}@{host}:{device_path}"
    
    print(f"  {local_path.name} -> {device_path}...", end=" ")
    
    if not local_path.exists():
        print(f"✗ Not found")
        return False
    
    try:
        run_cmd(["scp", str(local_path), remote], quiet=True)
        size_kb = local_path.stat().st_size / 1024
        print(f"✓ ({size_kb:.1f}KB)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed")
        return False


def deploy_system(
    host: str,
    device_path: str = "/home/root/speaker_recognition",
    enrollment_file: Path = Path("enrolled_speakers.json"),
    rknn_model: Path = Path("model.rknn"),
    user: str = "root",
    force: bool = False,
) -> bool:
    """Deploy complete recognition system to device."""
    
    print("=" * 70)
    print("Deploying Lightweight Speaker Recognition to Device")
    print("=" * 70)
    
    # Step 1: Connectivity
    print("\n[1] Device Connectivity")
    if not check_device_connectivity(host, user):
        return False
    
    # Step 2: Python
    print("\n[2] Python Installation")
    if not check_python_on_device(host, user):
        print("✗ Python 3 is required on device")
        return False
    
    # Step 3: Dependencies
    print("\n[3] Required Dependencies")
    if not check_dependencies_on_device(host, user):
        print("⚠ Install dependencies and retry, or use --skip-dep-check")
        if not input("\nContinue anyway? (y/n): ").lower().startswith('y'):
            return False
    
    # Step 4: Create directory on device
    print(f"\n[4] Setup")
    print(f"Creating directory on device: {device_path}", end=" ")
    try:
        run_cmd(
            ["ssh", f"{user}@{host}", "mkdir", "-p", device_path],
            quiet=True
        )
        print("✓")
    except subprocess.CalledProcessError:
        print("✗")
        return False
    
    # Step 5: Transfer files
    print(f"\n[5] File Transfer")
    print("Script files:")
    
    files_to_transfer = [
        (Path("recognition_rknn_lite.py"), f"{device_path}/recognition_rknn_lite.py"),
        (Path("rknn_ctypes.py"), f"{device_path}/rknn_ctypes.py"),
    ]
    
    for local, remote in files_to_transfer:
        transfer_file(local, remote, host, user, force)
    
    print("\nEnrollment files:")
    
    # Transfer enrollment metadata
    if enrollment_file.exists():
        transfer_file(enrollment_file, f"{device_path}/{enrollment_file.name}", host, user, force)
        
        # Transfer all embedding files
        with open(enrollment_file, 'r') as f:
            metadata = json.load(f)
        
        for speaker_id, info in metadata.items():
            emb_file = enrollment_file.parent / info['embedding_file']
            transfer_file(emb_file, f"{device_path}/{info['embedding_file']}", host, user, force)
    else:
        print(f"  ⚠ Enrollment file not found: {enrollment_file}")
    
    # Transfer RKNN model
    print("\nModel files:")
    if rknn_model.exists():
        transfer_file(rknn_model, f"{device_path}/{rknn_model.name}", host, user, force)
    else:
        print(f"  ⚠ RKNN model not found: {rknn_model}")
    
    # Step 6: Verification
    print(f"\n[6] Verification")
    print("Listing deployed files:", end=" ")
    try:
        files = run_cmd(
            ["ssh", f"{user}@{host}", "ls", "-lh", device_path],
            quiet=True
        )
        print("✓")
        print(files)
    except subprocess.CalledProcessError:
        print("✗")
    
    # Step 7: Quick test
    print(f"\n[7] Quick Test")
    print("Testing Python import...", end=" ")
    try:
        run_cmd(
            ["ssh", f"{user}@{host}", 
             f"cd {device_path} && python3 -c 'from rknn_ctypes import RKNNLite; print(\"OK\")'"],
            quiet=True
        )
        print("✓")
    except subprocess.CalledProcessError:
        print("✗ Import failed")
        return False
    
    print("\n" + "=" * 70)
    print("✓ Deployment Complete!")
    print("=" * 70)
    print(f"\nTo run recognition on device:")
    print(f"  ssh {user}@{host} 'cd {device_path} && \\")
    print(f"      python3 recognition_rknn_lite.py \\")
    print(f"      --audio-file your_audio.wav \\")
    print(f"      --rknn-model {rknn_model.name} \\")
    print(f"      --store-path {enrollment_file.name} \\")
    print(f"      --verbose'")
    
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy speaker recognition system to embedded device"
    )
    parser.add_argument(
        "--host",
        required=True,
        help="Device hostname or IP address"
    )
    parser.add_argument(
        "--user",
        default="root",
        help="SSH username (default: root)"
    )
    parser.add_argument(
        "--device-path",
        default="/home/root/speaker_recognition",
        help="Installation path on device"
    )
    parser.add_argument(
        "--enrollment-file",
        type=Path,
        default=Path("enrolled_speakers.json"),
        help="Enrollment metadata file"
    )
    parser.add_argument(
        "--rknn-model",
        type=Path,
        default=Path("model.rknn"),
        help="RKNN model file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files"
    )
    parser.add_argument(
        "--skip-dep-check",
        action="store_true",
        help="Skip dependency verification"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    success = deploy_system(
        host=args.host,
        device_path=args.device_path,
        enrollment_file=args.enrollment_file,
        rknn_model=args.rknn_model,
        user=args.user,
        force=args.force,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
