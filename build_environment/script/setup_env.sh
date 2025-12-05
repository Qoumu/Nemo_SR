#!/bin/bash

# This script sets up the environment for the project.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [--require] [--default]

Options:
  --require    Install Python dependencies listed in requirement.txt
  --default    Install system packages and core Python dependencies
  -h, --help   Show this help message
EOF
}

INSTALL_REQUIREMENTS=false
RUN_DEFAULT_SETUP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --require)
            INSTALL_REQUIREMENTS=true
            shift
            ;;
        --default)
            RUN_DEFAULT_SETUP=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# Prefer python3, fall back to python, otherwise warn.
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    echo "Warning: Python is not available on PATH. Skipping virtual environment creation." >&2
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REQUIREMENTS_FILE="$SCRIPT_DIR/../requirement.txt"
VENV_DIR=${VENV_DIR:-$SCRIPT_DIR/../../.venv}


# Allow overriding the venv location with VENV_DIR; default to .venv.
if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment already exists at $VENV_DIR."
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR using $PYTHON_BIN."
fi

PIP_BIN="$VENV_DIR/bin/pip"

if [[ -x "$PIP_BIN" ]]; then
    PIP_CMD=("$PIP_BIN")
else
    echo "Warning: pip not found in virtual environment; falling back to $PYTHON_BIN -m pip." >&2
    PIP_CMD=("$PYTHON_BIN" -m pip)
fi

# Start install required packages
if [[ "$INSTALL_REQUIREMENTS" == true ]]; then
    if [[ -f "$REQUIREMENTS_FILE" ]]; then
        echo "Installing Python dependencies from $REQUIREMENTS_FILE..."
        "${PIP_CMD[@]}" install -r "$REQUIREMENTS_FILE"
    else
        echo "Warning: requirements file not found at $REQUIREMENTS_FILE." >&2
    fi
fi

if [[ "$RUN_DEFAULT_SETUP" == true ]]; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "Installing system packages: libsndfile1, ffmpeg..."
        run_apt=true
        APT_PREFIX=()
        if [[ $EUID -ne 0 ]]; then
            if command -v sudo >/dev/null 2>&1; then
                APT_PREFIX=(sudo)
            else
                echo "Warning: Need root privileges to install system packages but sudo is not available. Skipping." >&2
                APT_PREFIX=()
                run_apt=false
            fi
        fi

        if [[ ${run_apt:-true} == true ]]; then
            "${APT_PREFIX[@]}" apt-get update
            "${APT_PREFIX[@]}" apt-get install -y libsndfile1 ffmpeg
        fi
    else
        echo "Warning: apt-get not available; skipping system package installation." >&2
    fi

    echo "Installing base Python packages..."
    "${PIP_CMD[@]}" install Cython packaging
    "${PIP_CMD[@]}" install "nemo_toolkit[all]"
fi
