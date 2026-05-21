#!/usr/bin/env bash
#
# media-analyzer setup script for macOS
# Creates a virtual environment and installs all dependencies.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== media-analyzer setup ==="
echo ""

# Check Python version
PYTHON="${PYTHON:-python3}"
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "Error: Python 3.10+ required (found $PY_VERSION)"
    echo "Install via: brew install python@3.12"
    exit 1
fi
echo "✓ Python $PY_VERSION"

# Check ffmpeg / ffprobe
if ! command -v ffprobe &>/dev/null; then
    echo ""
    echo "ffprobe not found. Installing ffmpeg via Homebrew..."
    if command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "Error: ffmpeg is required. Install via: brew install ffmpeg"
        exit 1
    fi
fi
echo "✓ ffmpeg/ffprobe available"

# Create venv
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi
echo "✓ Virtual environment at .venv/"

# Activate and install
source "$VENV_DIR/bin/activate"

echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -e "$SCRIPT_DIR" -q

# Verify critical imports
echo ""
echo "Verifying installation..."
"$VENV_DIR/bin/python" -c "
import cv2; print(f'  ✓ OpenCV {cv2.__version__}')
import librosa; print(f'  ✓ librosa {librosa.__version__}')
import numpy; print(f'  ✓ NumPy {numpy.__version__}')
import soundfile; print(f'  ✓ soundfile {soundfile.__version__}')
" || {
    echo "ERROR: Some dependencies failed to install."
    echo "Try: pip install opencv-python-headless librosa soundfile numpy scipy"
    exit 1
}

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run:"
echo "  media-analyzer audio <file.mp3>"
echo "  media-analyzer video <file.mp4>"
echo "  media-analyzer batch <directory> -r"
echo ""
