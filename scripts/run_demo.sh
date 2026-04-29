#!/usr/bin/env bash

# Exit on error
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "--dry-run" ]]; then
    test -f "src/llmvsslm/app.py"
    echo "[DRY RUN] Streamlit entrypoint: ${REPO_ROOT}/src/llmvsslm/app.py"
    echo "[DRY RUN] Model directory: ${LMVSSLM_MODEL_DIR:-${REPO_ROOT}/outputs/models}"
    exit 0
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment 'venv' created."
    # Activate venv
    source venv/bin/activate
    echo "Virtual environment 'venv' activated."
    # Upgrade pip
    pip install --upgrade pip
    echo "Pip upgraded."
    # install uv
    echo "Installing uv... (for faster package installation)"
    pip install uv -q
    echo "uv installed"
    # Clean up previous installations
    echo "Cleaning up previous installations..."
    uv pip install pip3-autoremove
    pip-autoremove torch torchvision torchaudio -y
    # Install requirements
    echo "Installing requirements..."
    uv pip install -r requirements.txt -q
    echo "Requirements installed."

fi

# Activate venv
source venv/bin/activate
echo "Virtual environment 'venv' activated."


echo "Running Streamlit app..."
# Run Streamlit app
streamlit run src/llmvsslm/app.py --server.port=8888