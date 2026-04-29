#!/usr/bin/env bash
set -euo pipefail
###############################################################################
# Download two Google Drive folders:
#   1) a full model tree -> ./outputs/models/
#   2) a data folder     -> ./data/
#
# Usage: ./scripts/get_assets.sh 1ancje2FsGw9dTMMCXCO2CfuXsqgDJKBE 1OpVGWl8JlRD3G3mB8IqoAE6DY1bd08sv
###############################################################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "[DRY RUN] Repository root: ${REPO_ROOT}"
  echo "[DRY RUN] Models destination: ${REPO_ROOT}/outputs/models"
  echo "[DRY RUN] Data destination: ${REPO_ROOT}/data"
  exit 0
fi

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <MODELS_FOLDER_ID> <DATA_FOLDER_ID>" >&2
  echo "       $0 --dry-run" >&2
  exit 1
fi

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

MODELS_ID="$1"
DATA_ID="$2"
MODELS_DEST="${REPO_ROOT}/outputs/models"
DATA_DEST="${REPO_ROOT}/data"

# Install gdown automatically if it is missing
command -v gdown >/dev/null 2>&1 || {
  echo "[INFO] Installing gdown..."
  python3 -m pip install --quiet --upgrade gdown
}

download_folder () {
  local DRIVE_ID="$1"
  local DEST="$2"
  mkdir -p "${DEST}"
  echo "[INFO] Downloading Drive folder ${DRIVE_ID} -> ${DEST}"
  gdown --fuzzy --folder "https://drive.google.com/drive/folders/${DRIVE_ID}" \
        -O "${DEST}"
}

download_folder "${MODELS_ID}" "${MODELS_DEST}"
download_folder "${DATA_ID}"   "${DATA_DEST}"

echo -e "\n[SUCCESS] Models in ${MODELS_DEST}"
echo "[SUCCESS] Data   in ${DATA_DEST}"
