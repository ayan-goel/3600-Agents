#!/bin/bash

#SBATCH --job-name=dontique-train
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=06:00:00
#SBATCH -o dontique-train-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module load python/3.10

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv_dontique"

if [[ ! -d "$VENV_PATH" ]]; then
    echo "[INFO] Creating virtual environment at $VENV_PATH"
    python3.10 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip
    pip install -r "$PROJECT_ROOT/requirements.txt"
else
    echo "[INFO] Using existing virtual environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

export PYTHONPATH="$PROJECT_ROOT/engine:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

echo "[INFO] Starting Dontique self-play training run"
python -u "$SCRIPT_DIR/train_s_agent.py"

