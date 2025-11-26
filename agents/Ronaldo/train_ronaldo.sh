#!/bin/bash

#SBATCH --job-name=RonaldoTrain
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=08:00:00
#SBATCH -o ronaldo-train-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=========================================="
echo "Ronaldo Neural Network Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load Python module
module load python/3.10

# Navigate to project root
cd $SLURM_SUBMIT_DIR
cd ../..  # Go from agents/Ronaldo to project root

# Create virtual environment if it doesn't exist
if [ ! -d "ronaldo_env" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv ronaldo_env
    source ronaldo_env/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch numpy pandas
else
    echo "Using existing virtual environment..."
    source ronaldo_env/bin/activate
fi

echo ""
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Set PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

# Phase 1: Train on all data with winner-weighting
echo "=========================================="
echo "Phase 1: Supervised Learning (All Data)"
echo "=========================================="
python -u -m agents.Ronaldo.train \
    --data data-matches.csv \
    --output agents/Ronaldo/ronaldo_weights.pt \
    --epochs 20 \
    --batch_size 256 \
    --lr 0.001 \
    --winner_weight 1.0 \
    --loser_weight 0.3 \
    --policy_weight 1.0 \
    --value_weight 0.5

echo ""
echo "Phase 1 complete!"
echo ""

# Phase 2: Fine-tune on winner moves only
echo "=========================================="
echo "Phase 2: Fine-tuning (Winner Moves Only)"
echo "=========================================="
python -u -m agents.Ronaldo.train \
    --data data-matches.csv \
    --output agents/Ronaldo/ronaldo_weights_winner.pt \
    --epochs 10 \
    --batch_size 256 \
    --lr 0.0001 \
    --winner_only \
    --policy_weight 1.0 \
    --value_weight 0.3

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="

