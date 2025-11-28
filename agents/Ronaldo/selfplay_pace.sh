#!/bin/bash

#SBATCH --job-name=RonaldoSelfPlay
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=16:00:00
#SBATCH -o ronaldo-selfplay-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=========================================="
echo "Ronaldo Self-Play Reinforcement Learning"
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

# Use existing environment or create new one
if [ -d "ronaldo_env" ]; then
    echo "Using existing virtual environment..."
    source ronaldo_env/bin/activate
else
    echo "Creating virtual environment..."
    python3.10 -m venv ronaldo_env
    source ronaldo_env/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch numpy pandas
fi

echo ""
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Set PYTHONPATH to include both project root and engine
export PYTHONPATH=$PWD:$PWD/engine:$PYTHONPATH

# Phase 1: Self-Play Training (1000 iterations, 100 games each = 100K games)
echo "=========================================="
echo "Phase 1: Self-Play Training (Large Scale)"
echo "=========================================="
echo "Target: 1000 iterations x 100 games = 100,000 self-play games"
echo ""

python -u -m agents.Ronaldo.selfplay \
    --weights agents/Ronaldo/ronaldo_weights.pt \
    --output agents/Ronaldo \
    --iterations 1000 \
    --games 100 \
    --batch_size 512 \
    --lr 0.0001 \
    --gamma 0.99 \
    --entropy_coef 0.02 \
    --value_coef 0.5 \
    --temp_start 1.0 \
    --temp_end 0.2 \
    --eval_interval 20 \
    --save_interval 100

echo ""
echo "Phase 1 complete!"
echo ""

# Phase 2: Fine-tuning with lower temperature (exploitation)
echo "=========================================="
echo "Phase 2: Fine-tuning (Low Temperature)"
echo "=========================================="

python -u -m agents.Ronaldo.selfplay \
    --weights agents/Ronaldo/ronaldo_selfplay_best.pt \
    --output agents/Ronaldo \
    --iterations 200 \
    --games 50 \
    --batch_size 256 \
    --lr 0.00005 \
    --gamma 0.99 \
    --entropy_coef 0.005 \
    --value_coef 0.5 \
    --temp_start 0.3 \
    --temp_end 0.1 \
    --eval_interval 10 \
    --save_interval 50

echo ""
echo "=========================================="
echo "Self-Play Training Complete!"
echo "End time: $(date)"
echo "=========================================="

# Copy best weights to main file
if [ -f "agents/Ronaldo/ronaldo_selfplay_best.pt" ]; then
    cp agents/Ronaldo/ronaldo_selfplay_best.pt agents/Ronaldo/ronaldo_weights.pt
    echo "Copied best self-play weights to ronaldo_weights.pt"
fi

