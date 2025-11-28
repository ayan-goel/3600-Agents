#!/bin/bash

#SBATCH --job-name=RonaldoTrain
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=12:00:00
#SBATCH -o ronaldo-train-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=========================================="
echo "Ronaldo Training Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load Python module
module load python/3.10

# Navigate to project root
cd $SLURM_SUBMIT_DIR
cd ../..

# Use existing environment
if [ -d "ronaldo_env" ]; then
    echo "Using existing virtual environment..."
    source ronaldo_env/bin/activate
else
    echo "Creating virtual environment..."
    python3.10 -m venv ronaldo_env
    source ronaldo_env/bin/activate
    pip install --upgrade pip
    pip install torch numpy pandas
fi

echo ""
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

export PYTHONPATH=$PWD:$PWD/engine:$PYTHONPATH

# ===========================================
# Phase 1: Supervised Learning on FILTERED DATA
# ===========================================
# Learn from winner moves in dominant wins (5+ egg diff)
# This filters out weak play and focuses on strong strategies
# ~8,900 matches → ~350K winner moves
# Estimated: ~1-2 hours
echo "=========================================="
echo "Phase 1: Supervised Learning (Filtered)"
echo "Training on winner moves from dominant wins"
echo "=========================================="

python -u -m agents.Ronaldo.train \
    --data data-matches.csv \
    --output agents/Ronaldo/ronaldo_supervised.pt \
    --epochs 30 \
    --batch_size 256 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --winner_only \
    --min_egg_diff 5 \
    --min_winner_eggs 15 \
    --exclude_draws \
    --policy_weight 1.0 \
    --value_weight 0.5

echo ""
echo "Phase 1 complete!"
echo ""

# ===========================================
# Phase 2: RL Polish vs Strong Opponents + Self
# ===========================================
# 60 iterations with anti-collapse safeguards
# Opponents: Messi, Fluffy, Pele, Self (self-play)
# Estimated: ~6 hours
echo "=========================================="
echo "Phase 2: RL Polish vs Strong Opponents + Self-Play"
echo "=========================================="

python -u -m agents.Ronaldo.league_train \
    --weights agents/Ronaldo/ronaldo_supervised.pt \
    --output agents/Ronaldo \
    --opponents Messi Fluffy Pele Self \
    --iterations 60 \
    --games_per_opp 8 \
    --batch_size 128 \
    --lr 0.000005 \
    --entropy_coef 0.15 \
    --temperature 0.9 \
    --eval_interval 5 \
    --save_interval 15

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="

# Copy best weights
if [ -f "agents/Ronaldo/ronaldo_league_best.pt" ]; then
    cp agents/Ronaldo/ronaldo_league_best.pt agents/Ronaldo/ronaldo_final.pt
    echo "Saved final model to ronaldo_final.pt"
fi

