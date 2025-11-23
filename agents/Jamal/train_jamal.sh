#!/bin/bash

#SBATCH --job-name=JamalTrain
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=06:00:00
#SBATCH -o jamal-train-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

module load python/3.10

if [ ! -d "jamal_env" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv jamal_env
    source jamal_env/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Using existing virtual environment..."
    source jamal_env/bin/activate
fi

echo "Starting Jamal supervised training from matches..."
python -u -m agents.Jamal.train_jamal --match_dir agents/matches --epochs 10 --limit 3000

echo "Starting Jamal self-play fine-tuning..."
python -u -m agents.Jamal.selfplay --iterations 12 --games-per-opp 12 --epochs 2


