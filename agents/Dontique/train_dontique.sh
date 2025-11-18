#!/bin/bash

#SBATCH --job-name=DontiqueTrain
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=06:00:00
#SBATCH -o dontique-train-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

module load python/3.10

if [ ! -d "dontique_env" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv dontique_env
    source dontique_env/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Using existing virtual environment..."
    source dontique_env/bin/activate
fi

echo "Starting Dontique training..."
python -u -m agents.Dontique.train_s_agent
