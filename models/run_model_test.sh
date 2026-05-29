#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --account=cseduproject
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=lars.zandbergen@ru.nl
#SBATCH --mail-type=FAIL

# Navigate to the script directory
cd /vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction

# Activate virtual environment
source /vol/csedu-nobackup/project/lzandbergen-storage/.venv/bin/activate

# Run the model testing script
python models/quick_model_test.py