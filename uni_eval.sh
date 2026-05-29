#!/bin/bash

#SBATCH --job-name=unified_testing
#SBATCH --output=logs/oamine/unified_testing_%j.out
#SBATCH --error=logs/oamine/unified_testing_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=icis
#SBATCH --account=icis
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=lars.zandbergen@ru.nl
#SBATCH --signal=B:TERM@60

# Navigate to the script directory
cd /vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/

# Activate your environment if needed
source /vol/csedu-nobackup/project/lzandbergen-storage/venv2/bin/activate

# Run the fine-tuning script with passed arguments
python unified_eval.py "$@"