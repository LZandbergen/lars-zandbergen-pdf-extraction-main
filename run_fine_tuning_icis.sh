#!/bin/bash -e
#SBATCH --partition=icis
#SBATCH --account=icis
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=lars.zandbergen@ru.nl
#SBATCH --output=/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/logs/fine-tuning/%j.out
#SBATCH --error=/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/logs/fine-tuning/%j.err
#SBATCH --signal=B:12@300

# Navigate to the script directory
cd /vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/

# Activate your environment if needed
source /vol/csedu-nobackup/project/lzandbergen-storage/venv2/bin/activate

# Run the fine-tuning script with passed arguments
srun python fine-tuning.py "$@"