#!/bin/bash -e
#SBATCH --partition=icis
#SBATCH --account=icis
#SBATCH --cpus-per-task=4      
#SBATCH --mem=4G               
#SBATCH --time=12:00:00
#SBATCH --output=../../logs/scraping-%j.out
#SBATCH --error=../../logs/scraing-%j.err
#SBATCH --mail-type=FAIL

# Activate your Python environment if needed
source /vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/.linux/bin/activate

# Run the scraping script
python /vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/data/bol/scrape_sites.py