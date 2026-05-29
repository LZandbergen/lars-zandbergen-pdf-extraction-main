#!/bin/bash

# Subset sizes to try
SUBSET_SIZES=(25 100 500 2500 5000)

# Quantization configurations
declare -A QUANT_CONFIGS=(
    ["4bit"]="--load_in_4bit"
    ["8bit"]="--load_in_8bit"
)

for size in "${SUBSET_SIZES[@]}"; do
    for quant_name in "${!QUANT_CONFIGS[@]}"; do
        config="${QUANT_CONFIGS[$quant_name]}"
        
        # Submit the job
        sbatch \
            --output="logs/${RUN_NAME}_%j.out" \
            --error="logs/${RUN_NAME}_%j.err" \
            run_fine_tuning_icis.sh \
            --subset_size "$size" \
            --run_name "$RUN_NAME" \
            $config
    done
done
