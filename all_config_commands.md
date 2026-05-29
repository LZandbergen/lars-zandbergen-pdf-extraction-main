# Commands to run fine-tuning experiments on ICIS and DAS

# Llama-3.1-8B

## ICIS with OA Mine Stratified Split (1400 samples)
### Fine-tune
```bash
sbatch run_fine_tuning_icis.sh --base_model "meta-llama/Llama-3.1-8B-Instruct" --data_path "data/OA-mine/dataset/stratified_split" --subset_size 1400 --load_in_8bit --run_name "llama-3.1-8B-oa-strat"
```

### Eval
```bash
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.1-8B-oa-strat" -d "data/OA-mine/dataset/stratified_split" --eval_count 290
```

## ICIS with OA Mine Category Split (1400 samples)
### Fine-tune
```bash
sbatch run_fine_tuning_icis.sh --base_model "meta-llama/Llama-3.1-8B-Instruct" --data_path "data/OA-mine/dataset/category_split" --subset_size 1400 --load_in_8bit --run_name "llama-3.1-8B-oa-cat"
```

### Eval
```bash
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.1-8B-oa-cat" -d "data/OA-mine/dataset/category_split" --eval_count 290
```

## ICIS with MAVE Dataset (10000 samples)
### Fine-tune
```bash
sbatch run_fine_tuning_icis.sh --base_model "meta-llama/Llama-3.1-8B-Instruct" --data_path "data/MAVE/dataset" --subset_size 10000 --load_in_8bit --run_name "llama-3.1-8B-mave"
```

### Eval
```bash
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.1-8B-mave" -d "data/MAVE/dataset" --eval_count 290
```

# Llama 3.2
## Llama-3.2-1B on DAS with OA Mine Stratified Split (1400 samples)
### Fine-Tune
```bash
sbatch run_fine_tuning_das.sh --base_model "meta-llama/Llama-3.2-1B-Instruct" --data_path "data/OA-mine/dataset/stratified_split" --subset_size 1400 --load_in_8bit --run_name "llama-3.2-1B-oa-strat"
```

### Eval
```bash
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.2-1B-oa-strat" -d "data/OA-mine/dataset/stratified_split" --eval_count 290
```

## Llama-3.2-1B on DAS with OA Mine Category Split (1400 samples)
### Fine-Tune
```bash
sbatch run_fine_tuning_das.sh --base_model "meta-llama/Llama-3.2-1B-Instruct" --data_path "data/OA-mine/dataset/category_split" --subset_size 1400 --load_in_8bit --run_name "llama-3.2-1B-oa-cat"
```

### Eval
```bash
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.2-1B-oa-cat" -d "data/OA-mine/dataset/category_split" --eval_count 290
```

## Llama-3.2-1B on DAS with MAVE Dataset (10000 samples)
### Fine-Tune
```bash
sbatch run_fine_tuning_das.sh --base_model "meta-llama/Llama-3.2-1B-Instruct" --data_path "data/MAVE/dataset" --subset_size 10000 --load_in_8bit --run_name "llama-3.2-1B-mave"
```

### Eval
```bash
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.2-1B-mave" -d "data/MAVE/dataset" --eval_count 290
```

## Phi-3.5-mini on DAS with OA Mine Stratified Split (1400 samples)
### Fine-Tune
```bash
sbatch run_fine_tuning_das.sh --base_model "microsoft/Phi-3.5-mini-instruct" --data_path "data/OA-mine/dataset/stratified_split" --subset_size 1400 --load_in_8bit --run_name "phi-3.5-mini-oa-strat"
```

### Eval
```bash
sbatch uni_eval.sh -m "models/fine-tuned/phi-3.5-mini-oa-strat" -d "data/OA-mine/dataset/stratified_split" --eval_count 290
```

## Phi-3.5-mini on DAS with OA Mine Category Split (1400 samples)
### Fine-Tune
```bash
sbatch run_fine_tuning_das.sh --base_model "microsoft/Phi-3.5-mini-instruct" --data_path "data/OA-mine/dataset/category_split" --subset_size 1400 --load_in_8bit --run_name "phi-3.5-mini-oa-cat"
```

### Eval
```bash
sbatch uni_eval.sh -m "models/fine-tuned/phi-3.5-mini-oa-cat" -d "data/OA-mine/dataset/category_split" --eval_count 290
```

## Phi-3.5-mini on DAS with MAVE Dataset (10000 samples)
### Fine-Tune
```bash
sbatch run_fine_tuning_das.sh --base_model "microsoft/Phi-3.5-mini-instruct" --data_path "data/MAVE/dataset" --subset_size 10000 --load_in_8bit --run_name "phi-3.5-mini-mave"
```

### Eval
```bash
sbatch uni_eval.sh -m "models/fine-tuned/phi-3.5-mini-mave" -d "data/MAVE/dataset" --eval_count 290
```

# Batch Commands

## All Fine-Tuning Commands
```bash
sbatch run_fine_tuning_icis.sh --base_model "meta-llama/Llama-3.1-8B-Instruct" --data_path "data/OA-mine/dataset/stratified_split" --subset_size 1400 --load_in_8bit --run_name "llama-3.1-8B-oa-strat" && \
sbatch run_fine_tuning_icis.sh --base_model "meta-llama/Llama-3.1-8B-Instruct" --data_path "data/OA-mine/dataset/category_split" --subset_size 1400 --load_in_8bit --run_name "llama-3.1-8B-oa-cat" && \
sbatch run_fine_tuning_icis.sh --base_model "meta-llama/Llama-3.1-8B-Instruct" --data_path "data/MAVE/dataset" --subset_size 10000 --load_in_8bit --run_name "llama-3.1-8B-mave" && \
sbatch run_fine_tuning_das.sh --base_model "meta-llama/Llama-3.2-1B-Instruct" --data_path "data/OA-mine/dataset/stratified_split" --subset_size 1400 --load_in_8bit --run_name "llama-3.2-1B-oa-strat" && \
sbatch run_fine_tuning_das.sh --base_model "meta-llama/Llama-3.2-1B-Instruct" --data_path "data/OA-mine/dataset/category_split" --subset_size 1400 --load_in_8bit --run_name "llama-3.2-1B-oa-cat" && \
sbatch run_fine_tuning_das.sh --base_model "meta-llama/Llama-3.2-1B-Instruct" --data_path "data/MAVE/dataset" --subset_size 10000 --load_in_8bit --run_name "llama-3.2-1B-mave" && \
sbatch run_fine_tuning_das.sh --base_model "microsoft/Phi-3.5-mini-instruct" --data_path "data/OA-mine/dataset/stratified_split" --subset_size 1400 --load_in_8bit --run_name "phi-3.5-mini-oa-strat" && \
sbatch run_fine_tuning_das.sh --base_model "microsoft/Phi-3.5-mini-instruct" --data_path "data/OA-mine/dataset/category_split" --subset_size 1400 --load_in_8bit --run_name "phi-3.5-mini-oa-cat" && \
sbatch run_fine_tuning_das.sh --base_model "microsoft/Phi-3.5-mini-instruct" --data_path "data/MAVE/dataset" --subset_size 10000 --load_in_8bit --run_name "phi-3.5-mini-mave"
```

## All Evaluation Commands
### Evaluate runs on their own datasets.
```bash
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.1-8B-oa-strat" -d "data/OA-mine/dataset/stratified_split" --eval_count 290 && \
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.1-8B-oa-cat" -d "data/OA-mine/dataset/category_split" --eval_count 290 && \
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.1-8B-mave" -d "data/MAVE/dataset" --eval_count 1000 && \
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.2-1B-oa-strat" -d "data/OA-mine/dataset/stratified_split" --eval_count 290 && \
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.2-1B-oa-cat" -d "data/OA-mine/dataset/category_split" --eval_count 290 && \
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.2-1B-mave" -d "data/MAVE/dataset" --eval_count 1000 && \
sbatch uni_eval.sh -m "models/fine-tuned/phi-3.5-mini-oa-strat" -d "data/OA-mine/dataset/stratified_split" --eval_count 290 && \
sbatch uni_eval.sh -m "models/fine-tuned/phi-3.5-mini-oa-cat" -d "data/OA-mine/dataset/category_split" --eval_count 290 && \
sbatch uni_eval.sh -m "models/fine-tuned/phi-3.5-mini-mave" -d "data/MAVE/dataset" --eval_count 1000
```

### Evaluate runs on each others datasets.
```bash
# FT: OA-Mine, DS: MAVE
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.1-8B-oa-strat" -d "data/MAVE/dataset" --eval_count 290 && \
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.2-1B-oa-strat" -d "data/MAVE/dataset" --eval_count 290 && \
sbatch uni_eval.sh -m "models/fine-tuned/phi-3.5-mini-oa-strat" -d "data/MAVE/dataset" --eval_count 290 && \

# FT: MAVE, DS: OA-Mine
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.1-8B-mave" -d "data/OA-mine/dataset/stratified_split" --eval_count 290 && \
sbatch uni_eval.sh -m "models/fine-tuned/llama-3.2-1B-mave" -d "data/OA-mine/dataset/stratified_split" --eval_count 290 && \
sbatch uni_eval.sh -m "models/fine-tuned/phi-3.5-mini-mave" -d "data/OA-mine/dataset/stratified_split" --eval_count 290
```