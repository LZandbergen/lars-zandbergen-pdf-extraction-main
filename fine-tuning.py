# Standard library
import os
import sys
import re
import signal
import logging
import argparse
import datetime
import subprocess
from dotenv import load_dotenv

# ML/Data libraries
import torch
import numpy as np
import evaluate
from datasets import load_from_disk

# HuggingFace ecosystem
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback)
from peft import (LoraConfig, get_peft_model)
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login as hf_login

# Monitoring and tracking
import wandb
import weave
from setproctitle import setproctitle

# Import configuration
from finetuning_config import (
    TrainingConfig, get_config_from_args, prepare_wandb_config,
    RESPONSE_TEMPLATE, TARGET_MODULES,
    DEBUG_LEVEL, WANDB_PROJECT_NAME, DATA_PATH
)

setproctitle("lars-fine-tuning")
os.environ["WANDB_SILENT"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Should help with memory fragmentation and OOM errors
os.environ["TQDM_INTERVAL"] = "50" # Reduces the constant logging from tqdm clogging up the logs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default=TrainingConfig().base_model)
    parser.add_argument('--subset_size', type=int, default=TrainingConfig().subset_size)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--load_in_4bit', action='store_true', default=TrainingConfig().load_in_4bit)
    parser.add_argument('--load_in_8bit', action='store_true', default=TrainingConfig().load_in_8bit)
    parser.add_argument('--JSON-mode', action='store_true', help='Activate JSON prompt and processing', default=TrainingConfig().json_mode)
    parser.add_argument('--data_path', type=str, default=TrainingConfig().data_path)  # Add this line
    return parser.parse_args()

#Configure logging
def setup_logging(run_name):
    logging.basicConfig(
        level=DEBUG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

args = parse_args()
load_dotenv()

# Get configuration from args
config = get_config_from_args(args)

# Use system message from config
system_message = config.system_message

# Get the SLURM job ID and append to the run name
job_id = os.getenv("SLURM_JOB_ID")
if job_id:
    logging.info(f"SLURM job ID: {job_id}")
    config.job_id = job_id

config.timestamp = datetime.datetime.now().strftime("%d-%m-%H:%M")
run_base_name = config.run_name if config.run_name else config.base_model.split('/')[-1] + f"-{config.quant_config}" + f"-{config.subset_size}" + f"-{config.job_id}" + f"-{config.timestamp}"
new_model = f"models/fine-tuned/{run_base_name}"
config.model_save_path = new_model

setup_logging(run_base_name)
logger = logging.getLogger(__name__)
logger.info(f"New model path: {config.model_save_path}")

hf_token = os.getenv("HUGGINGFACE_TOKEN")
hf_login(token=hf_token)
wb_key = os.getenv("WANDB_KEY")
wandb.login(key=wb_key)

wandb_config = prepare_wandb_config(config)
run = wandb.init(
    project=WANDB_PROJECT_NAME,
    job_type='fine-tuning',
    dir="/vol/csedu-nobackup/project/lzandbergen-storage/wandb",
    anonymous='allow',
    name=run_base_name,
    config=wandb_config
)
weave.init(project_name=WANDB_PROJECT_NAME)
wandb.define_metric("gpu.0.memoryAllocatedBytes", summary="max")
wandb.define_metric("batch_size", summary="last")

logger.info("Start loading the tokenizer")
tokenizer = AutoTokenizer.from_pretrained(config.base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

logger.info(f"Start loading the dataset")
dataset = load_from_disk(config.data_path)

# Get actual dataset sizes
train_size = len(dataset['train'])
test_size = len(dataset['test'])
logger.info(f"Original dataset sizes - Train: {train_size}, Test: {test_size}")

# Adjust subset size if needed
effective_subset_size = min(config.subset_size, train_size)
effective_test_size = min(config.test_size, test_size)

if effective_subset_size < config.subset_size:
    logger.warning(f"Requested training subset size {config.subset_size} is larger than available data ({train_size}). Using all available data.")

if effective_test_size < config.test_size:
    logger.warning(f"Requested test subset size {config.test_size} is larger than available data ({test_size}). Using all available data.")

# Update the config with actual sizes used
config.effective_subset_size = effective_subset_size
config.effective_test_size = effective_test_size

# Select the data with safe ranges
train_dataset = dataset['train'].shuffle(seed=42).select(range(effective_subset_size))
test_dataset = dataset['test'].shuffle(seed=42).select(range(effective_test_size))

# Log the actual sizes used
logger.info(f"Using {len(train_dataset)} training samples and {len(test_dataset)} test samples")

# Sample from dataset
logger.debug(f"Sample from training dataset: {train_dataset[0]}")

# Define the function to format the chat template
def format_chat_template(dataset_entry):
    combined_text = "\n".join([para['text'] for para in dataset_entry['paragraphs']])

    # Generate feature names template
    feature_names = [f'    "{attr["key"]}" : "value_{i+1}"' for i, attr in enumerate(dataset_entry['labels'])]
    feature_names_str = "\n".join(feature_names)

    user_message = f"""Here is the product information:
        \n\n{combined_text}

Provide the output in new lines format with the feature names as keys and their corresponding "values",
use the template below, where each feature is on a new line. Only respond with the features that are requested.:

{feature_names_str}\n """

    prompt_text = f'### Question: \n{system_message} \n{user_message} \n\n### Answer: \n'

    labels_list = [f'{lbl["key"].lower()} : {lbl["value"].lower()}' for lbl in dataset_entry['labels']]
    response_text = "\n".join(labels_list)

    return {"prompt": prompt_text, "response": response_text}

response_template = RESPONSE_TEMPLATE
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]

train_dataset = train_dataset.map(format_chat_template)
test_dataset = test_dataset.map(format_chat_template)
logger.info("Got past the dataset mapping, converted to chat template.")

def tokenize_function(dataset):
    prompts = dataset['prompt']
    responses = dataset['response']
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]

    # Tokenize prompts and responses separately, NO PADDING here initially
    tokenized_prompts = tokenizer(
        prompts,
        truncation=True,
        max_length=config.max_input_length - 1,
        return_attention_mask=False,
        add_special_tokens=False
    )
    tokenized_responses = tokenizer(
        responses,
        truncation=True,
        max_length=config.max_response_length,
        return_attention_mask=False,
        add_special_tokens=False
    )

    # Combine tokenized prompts and responses and add EOS token at the end
    input_ids = []
    attention_masks = []
    labels = []
    max_combined_length = config.max_input_length + config.max_response_length # Calculate max combined length

    for i in range(len(tokenized_prompts['input_ids'])):
        combined_input_ids = list(tokenized_prompts['input_ids'][i]) + list(tokenized_responses['input_ids'][i]) + [tokenizer.eos_token_id]
        combined_attention_mask = [1] * len(combined_input_ids)
        combined_labels = combined_input_ids.copy()

        response_template_start_index = -1

        # Find the start index of the response template in the combined input
        for index in range(len(combined_input_ids) - len(response_template_ids) + 1):
            if combined_input_ids[index:index + len(response_template_ids)] == response_template_ids:
                response_template_start_index = index
                break
            else:
                # Log debug information
                logger.debug(f"Tokenized input: {combined_input_ids}")
                logger.debug(f"Tokenized response template: {response_template_ids}")
                logger.debug(f"Tokenized response template start index: {response_template_start_index}")

        # Set labels to -100 for all tokens before the response template
        if response_template_start_index != -1:
            combined_labels[:response_template_start_index + len(response_template_ids)] = [-100] * (response_template_start_index + len(response_template_ids))
        else:
            logger.warning(f"Response template not found in input for index {i} (refined search). Setting all labels to -100.")
            combined_labels = [-100] * len(combined_input_ids)

        # Padding to max_combined_length
        padding_length = max_combined_length - len(combined_input_ids)
        if padding_length > 0:
            combined_input_ids = combined_input_ids + [tokenizer.pad_token_id] * padding_length
            combined_attention_mask = combined_attention_mask + [0] * padding_length # Pad attention mask with 0s
            combined_labels = combined_labels + [-100] * padding_length # Pad labels with -100

        input_ids.append(combined_input_ids)
        attention_masks.append(combined_attention_mask)
        labels.append(combined_labels)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_train = tokenized_train.remove_columns([col for col in tokenized_train.column_names if col not in ['attention_mask', 'input_ids', 'labels']])
tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_test = tokenized_test.remove_columns([col for col in tokenized_test.column_names if col not in ['attention_mask', 'input_ids', 'labels']])

# Sample from tokenized dataset
sample = tokenized_train[0]
logger.debug(f"Sample from tokenized training dataset: {list(zip(sample['input_ids'], sample['attention_mask'], sample['labels']))}")

# Check the device of tensors in the tokenized dataset
logger.debug("Checking device of tensors in tokenized_test dataset...")
sample_eval_dataset = tokenized_test[0] # Get the first sample
# logger.info(f"Sample from tokenized_test: {sample_eval_dataset}")
for key, value in sample_eval_dataset.items():
    if torch.is_tensor(value):
        logger.debug(f"Column '{key}' is a Tensor with shape {value.shape} on device {value.device}")
    else:
        logger.warning(f"Column '{key}' is NOT a Tensor (got type: {type(value)})")

torch_dtype = torch.float16
attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=config.load_in_4bit,
    load_in_8bit=config.load_in_8bit,
    bnb_4bit_compute_dtype=torch_dtype,
)

# Clear cache to make sure enough memory is available
torch.cuda.empty_cache()

# Load model
model = AutoModelForCausalLM.from_pretrained(
    config.base_model,
    # local_files_only=True,
    quantization_config=bnb_config, 
    device_map = "auto",
    # attn_implementation=attn_implementation
)
logger.info("Model loaded")
model.enable_input_require_grads()
model.config.use_cache = False

peft_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="causal_lm",
    target_modules=TARGET_MODULES
)
model = get_peft_model(model, peft_config)

metric = evaluate.load("accuracy")
def compute_only_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

@weave.op()
class WeaveCallback(TrainerCallback):
    def __init__(self):
        self.step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            metrics_to_log = {}
            # Training metrics
            if "loss" in logs:
                metrics_to_log.update({
                    "train/loss": logs.get("loss", 0),
                    "train/learning_rate": logs.get("learning_rate", 0),
                    "train/epoch": logs.get("epoch", 0)
                })

training_arguments = SFTConfig(
    max_seq_length=config.max_input_length + config.max_response_length,
    output_dir=config.model_save_path,
    auto_find_batch_size=False,
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    num_train_epochs=config.num_train_epochs,
    eval_strategy="no",
    save_strategy="epoch",
    save_total_limit=3,
    logging_steps=50,
    logging_strategy="steps",
    learning_rate=config.learning_rate,
    fp16=True,
    bf16=False,
    report_to="wandb",
    dataloader_num_workers=4,
    run_name=run_base_name,
)

# Log the chosen batch size to wandb
chosen_batch_size = training_arguments.per_device_train_batch_size
logger.info(f"Chosen batch size: {chosen_batch_size}")
wandb.config.update({"chosen_batch_size": chosen_batch_size}, allow_val_change=True)

torch.cuda.empty_cache()
logger.info(f"Run Name: {training_arguments.run_name}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    peft_config=peft_config,
    args=training_arguments,
    packing=False,
    callbacks=[WeaveCallback()],
    compute_metrics=compute_only_accuracy
)

def save_model_on_signal():
    logger.info(f"Signal received, saving model to {new_model}")
    trainer.save_model(new_model)
    model.save_pretrained(new_model)

# Register the signal handler
signal.signal(12, save_model_on_signal)

try:
    trainer.train()
    save_model_on_signal()
    
    # Update WandB summary with the model save location
    wandb.run.summary["new_model_path"] = config.model_save_path

    # Run the evaluation straight away
    subprocess.run([
        sys.executable, "unified_eval.py",
        "-m", config.model_save_path,
    ])
except Exception as e:
    logger.info(f"Exception occurred: {e}")
    save_model_on_signal()
    raise e
finally:
    run.finish()
    logger.info("Run finished, exiting.")
    sys.exit(0)