"""
amazon_pipeline.py

This script orchestrates evaluating a model against a product feature dataset.

Usage:
  python amazon_pipeline.py

Requirements:
  - Weave for evaluation
  - A model pipeline capable of text generation
  - feature_validator.py for scoring logic
"""

import os
import json
import weave, wandb
import random
import torch
import logging
import datetime
import signal
import argparse

from typing import List, Dict, Any
from tools.unified_metrics import calculate_metrics_and_f1
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

PROJECT_NAME = "amazon-evaluation-run"

MODEL_PATH = "microsoft/Phi-3.5-mini-instruct"
# MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = None
ADAPTER_PATH = "models/fine-tuned/Phi-3.5-mini-instruct-8bit-10000-5814732-11-03-13:33"

# TODO: Should use load_from_disk just like in the model training
EVAL_PATH = "/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/data/MAVE/merged/test_set.jsonl"
EVALUATION_COUNT = 10

random.seed(42)
model_name = os.path.basename(MODEL_PATH.split("/checkpoint")[0])
adapter_name = os.path.basename(ADAPTER_PATH.split("/checkpoint")[0]) if ADAPTER_PATH else None
start_moment = datetime.datetime.now().strftime("%d-%m_%H:%m_")

model_or_adapter_name = adapter_name if adapter_name else model_name

config = {
    "model_name" : model_name,
    "adapter_name" : adapter_name,
    "started_at" : start_moment,
    "eval_count" : EVALUATION_COUNT,
    "eval_dataset" : os.path.basename(EVAL_PATH),
}

local_name = start_moment + model_or_adapter_name + f"-{EVALUATION_COUNT}_eval"

system_prompt = """You are an expert in extracting product features from technical documents. \n
The user will provide a template showing the features they'd like to extract.
Your task is to fill in the "values" for each feature from the provided product information.
If a feature cannot be found, fill in "None" for its value. 
Only answer in newline-formatted responses. \n"""

wb_key = os.getenv("WANDB_KEY")
wandb.login(key=wb_key)

intercepted = False

# Configure logging
def setup_logging(log_name: str, debug_level=logging.DEBUG):
    log_dir = "/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/logs/amazon"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_name + ".log")

    logging.basicConfig(
        level=debug_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def toggle_intercepted():
    global intercepted
    intercepted = True

@weave.op()
def evaluate_amazon_labels(model, tokenizer, jsonl_path: str, evaluation_size: int = 200):
    correct = 0
    total = 0
    errors = 0
    
    # Lists to store per-entry scores
    f1_scores = []
    precision_scores = []
    recall_scores = []

    logger.info(f"Loading data from {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        sample_lines = enumerate(random.sample(lines, evaluation_size))
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            max_new_tokens=100,
            return_full_text=False,
            truncation=True,
            num_return_sequences=1,
            temperature=0.5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        for i, entry in sample_lines:
            if intercepted:
                logging.warning("Ran out of time, printing final logs")
                break

            entry = json.loads(entry)
            labels = entry["labels"]
            label_dict = {lbl["key"].lower(): lbl["value"].lower() for lbl in labels}
            combined_text = " ".join([para['text'] for para in entry["paragraphs"]])
            feature_names = [f'    {attr["key"]} : value_{i+1}' for i, attr in enumerate(entry["attributes"])]
            feature_names = "\n".join(feature_names)

            user_prompt = f"""Here is the product information:
            \n\n{combined_text}

Provide the output in the format with the feature names as keys and their corresponding "values",
using the template below (each feature on a new line):

{feature_names} \n
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            chat = pipe(messages)
            response = chat[0]["generated_text"]
            actual_response = response.replace("assistant", "", 1).strip().lower()
            logger.debug(f"Model response:\n {actual_response}")
            response_lines = actual_response.split("\n")
            
            total += len(label_dict)
            # Updated call to calculate_metrics
            (resp_correct, resp_error, precision, recall, f1) = calculate_metrics_and_f1(label_dict, response_lines, logger)
            
            correct += resp_correct
            errors += resp_error
            
            # Save per-entry scores for later averaging
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            # Calculate intermediate accuracy (optional)
            accuracy = correct / total if total else 0
            avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
            avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

            if i % 100 == 0:
                logger.info(f"Processed {i} entries")
                logger.info(f"Intermediate accuracy: {accuracy:.2f}")
                logger.info(f"Averaged Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1: {avg_f1:.2f}")
                wandb.log({
                    "accuracy": accuracy,
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "f1": avg_f1,
                    "errors": errors,
                    "total_entries": total
                })

    # Compute overall averages after evaluation
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    logger.info(f"Processed {total} entries")
    logger.info(f"Final accuracy: {accuracy:.2f}")
    logger.info(f"Average Precision: {avg_precision:.2f}")
    logger.info(f"Average Recall: {avg_recall:.2f}")
    logger.info(f"Average F1: {avg_f1:.2f}")
    logger.info(f"Errors: {errors}")
    
    # Log the final averaged metrics to wandb
    wandb.log({
        "accuracy": accuracy,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "errors": errors,
        "total_entries": total
    })

    return {
        "accuracy": accuracy, 
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "errors": errors, 
        "total_entries": total
    }

signal.signal(signal.SIGTERM, toggle_intercepted)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_PATH)
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    ADAPTER_PATH = args.adapter_path

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        # local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    adapter_dir = ADAPTER_PATH
    
    if adapter_dir:
        # Load and apply the PEFT adapter
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
            torch_dtype=torch.bfloat16
        )

        # Merge the adapter weights with the base model
        model = model.merge_and_unload()

    setup_logging(local_name)
    
    try:
        wandb.init(project=PROJECT_NAME, name=local_name, config=config)
        weave.init(project_name=PROJECT_NAME)
        logger = logging.getLogger(__name__)
        results = evaluate_amazon_labels(model, tokenizer, EVAL_PATH, evaluation_size=EVALUATION_COUNT)
        wandb.log(results)
    finally:
        # Make sure wandb run is properly closed
        wandb.finish()