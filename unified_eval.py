"""
unified_eval.py

This script completely evaluates the performance of either a local LLM or an OpenAI model on the task of extracting product feature values from texts.

Usage:
    python unified_eval.py

Requirements:
    - Either an OpenAI API key or the model path to an LLM that can be locally inferenced.
    - A weave account to store the evaluation results.
"""

import argparse
import datetime
import logging
import os
import random
import torch
import weave, wandb
import gc

from datasets import load_from_disk
from dotenv import load_dotenv
from openai import OpenAI
from tools.unified_metrics import evaluate_response, calculate_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

PROJECT_NAME = "unified-evaluation"
EVALUATION_COUNT = 290
DEBUG_LEVEL = logging.DEBUG

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Unified LLM PAVE Evaluation",
        description=__doc__,
    )
    parser.add_argument('-m', '--model_path', type=str, default=None, help="The path to either a base model, or the adapters of a finetuned model. Overrides an OpenAI model")
    parser.add_argument('-o', '--openai_model', type=str, default=None, help="The OpenAI model you'd like to use. Is ignored when a local model is also provided")
    parser.add_argument('-d', '--dataset', type=str, default=None, help="The local path to the dataset dictionary you've created.")
    parser.add_argument('--eval_count', type=int, default=2000, help="The amount of samples you'd like to evaluate, defaults to 2000")

    args = parser.parse_args()

    if args.model_path and args.openai_model:
        parser.error("Both an OpenAI model and a local model have been provided, exiting.")
        raise SystemExit(1)
    elif not args.model_path and not args.openai_model:
        parser.error("No model has been provided to evaluate, exiting.")
        raise SystemExit(1)
    return args

def setup_logging(log_name: str, debug_level=DEBUG_LEVEL):
    log_dir = "/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/logs/amazon"
    os.makedirs(log_dir, exist_ok=True)
    log_name = log_name.split("/")[-1]
    log_file = os.path.join(log_dir, log_name + ".log")

    logging.basicConfig(
        level=debug_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_model(model_path: str, temperature=0.3):
    if model_path.startswith("openai"):
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_KEY:
            raise ValueError("OPENAI_API_KEY environment variable couldn't be found")
        
        model = model_path.split('/')[-1]
        client = OpenAI(api_key=OPENAI_KEY)
        
        # Check if model is valid by attempting to list models
        try:
            available_models = [m.id for m in client.models.list()]
            if model not in available_models:
                raise ValueError(f"Model {model} is not available. Available models: {available_models}")
        except Exception as e:
            raise ValueError(f"Error validating OpenAI model: {str(e)}")
            
        return client
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype = torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            max_new_tokens=100,
            return_full_text=False,
            truncation=True,
            num_return_sequences=1,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return pipe

def get_model_response(endpoint, model_path: str, messages: dict):
    """Gets the response of either a local LLM or OpenAI, dependent on model_path"""

    if model_path.startswith("openai"):
        completion = endpoint.chat.completions.create(
            model = model_path.split('/')[-1],
            messages = messages
        )
        response = completion.choices[0].message.content
    else:
        chat = endpoint(messages)
        response = chat[0]["generated_text"]
        
        # Clear CUDA cache after each local model inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
    return response

def evaluate_dataset(endpoint, eval_path, eval_count):
    system_prompt = """You are an expert in extracting product features from technical documents. \n
The user will provide a template showing the features they'd like to extract.
Your task is to fill in the "values" for each feature from the provided product information.
If a feature cannot be found, fill in "None" for its value. 
Only answer in newline-formatted responses. \n"""
    system_prompt = """You are a world-class algorithm for extracting information in structured formats.""" # Brinkmann

    total_eval = {"nn":0, "nv":0, "vn":0, "vw":0, "vc":0}

    data = load_from_disk(eval_path)
    val_data = data['test']

    # Check if we have enough examples
    total_examples = len(val_data)
    if eval_count > total_examples:
        logger.warning(f"Requested sample size {eval_count} is larger than dataset size {total_examples}. Using all examples.")
        eval_count = total_examples
    
    # Create a random sample of indices and extract the data at those indices
    indices = random.sample(range(total_examples), eval_count)
    sample_lines = enumerate([val_data[idx] for idx in indices])
    
    for i, entry in sample_lines:
        labels = entry["labels"]
        label_dict = {lbl["key"].lower(): lbl["value"].lower() for lbl in labels}
        combined_text = "\n".join([para['text'] for para in entry["paragraphs"]])
        feature_names = [f'    {attr["key"]} : value_{i+1}' for i, attr in enumerate(entry["labels"])]
        features_string = "\n".join(feature_names)

        user_prompt = f"""Here is the product information:
        \n\n{combined_text}

Provide the output in the format with the feature names as keys and their corresponding "values",
using the template below (each feature on a new line):

{features_string} \n
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        logger.debug(f"Model input: {messages}")
        response = get_model_response(endpoint, model_name, messages)

        response = response.replace("assistant", "", 1).strip().lower()
        logger.info(f"Model response:\n {response}")
        response_lines = response.split("\n")
        
        eval_dict = evaluate_response(label_dict, response_lines, logger)
        for metric in eval_dict:
            total_eval[metric] += eval_dict[metric]
        total = sum(total_eval.values())

        if i % 50 == 0: 
            precision, recall, f1 = calculate_metrics(total_eval)
            logger.info(f"Processed {i} entries")
            logger.info(f"Averaged Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

    logger.info(f"Final eval dict: {total_eval}")
    precision, recall, f1 = calculate_metrics(total_eval)

    logger.info(f"Final Precision: {precision:.2f}")
    logger.info(f"Final Recall: {recall:.2f}")
    logger.info(f"Final F1: {f1:.2f}")
    logger.info(f"Processed {total} attributes")

    return {
        "avg_precision": precision,
        "avg_recall": recall,
        "avg_f1": f1, 
        "total_attributes": total
    }

if __name__ == "__main__":
    args = parse_args()
    load_dotenv()

    if args.model_path:
        model_name = args.model_path
    elif args.openai_model:
        model_name = "openai/" + args.openai_model

    start_moment = datetime.datetime.now().strftime("%d-%m_%H:%M_")
    local_name = start_moment + model_name.split("/")[-1] + f"-{EVALUATION_COUNT}_eval"
    logger = setup_logging(local_name)

    EVAL_PATH = args.dataset # Added this line

    config = {
        "model_name" : model_name,
        "eval_count" : EVALUATION_COUNT,
        "eval_dataset" : EVAL_PATH,
        "started_at" : start_moment,
    }
    
    wandb.init(project=PROJECT_NAME, name=local_name, config=config)
    weave.init(project_name=PROJECT_NAME)
    endpoint = create_model(model_name)
    results = evaluate_dataset(endpoint, EVAL_PATH, EVALUATION_COUNT)
    wandb.log(results)