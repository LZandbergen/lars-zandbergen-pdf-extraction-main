"""
This script evaluates OpenAI models against the OA-mine product attribute dataset.

Usage:
  python oa_pipeline_openai.py --openai_model gpt-4o --total_samples 100

Features:
  - Randomly samples products across all available categories
  - Tracks metrics per category and overall
  - Logs results to WandB dashboard
"""

import os
import json
import wandb
import weave
import random
import logging
import datetime
import signal
import argparse
import time
from pathlib import Path
from openai import OpenAI
from collections import defaultdict
from dotenv import load_dotenv

from typing import List, Dict, Any, Set, Tuple
from tools.unified_metrics import calculate_metrics_and_f1

# Default configuration
load_dotenv()

OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WANDB_KEY = os.getenv("WANDB_KEY")
TOTAL_SAMPLES = 100
MAX_SAMPLES_PER_CATEGORY = 20

# Dataset paths
BASE_PATH = "/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/data/OA-mine"
RAW_PATH = os.path.join(BASE_PATH, "raw")
TEST_PATH = os.path.join(BASE_PATH, "test")
ANNOTATION_PATH = os.path.join(BASE_PATH, "annotations")

# Initialize logger
logger = None
intercepted = False

def setup_logging(run_name: str, debug_level=logging.INFO):
    """Configure logging to file and console"""
    log_dir = "/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/logs/oamine"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{run_name}.log")

    logging.basicConfig(
        level=debug_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    global logger
    logger = logging.getLogger(__name__)
    return logger

def toggle_intercepted(sig, frame):
    """Signal handler for graceful shutdown"""
    global intercepted
    intercepted = True
    logger.warning("Received termination signal. Will save results and exit after current product.")

class OAmineEvaluator:
    """Handles evaluation of OA-mine dataset using OpenAI models"""
    
    def __init__(self, model_name: str, total_samples: int, max_per_category: int = 20):
        self.model_name = model_name
        self.total_samples = total_samples
        self.max_per_category = max_per_category
        self.client = self._setup_openai_client()
        self.categories = self._list_available_categories()
        
        logger.info(f"Initialized evaluator for model: {model_name}")
        logger.info(f"Found {len(self.categories)} available categories")
    
    def _setup_openai_client(self) -> OpenAI:
        """Set up OpenAI client"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return OpenAI(api_key=OPENAI_API_KEY)
    
    def _list_available_categories(self) -> List[str]:
        """Get list of all product categories in the dataset"""
        categories = []
        for file in os.listdir(RAW_PATH):
            if file.endswith('.jsonl'):
                categories.append(file.replace('.jsonl', ''))
        return sorted(categories)
    
    def _load_raw_products(self, category: str) -> Dict[str, Dict]:
        """Load raw product data for a specific category"""
        raw_file = os.path.join(RAW_PATH, f"{category}.jsonl")
        products = {}
        
        try:
            with open(raw_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        asin = data.get("asin")
                        if asin:
                            products[asin] = data
                    except json.JSONDecodeError:
                        continue
                        
            logger.debug(f"Loaded {len(products)} raw products from {category}")
        except Exception as e:
            logger.error(f"Error loading raw products from {category}: {e}")
            
        return products
    
    def _load_test_labels(self, category: str) -> Dict[str, Dict[str, List[str]]]:
        """Load test labels for a specific category"""
        test_file = os.path.join(TEST_PATH, f"{category}.jsonl")
        labels = {}
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Test data format is ["attribute_name": ["val1", "val2", ...]]
                        if len(data) != 1:
                            continue
                            
                        for attr_name, values in data.items():
                            asin = attr_name.split('_')[0] if '_' in attr_name else None
                            attr = attr_name.split('_')[1] if '_' in attr_name else attr_name
                            
                            if asin:
                                if asin not in labels:
                                    labels[asin] = {}
                                labels[asin][attr] = values
                    except json.JSONDecodeError:
                        continue
                        
            logger.debug(f"Loaded test labels for {len(labels)} products in {category}")
        except Exception as e:
            logger.error(f"Error loading test labels from {category}: {e}")
            
        return labels
    
    def _prepare_evaluation_sample(self) -> List[Dict]:
        """Create a sample dataset across all categories for evaluation"""
        all_samples = []
        category_counts = defaultdict(int)
        
        # First pass: collect candidate products from each category
        candidates_by_category = {}
        
        for category in self.categories:
            try:
                raw_products = self._load_raw_products(category)
                test_labels = self._load_test_labels(category)
                
                # Find products that have both raw data and test labels
                common_asins = set(raw_products.keys()) & set(test_labels.keys())
                
                if not common_asins:
                    logger.info(f"No products with labels found for category: {category}")
                    continue
                
                # Create evaluation entries
                category_candidates = []
                for asin in common_asins:
                    product = raw_products[asin]
                    labels = test_labels[asin]
                    
                    # Combine product text fields
                    product_text = f"Title: {product.get('title', '')}\n"
                    if product.get('bullet_point'):
                        product_text += f"Features: {product.get('bullet_point', '')}\n"
                    if product.get('description'):
                        product_text += f"Description: {product.get('description', '')}"
                    
                    category_candidates.append({
                        "asin": asin,
                        "category": category,
                        "product_text": product_text,
                        "labels": labels,
                        "attribute_names": list(labels.keys())
                    })
                
                candidates_by_category[category] = category_candidates
                logger.info(f"Category {category}: {len(category_candidates)} products available")
            except Exception as e:
                logger.error(f"Error processing category {category}: {e}")
        
        # Second pass: sample from categories with available products
        available_categories = [cat for cat in candidates_by_category if candidates_by_category[cat]]
        
        if not available_categories:
            logger.error("No products with labels found in any category!")
            return []
            
        logger.info(f"Found {len(available_categories)} categories with available products")
        
        # Calculate samples per category
        target_per_category = min(
            self.max_per_category,
            max(1, self.total_samples // len(available_categories))
        )
        
        # Sample from each category
        for category in available_categories:
            candidates = candidates_by_category[category]
            sample_size = min(target_per_category, len(candidates))
            
            if sample_size > 0:
                category_samples = random.sample(candidates, sample_size)
                all_samples.extend(category_samples)
                category_counts[category] = sample_size
        
        # If we need more samples to reach total_samples, get them from categories with leftover products
        if len(all_samples) < self.total_samples:
            additional_needed = self.total_samples - len(all_samples)
            additional_candidates = []
            
            for category in available_categories:
                current_count = category_counts[category]
                remaining = len(candidates_by_category[category]) - current_count
                
                if remaining > 0:
                    # Get remaining products
                    remaining_products = [
                        p for p in candidates_by_category[category] 
                        if p not in all_samples[:current_count]
                    ]
                    additional_candidates.extend(remaining_products)
            
            # Sample additional products if available
            if additional_candidates:
                additional_sample_size = min(additional_needed, len(additional_candidates))
                additional_samples = random.sample(additional_candidates, additional_sample_size)
                all_samples.extend(additional_samples)
                
                # Update category counts
                for sample in additional_samples:
                    category_counts[sample["category"]] += 1
        
        # Final shuffle to mix categories
        random.shuffle(all_samples)
        
        # Log category distribution
        logger.info(f"Final sample distribution across {len(category_counts)} categories:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {category}: {count} products")
            
        return all_samples[:self.total_samples]
    
    def _generate_with_openai(self, messages: List[Dict], max_tokens: int = 300) -> str:
        """Call OpenAI API to generate response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return ""
    
    def run_evaluation(self) -> Dict:
        """Run the full evaluation and return results"""
        system_prompt = """You are an expert in extracting product attributes from Amazon product descriptions.
The user will provide a product description and a list of attribute names.
Your task is to identify the specific values for each requested attribute from the product information.
If an attribute cannot be found, respond with "None" for its value.
Format your response with each attribute on a new line as: "attribute_name : attribute_value"
"""
        
        evaluation_samples = self._prepare_evaluation_sample()
        total_samples = len(evaluation_samples)
        
        if not total_samples:
            logger.error("No samples available for evaluation!")
            return {}
            
        logger.info(f"Starting evaluation with {total_samples} products")
        
        # Initialize metrics
        metrics = {
            "overall": {
                "correct": 0,
                "total": 0,
                "errors": 0,
                "f1_scores": [],
                "precision_scores": [],
                "recall_scores": [],
                "bleu_scores": [],
                "jaccard_scores": [],
                "products": 0
            }
        }
        
        # Process each product
        for idx, sample in enumerate(evaluation_samples):
            if intercepted:
                logger.warning("Execution interrupted. Stopping evaluation.")
                break
                
            category = sample["category"]
            asin = sample["asin"]
            product_text = sample["product_text"]
            labels = sample["labels"]
            attribute_names = sample["attribute_names"]
            
            # Initialize category metrics if new
            if category not in metrics:
                metrics[category] = {
                    "correct": 0,
                    "total": 0,
                    "errors": 0,
                    "f1_scores": [],
                    "precision_scores": [],
                    "recall_scores": [],
                    "bleu_scores": [],
                    "jaccard_scores": [],
                    "products": 0
                }
            
            # Create label dictionary for evaluation
            label_dict = {}
            for attr, values in labels.items():
                value = values[0] if values else "None"
                label_dict[attr.lower()] = value.lower()
            
            # Build template for extraction
            attribute_template = "\n".join([f"    {attr} : value" for attr in attribute_names])
            
            user_prompt = f"""Here is the product information for Amazon product (ASIN: {asin}, Category: {category}):
            
{product_text}

Please extract the following attributes from this product information:

{attribute_template}

Provide only the attribute name and extracted value on each line, separated by a colon (:).
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call OpenAI API
            start_time = time.time()
            response = self._generate_with_openai(messages)
            api_time = time.time() - start_time
            
            # Process response
            actual_response = response.lower()
            response_lines = actual_response.split("\n")
            
            # Calculate metrics
            (correct, error, bleu, jaccard, precision, recall, f1) = calculate_metrics_and_f1(
                label_dict, response_lines, logger
            )
            
            # Update overall metrics
            metrics["overall"]["correct"] += correct
            metrics["overall"]["total"] += len(label_dict)
            metrics["overall"]["errors"] += error
            metrics["overall"]["bleu_scores"].extend(bleu)
            metrics["overall"]["jaccard_scores"].extend(jaccard)
            metrics["overall"]["f1_scores"].append(f1)
            metrics["overall"]["precision_scores"].append(precision)
            metrics["overall"]["recall_scores"].append(recall)
            metrics["overall"]["products"] += 1
            
            # Update category metrics
            metrics[category]["correct"] += correct
            metrics[category]["total"] += len(label_dict)
            metrics[category]["errors"] += error
            metrics[category]["bleu_scores"].extend(bleu)
            metrics[category]["jaccard_scores"].extend(jaccard)
            metrics[category]["f1_scores"].append(f1)
            metrics[category]["precision_scores"].append(precision)
            metrics[category]["recall_scores"].append(recall)
            metrics[category]["products"] += 1
            
            # Log progress
            if (idx + 1) % 10 == 0 or idx == total_samples - 1:
                # Calculate running metrics
                overall = metrics["overall"]
                accuracy = overall["correct"] / overall["total"] if overall["total"] else 0
                avg_f1 = sum(overall["f1_scores"]) / len(overall["f1_scores"]) if overall["f1_scores"] else 0
                avg_precision = sum(overall["precision_scores"]) / len(overall["precision_scores"]) if overall["precision_scores"] else 0
                avg_recall = sum(overall["recall_scores"]) / len(overall["recall_scores"]) if overall["recall_scores"] else 0
                
                logger.info(f"Progress: {idx+1}/{total_samples} products processed")
                logger.info(f"Current metrics - F1: {avg_f1:.3f}, P: {avg_precision:.3f}, R: {avg_recall:.3f}")
                
                # Log to wandb
                wandb.log({
                    "progress": (idx + 1) / total_samples,
                    "overall_f1": avg_f1,
                    "overall_precision": avg_precision,
                    "overall_recall": avg_recall,
                    "overall_accuracy": accuracy,
                    "api_response_time": api_time
                })
            
            # Add delay to avoid hitting API rate limits
            time.sleep(0.5)
        
        # Calculate final metrics
        results = {"categories": {}}
        
        for category, category_metrics in metrics.items():
            if category_metrics["total"] == 0:
                continue
                
            accuracy = category_metrics["correct"] / category_metrics["total"]
            avg_bleu = sum(category_metrics["bleu_scores"]) / len(category_metrics["bleu_scores"]) if category_metrics["bleu_scores"] else 0
            avg_jaccard = sum(category_metrics["jaccard_scores"]) / len(category_metrics["jaccard_scores"]) if category_metrics["jaccard_scores"] else 0
            avg_f1 = sum(category_metrics["f1_scores"]) / len(category_metrics["f1_scores"]) if category_metrics["f1_scores"] else 0
            avg_precision = sum(category_metrics["precision_scores"]) / len(category_metrics["precision_scores"]) if category_metrics["precision_scores"] else 0
            avg_recall = sum(category_metrics["recall_scores"]) / len(category_metrics["recall_scores"]) if category_metrics["recall_scores"] else 0
            
            category_result = {
                "products": category_metrics["products"],
                "attributes": category_metrics["total"],
                "accuracy": accuracy,
                "bleu": avg_bleu,
                "jaccard": avg_jaccard,
                "f1": avg_f1,
                "precision": avg_precision,
                "recall": avg_recall
            }
            
            if category == "overall":
                results["overall"] = category_result
            else:
                results["categories"][category] = category_result
                
                # Log per-category metrics to wandb
                wandb.log({
                    f"category_{category}_f1": avg_f1,
                    f"category_{category}_precision": avg_precision,
                    f"category_{category}_recall": avg_recall,
                    f"category_{category}_products": category_metrics["products"]
                })
        
        return results

@weave.op()
def evaluate_oamine_random(model_name: str, total_samples: int, max_per_category: int) -> Dict:
    """Weave operation to evaluate OpenAI model on random samples from OA-mine dataset"""
    evaluator = OAmineEvaluator(model_name, total_samples, max_per_category)
    results = evaluator.run_evaluation()
    
    # Log final results to both logger and wandb
    if "overall" in results:
        overall = results["overall"]
        logger.info("===== EVALUATION COMPLETE =====")
        logger.info(f"Model: {model_name}")
        logger.info(f"Products evaluated: {overall['products']}")
        logger.info(f"Attributes evaluated: {overall['attributes']}")
        logger.info(f"Final F1 score: {overall['f1']:.4f}")
        logger.info(f"Final Precision: {overall['precision']:.4f}")
        logger.info(f"Final Recall: {overall['recall']:.4f}")
        
        # Final wandb update
        wandb.log({
            "final_f1": overall["f1"],
            "final_precision": overall["precision"],
            "final_recall": overall["recall"],
            "total_products": overall["products"],
            "total_attributes": overall["attributes"]
        })
    
    return results

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate OpenAI models on OA-mine dataset")
    parser.add_argument("--openai_model", type=str, default=OPENAI_MODEL, help="OpenAI model name")
    parser.add_argument("--total_samples", type=int, default=TOTAL_SAMPLES, help="Total number of products to evaluate")
    parser.add_argument("--max_per_category", type=int, default=MAX_SAMPLES_PER_CATEGORY, help="Maximum samples per category")
    args = parser.parse_args()
    
    # Configure run name and logging
    timestamp = datetime.datetime.now().strftime("%d-%m_%H:%M")
    run_name = f"{timestamp}_{args.openai_model}_random_{args.total_samples}"
    setup_logging(run_name)
    
    # Register signal handler
    signal.signal(signal.SIGTERM, toggle_intercepted)
    signal.signal(signal.SIGINT, toggle_intercepted)
    
    # Set up W&B
    if WANDB_KEY:
        wandb.login(key=WANDB_KEY)
    else:
        logger.warning("WANDB_KEY not found in environment. W&B logging may not work.")
    
    try:
        # Initialize wandb
        config = {
            "model_name": args.openai_model,
            "total_samples": args.total_samples,
            "max_per_category": args.max_per_category,
            "timestamp": timestamp
        }
        
        wandb.init(project="oamine-evaluation", name=run_name, config=config)
        weave.init(project_name="oamine-evaluation")
        
        # Run evaluation
        results = evaluate_oamine_random(
            args.openai_model, 
            args.total_samples,
            args.max_per_category
        )
        
        # Save results to file
        results_file = f"/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/logs/oamine/results_{run_name}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")
    finally:
        wandb.finish()
        logger.info("Evaluation finished")