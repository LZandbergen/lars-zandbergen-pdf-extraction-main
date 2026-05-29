import os
import json
import logging
import random
import argparse
import weave
import asyncio

from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from dotenv import load_dotenv
from tools import parse_products
from models.pdf_assistant import OpenAIAssistant, ClaudeAssistant
from models.local_assistant import LocalLLMAssistant, LocalPDFAssistant, FineTunedLlamaAssistant, PDFProcessor
from tools.product_processor import ProductProcessor
from tools.feature_validator import FeatureValidator, ModelFeatureValidator, feature_extraction_score

# Load environment variables
load_dotenv()

# Silence weave
os.environ["WANDB_SILENT"] = "true"
os.environ["WEAVE_SILENT"] = "true"

# Configuration constants
USE_LANGCHAIN = False
EXPLORE_FEATURES = False
MANUAL_SUBMISSION = False
NUMBER_OF_EXAMPLES = 0
STORE_CORRECT = False
TEST_SHOES = True
N_SUBMISSIONS = 1
MAX_ATTEMPTS = 3
SAVE_PROMPT_LOCATION = "prompt.txt"
WANDB_EVAL = True
USE_VISION = False

_feature_logger = None
_api_logger = None

AVAILABLE_MODELS = {
    "gpt-4o-mini": {
        "name": "GPT-4 Mini Optimized",
        "type": "openai",
        "model_path": "gpt-4o-mini"
    },
    "gpt-4o": {
        "name": "GPT-4 Optimized",
        "type": "openai",
        "model_path": "gpt-4o"
    },
    "claude-3.5": {
        "name": "Claude 3.5",
        "type": "anthropic",
        "model_path": "claude-3-5-sonnet-20241022"
    },
    "llama-3.1-8b": {
        "name": "LLAMA 3.1 8B",
        "type": "local_pdf",
        "model_path": "models/meta-llama/Llama-3.1-8B-Instruct/",
        "tokenizer_path": "models/meta-llama/Llama-3.1-8B-Instruct"
    },
    "llama-3.1-8b-finetuned": {
        "name": "LLAMA 3.1 8B Finetuned",
        "type": "local_finetuned",
        "model_path": "models/fine-tuned/llama-3-8b-ave-amazon"
    },
    "llama-3.1-8b-200k": {
        "name": "LLAMA 3.1 8B 200K samples",
        "type": "local_finetuned",
        "model_path": "models/fine-tuned/17-12-16:32-llama-3-8b-amazon-200000-samples-lora",
        "tokenizer_path": "models/meta-llama/Llama-3.1-8B-Instruct"
    }
}

def get_valid_filepath(prompt: str, default: str) -> str:
    """
    Get a valid file path from user input or default value.
    
    Args:
        prompt (str): Prompt to show user
        default (str): Default file path
        
    Returns:
        Joined path: Valid file path, will 
    """
    datapath = "data/" + ("allshoes_catalogus" if TEST_SHOES else "pilot_devices")
    while True:
        filepath = input(prompt) or default
        if os.path.isfile(os.path.join(datapath, filepath)):
            return filepath
        else:
            print(f"Invalid file path: {os.path.join(datapath, filepath)}. Please try again.")

def setup_logging() -> None:
    """Configure logging settings."""
    global _feature_logger, _api_logger
    
    os.makedirs('logs', exist_ok=True)
    file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    
    # Setup feature logger
    _feature_logger = logging.getLogger('feature_logger')
    feature_handler = logging.FileHandler('logs/missing_features.log')
    feature_handler.setFormatter(file_formatter)
    _feature_logger.addHandler(feature_handler)
    
    # Setup API logger
    _api_logger = logging.getLogger('api_logger')
    api_handler = logging.FileHandler('logs/api_requests.log')
    api_handler.setFormatter(file_formatter)
    _api_logger.addHandler(api_handler)
    
    # Prevent propagation
    _feature_logger.propagate = False
    _api_logger.propagate = False

def get_loggers() -> Tuple[logging.Logger, logging.Logger]:
    """Get the feature and API loggers."""
    global _feature_logger, _api_logger

    if _feature_logger is None or _api_logger is None:
        setup_logging()

    return _feature_logger, _api_logger

def get_pdf_path() -> str:
    """Get the PDF file path based on configuration."""
    if TEST_SHOES:
        return get_valid_filepath(
            "Enter the path to the PDF file: ",
            "/data/allshoes_catalogus/pdfs/allshoes_catalogus.pdf"
        )
    else:
        return get_valid_filepath(
            "Enter the path to the PDF file: ",
            "/data/pilot_devices/pdfs/pilot_devices.pdf"
        )
    
def get_product_ids_sample(only_verified_products, validator, feature_logger, product_identifiers) -> list:
    """
    Retrieves a list of product identifiers based on verification status. Uses sets to ensure randomization.
    Args:
        only_verified_products (bool): Flag indicating whether to filter only verified products.
        validator (Validator): An object that contains the correct features of verified products.
        feature_logger (Logger): Logger object for logging information and debug messages.
        product_identifiers (list): List of product identifiers to be filtered.
    Returns:
        list: A list of valid product identifiers based on the verification status.
    """
    if only_verified_products:
        verified_ids = set(validator.correct_features.keys())
        feature_logger.info(f"Number of verified products: {len(verified_ids)}")
        feature_logger.debug(f"Verfiied products: {verified_ids}")
        available_ids = set(product_identifiers) & verified_ids
        feature_logger.info(f"Number of available verified products: {len(available_ids)}")
        feature_logger.debug(f"Available verified products: {available_ids}")
        example_products = random.sample(list(available_ids), min(NUMBER_OF_EXAMPLES, len(available_ids)))
        valid_product_identifiers = list(available_ids - set(example_products))

        if len(valid_product_identifiers) < N_SUBMISSIONS:
            print(f"Not enough verified products to process. Submission count changed to {len(valid_product_identifiers)}.")
            feature_logger.warning(f"Not enough verified products to process. Submission count changed to {len(valid_product_identifiers)}.")
    else:
        valid_product_identifiers = product_identifiers
    return valid_product_identifiers

def process_manual_submission(processor: ProductProcessor, assistant: OpenAIAssistant, validator: FeatureValidator):
    """Handle manual product submission processing. Doesn't utilize examples"""
    while True:
        product_identifier = input("Enter the product identifier to process (or 'exit' to quit): ")
        if product_identifier.lower() == 'exit':
            return
        
        result = processor.process_product(product_identifier, assistant, [])
        if result:
            validation_results = validator.validate_product(product_identifier, result.features)
            if "error" not in validation_results:
                validator.log_validation_results(product_identifier, validation_results)

def process_batch_submissions(processor: ProductProcessor, assistant: OpenAIAssistant,
                            validator: FeatureValidator, n_submissions: int, only_verified_products: bool):
    """Handle batch processing of multiple products."""
    overall_results = []
    product_identifiers = [str(key) for key in processor.product_features.keys()] 
    feature_logger, _ = get_loggers()
    feature_logger.debug(f"Product_identifiers: {product_identifiers}")
    example_products = []

    valid_product_identifiers = get_product_ids_sample(only_verified_products, validator, feature_logger, product_identifiers)
    if not valid_product_identifiers:
        feature_logger.warning("No valid product identifiers found.")
        return overall_results

    for _ in tqdm(range(n_submissions)):
        product_identifier = valid_product_identifiers.pop()
        result = processor.process_product(product_identifier, assistant, example_products)
        if result:
            if only_verified_products:
                validation_results = validator.validate_product(product_identifier, result.features)
                if validation_results:
                    validator.log_validation_results(product_identifier, validation_results)
                    overall_results.append(validation_results)
                    feature_logger.debug(f"Validation results for product {product_identifier}: {validation_results}")
                else:
                    print(f"Validation skipped for product {product_identifier}.")
            else:
                print(f"Processed product {product_identifier}: {result.features}")

    return overall_results

def process_wandb_submission(processor: ProductProcessor, assistant: OpenAIAssistant, validator: FeatureValidator, correct_features_file, only_verified_products):
    model = ModelFeatureValidator(
            processor=processor,
            correct_features_path=correct_features_file,
            llm_model=assistant,
        )

    product_identifiers = [str(key) for key in processor.product_features.keys()] 
    feature_logger, _ = get_loggers()
    valid_product_identifiers = get_product_ids_sample(only_verified_products, validator, feature_logger, product_identifiers)

    # Create evaluation dataset from verified products
    examples = []
    for pid in valid_product_identifiers:
        if len(examples) >= N_SUBMISSIONS:
            feature_logger.warning(f"Reached maximum number of submissions: {N_SUBMISSIONS}")
            break
        examples.append({
            "product_id": pid,
            "expected": validator.correct_features[pid]
        })

    # Evaluate the model
    weave.init('evaluation-test')
    evaluation = weave.Evaluation(
        dataset=examples,
        scorers=[feature_extraction_score],
    )
    results = asyncio.run(evaluation.evaluate(model))

def get_model_choice() -> dict:
    """
    Prompt user to select a model from available options.
    
    Returns:
        dict: Selected model's properties
    """
    print("\nAvailable models:")
    for i, model_info in enumerate(AVAILABLE_MODELS.values()):
        print(f"{i + 1}: {model_info['name']}")
    
    while True:
        try:
            choice = input("\nSelect model number (default is 1): ").strip() or "1"
            index = int(choice) - 1
            if 0 <= index < len(AVAILABLE_MODELS):
                selected_model = list(AVAILABLE_MODELS.values())[index]
                return selected_model
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def parse_model(model_type: str, model_path: str, pdf_path: str, cache_dir: Path):
    """
    Parses and initializes a model assistant based on the specified model type.

    Args:
        model_type (str): The type of the model to be used. Options are 'openai', 'anthropic', 'local', 'local_pdf', and 'local_finetuned'.
        model_path (str): The path to the model.
        pdf_path (str): The path to the PDF file to be processed.
        cache_dir (Path): The directory path for caching purposes.

    Returns:
        assistant: An instance of the appropriate assistant class based on the model type.

    Raises:
        ValueError: If an unknown model type is provided.
    """
    if model_type == 'openai':
        weave.init(project_name='Openai', settings={'print_call_link': False})
        assistant = OpenAIAssistant(model_name=model_path)
        assistant.upload_file(pdf_path)
        print(f"OpenAI Assistant ID: {assistant.assistant_id}")
    elif model_type == 'anthropic':
        weave.init(project_name='Anthropic', settings={'print_call_link': False})
        assistant = ClaudeAssistant(model_name=model_path)
        assistant.upload_file(pdf_path)
    elif model_type == 'local_pdf':
        # Find the model info in AVAILABLE_MODELS to get the tokenizer path
        model_info = next((m for m in AVAILABLE_MODELS.values() if m['model_path'] == model_path), None)

        tokenizer_path = model_info.get('tokenizer_path', model_path) if model_info else model_path
        assistant = LocalPDFAssistant(model_name=model_path, tokenizer_path=tokenizer_path, cache_dir=str(cache_dir))
        assistant.upload_file(pdf_path)
    elif model_type == 'local_finetuned':
        assistant = FineTunedLlamaAssistant()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return assistant

def print_summary(overall_results, validator: FeatureValidator):
    """Print summary of processing results."""
    if not overall_results:
        print("No validation results available.")
        return
        
    success_rates = []
    for results in overall_results:
        if isinstance(results, dict):
            rate = validator.parse_validation_results(results)
            if rate > 0:
                success_rates.append(rate)
    
    if success_rates:
        avg_success_rate = sum(success_rates) / len(success_rates)
        print(f"\nValidation Summary:")
        print(f"Number of products processed: {len(overall_results)}")
        print(f"Products with valid results: {len(success_rates)}")
        print(f"Average Success Rate: {avg_success_rate:.2f}%")
    else:
        print("\nNo valid success rates could be calculated.")

def get_test_option() -> bool:
    """Ask whether to test only on products with verified features."""
    while True:
        choice = input("Do you want to test only on products that have at least some verified features (n)? Yes by default: ").strip().lower()
        if choice == 'n' or choice == 'no':
            return False
        else:
            return True

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Product Feature Extraction Pipeline')
    parser.add_argument('--add-new-products', action='store_true',
                        help='Add products that have not been stored yet')
    parser.add_argument('--product-path', type=str,
                        default='data/allshoes_catalogus/products.csv' if TEST_SHOES else 'data/pilot_devices/products.csv',
                        help='Path to the file with products')
    parser.add_argument('--pdf-path', type=str,
                        default='data/allshoes_catalogus/pdfs/allshoes_catalogus.pdf' if TEST_SHOES else 'data/pilot_devices/pdfs/pilot_devices.pdf',
                        help='Path to the PDF file')
    parser.add_argument('--manual-submission', action='store_true', default=False, help='Choose which products to process manually')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a local model to use')
    parser.add_argument('--only-verified', action='store_true',
                        help='Test only on products with verified features')
    parser.add_argument('--selected-column', type=str, default=None,
                        help='Specify the column to select')
    parser.add_argument('--use-vision', action='store_true', default=False,
                        help='Use vision-based prompt for product extraction')
    return parser.parse_args()

def add_new_products(processor: ProductProcessor, assistant: OpenAIAssistant):
    """Add products that have not been stored yet."""
    new_products = processor.get_unstored_products()
    if not new_products:
        print("No new products to add.")
        return
    
    for product_identifier in tqdm(new_products, desc="Adding New Products"):
        result = processor.process_product(product_identifier, assistant, [])
        if not result:
            print(f"Failed to add product {product_identifier}.")

def process_pdf(pdf_path: str, cache_dir: str) -> Path:
    """Process a PDF and return path to markdown file"""
    processor = PDFProcessor(cache_dir)
    
    # Generate cache path
    pdf_path = Path(pdf_path)
    cache_path = Path(cache_dir) / "markdown" / f"{pdf_path.stem}.md"
    
    # Process if not cached
    if not cache_path.exists():
        markdown_content = processor.process_pdf(str(pdf_path))
        cache_path.write_text(markdown_content)
    
    return cache_path

def main():
    """Main pipeline execution function."""
    # Setup loggers
    feature_logger, api_logger = get_loggers()
    
    # Parse command-line arguments
    args = parse_arguments()
    if args:
        STORE_CORRECT = args.add_new_products
        USE_VISION = args.use_vision

    if args.model_path:
        # Use the provided model path and assume it's a local model
        selected_model_info = {
            "name": os.path.basename(args.model_path),
            "type": "local_pdf",
            "model_path": args.model_path
        }
    else:
        selected_model_info = get_model_choice()

    selected_model_name = selected_model_info['name']
    print(f"\nUsing model: {selected_model_name}")
    
    # Ask the user for testing option
    only_verified_products = args.only_verified or get_test_option()
    if only_verified_products:
        weave.attributes({"model": selected_model_name, "max_attempts": MAX_ATTEMPTS, "example_count": NUMBER_OF_EXAMPLES})
    
    # Initialize processor
    basepath = "data/" + ("allshoes_catalogus" if TEST_SHOES else "pilot_devices")
    pdf_path = args.pdf_path  # Get pdf_path from arguments
    processor = ProductProcessor(
        basepath=basepath,
        test_shoes=TEST_SHOES,
        store_correct=STORE_CORRECT,
        use_langchain=USE_LANGCHAIN,
        max_attempts=MAX_ATTEMPTS,
        none_threshold=0.8,
        product_file_path=args.product_path,  
        pdf_path=pdf_path,  
        selected_column=args.selected_column,
        logger=feature_logger,
        use_vision=USE_VISION
    )
    
    # Create cache directory
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Initialize assistant based on model type
    model_type = selected_model_info.get('type')
    model_path = selected_model_info.get('model_path')
    
    assistant = parse_model(model_type, model_path, pdf_path, cache_dir)
    
    # Initialize validator
    correct_features_file = os.path.join(basepath, 'correct_product_features.json')
    validator = FeatureValidator(correct_features_file)
    
    if args.add_new_products:
        # Process and add new products that have not been stored yet
        add_new_products(processor, assistant)
    elif MANUAL_SUBMISSION:
        process_manual_submission(processor, assistant, validator)
    elif WANDB_EVAL:
        process_wandb_submission(processor, assistant, validator, correct_features_file, only_verified_products)
    else:
        overall_results = process_batch_submissions(processor, assistant, validator, N_SUBMISSIONS, only_verified_products)
        if only_verified_products:
            print_summary(overall_results, validator)
            # Generate and save validation report
            validator.generate_report(overall_results, output_file='logs/validation_report.txt')
            print("\nValidation report has been saved to logs/validation_report.txt")
        else:
            print("\nValidation skipped due to working with partly unverified products.")

if __name__ == "__main__":
    main()