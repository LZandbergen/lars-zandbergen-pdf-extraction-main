import json
import logging
import weave
import os
import sys
import glob
import logging
from pydantic import ValidationError
from tools.prompts.prompt_creation import get_default_prompt
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

# Add parent directory to Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from models import FeatureExtractionResult
from tools import get_class_info, parse_products

class ProductProcessor:
    def __init__(self, basepath: str, test_shoes: bool = True, store_correct: bool = False,
                 use_langchain: bool = False, max_attempts: int = 3, none_threshold: float = 0.8,
                 product_file_path: str = 'products.json', pdf_path: str = None, save_prompt_location: str = None,
                 save_extraction_results: bool = False, selected_column: str = None, logger: logging.Logger = logging.Logger(name='ProductProcessor'),
                 use_vision: bool = False):
        """
        Initialize the ProductProcessor with configuration settings.
        
        Args:
            basepath (str): Base path for data files
            test_shoes (bool): Whether processing shoes data
            store_correct (bool): Whether to store correct features
            use_langchain (bool): Whether to use LangChain
            max_attempts (int): Maximum number of retry attempts
            none_threshold (float): Maximum acceptable proportion of None values
            product_file (str): Path to the product file
            pdf_path (str): Path to the PDF file
        """
        self.basepath = basepath
        self.test_shoes = test_shoes
        self.store_correct = store_correct
        self.use_langchain = use_langchain
        self.max_attempts = max_attempts
        self.none_threshold = none_threshold
        self.product_file_path = product_file_path
        self.pdf_path = pdf_path
        self.store_prompt = save_prompt_location
        self.output_dir = f"output/{basepath.split('/')[-1]}"
        self.prompt = get_default_prompt()
        self.save_extraction_results = save_extraction_results
        self.logger = logger
        self.use_vision = use_vision
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize resources
        self.product_features = parse_products.get_product_data(self.product_file_path, selected_column)

        weave.init('openai')
        
        if not test_shoes:
            self._init_taxonomy_resources()
    
    def _init_taxonomy_resources(self):
        """Initialize taxonomy-related resources for non-shoes products."""
        with open('data/taxonomy/taxonomy_data.json', 'r', encoding='utf-8') as f:
            self.taxonomy_data = json.load(f)
        self.product_class_code_df = pd.read_csv(self.basepath + "/export_lars_classes.csv")
    
    def _get_features_to_extract(self, product_identifier: str) -> Tuple[List[Dict], str]:
        """
        Get the features to extract for a product.
        
        Args:
            product_identifier (str): The product identifier
            
        Returns:
            Tuple[List[Dict], str]: Features to extract and product text
        """
        if not self.test_shoes:
            return self._get_taxonomy_features(product_identifier)
        else:
            return self._get_shoe_features(product_identifier)
    
    def _get_taxonomy_features(self, product_identifier: str) -> Tuple[List[Dict], str]:
        """Get features for taxonomy-based products."""
        class_code_row = self.product_class_code_df[
            self.product_class_code_df['product_identifier'] == product_identifier]
        
        if (class_code_row.empty):
            raise ValueError(f"No class code found for product identifier {product_identifier}")
            
        class_code = class_code_row['taxonomy_class_code'].values[0]
        class_name, features_dict = get_class_info.get_class_info(
            self.taxonomy_data, class_code)
        
        features_to_extract = []
        for feature_code in features_dict:
            feature_name, options = self._get_feature_options(feature_code, features_dict)
            if feature_name:
                features_to_extract.append({
                    'feature_code': feature_code,
                    'feature_name': feature_name,
                    'options': options
                })
        
        product_text = product_identifier + self.product_features[product_identifier]['product_description']
        return features_to_extract, product_text
    
    def _get_shoe_features(self, product_identifier: str) -> Tuple[List[Dict], str]:
        """Get features for shoe products."""
        with(open(self.basepath + '/taxonomy' + '/features.txt', 'r', encoding='utf-8')) as f:
            feature_names = f.read().splitlines()

        features_to_extract = [
            {'feature_name': name, 'options': None} for name in feature_names
        ]
        return features_to_extract, str(product_identifier)
    
    def _get_feature_options(self, feature_code: str, features_dict: dict) -> Tuple[Optional[str], Optional[List[str]]]:
        """Get feature name and options for a feature code."""
        feature_data = features_dict.get(feature_code)
        if not feature_data:
            return None, None
            
        feature_name = feature_data['feature_name']
        values = feature_data.get('values', {})
        options_list = list(values.keys()) if values else None
        
        return feature_name, options_list
    
    def _calculate_none_proportion(self, features: dict) -> float:
        """Calculate the proportion of None values in the features."""
        if not features:
            return 1.0
        none_count = sum(1 for value in features.values() 
                        if value is None or value == "None" or value == "none")
        return none_count / len(features)
    
    def _save_result(self, result: FeatureExtractionResult, product_identifier: str, 
                     product_text: str) -> None:
        """Save the extraction result to files."""
        if self.save_extraction_results:
            # Save to output directory
            output_path = f"{self.output_dir}/{product_identifier}.json"
            output_folder = os.path.dirname(output_path)
            os.makedirs(output_folder, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.model_dump_json(), f, ensure_ascii=False, indent=4)
            
        # Store correct features if enabled
        if self.store_correct:
            self._store_correct_features(product_identifier, product_text, result.features)

        # Store the prompt if enabled
        if self.store_prompt:
            with open((output_path + self.store_prompt), 'w', encoding='utf-8') as f:
                f.write(self.prompt)
    
    def _store_correct_features(self, product_identifier: str, product_text: str, 
                              result: Dict[str, Any]) -> None:
        """Store the correct features for manual labeling."""
        correct_product_features_path = self.basepath + '/correct_product_features.json'
    
        # Convert product_identifier to string
        product_identifier = str(product_identifier)
        
        # Load existing correct features
        if os.path.exists(correct_product_features_path):
            with open(correct_product_features_path, 'r', encoding='utf-8') as f:
                correct_product_features = json.load(f)
        else:
            correct_product_features = {}
        
        print(f"product_identifier: {product_identifier} is in correct_product_features: {product_identifier in correct_product_features}")
        
        # Get existing product data or initialize new entry
        if product_identifier in correct_product_features:
            product_entry = correct_product_features[product_identifier]
            verified_values = product_entry.get('values', {})
            print(f"verified_values: {verified_values}")
        else:
            verified_values = {}
            product_entry = {
                "product_description": product_text,
                "values": verified_values,
                "unverified_features": {}
            }
        
        # Process new results - only add to unverified if not already verified
        unverified_features = {}
        for feature, value in result.items():
            if feature not in verified_values:
                unverified_features[feature] = value
        
        # Update the product entry - keeping verified values intact
        product_entry["unverified_features"] = unverified_features
        correct_product_features[product_identifier] = product_entry
        
        # Store new version of the correct product features
        with open(correct_product_features_path, 'w', encoding='utf-8') as f:
            json.dump(correct_product_features, f, ensure_ascii=False, indent=4, sort_keys=True)
    
    @weave.op(name='Process product')
    def process_product(self, product_identifier: str, assistant, examples) -> Optional[FeatureExtractionResult]:
        """
        Process a product with retry logic for high None-value responses.
        
        Args:
            product_identifier (str): The product identifier
            assistant: The PDF assistant instance
            
        Returns:
            Optional[FeatureExtractionResult]: The extraction result, or None if failed
        """
        attempts = 0
        best_result = None
        best_none_proportion = 1.0
        
        try:
            features_to_extract, product_text = self._get_features_to_extract(product_identifier)
        except ValueError as e:
            print(f"Error getting features: {e}")
            return None
        
        while attempts < self.max_attempts:
            if attempts > 0:
                print(f"\nRetry attempt {attempts} for product {product_identifier}")
            
            try:
                # Create and send prompt
                prompt_text = self._create_prompt(product_text, features_to_extract, examples, use_vision=self.use_vision)
                extracted_features = self._get_model_response(assistant, prompt_text)
                
                if not extracted_features:
                    print("No response from model")
                    attempts += 1
                    continue
                
                # Calculate proportion of None values
                none_proportion = self._calculate_none_proportion(extracted_features)
                
                # Update best result if this attempt has fewer Nones
                if none_proportion < best_none_proportion:
                    best_none_proportion = none_proportion
                    best_result = extracted_features
                
                # If we have a good enough result, break
                if none_proportion < self.none_threshold:
                    break
                else:
                    self.logger.info(f"Model had too many none proportions. None proportion: {none_proportion:.1%}")

                    self.logger.degug(f"Output of features extracted:\n {extracted_features}")
                    
                attempts += 1
                
            except Exception as e:
                error_message = str(e).lower()
                if "file not found" in error_message or "vector store is empty" in error_message:
                    print("PDF file not found in assistant's vector storage. Re-uploading the PDF file.")
                    assistant.upload_file(self.pdf_path)
                    attempts += 1
                    continue
                else:
                    print(f"Error processing product {product_identifier} (attempt {attempts + 1}): {e}")
                    attempts += 1
        
        if best_result is None:
            print(f"Failed to get valid results for product {product_identifier} after {self.max_attempts} attempts")
            return None
        
        if best_none_proportion >= self.none_threshold:
            print(f"Warning: Best result for product {product_identifier} still has {best_none_proportion:.1%} None values")
        
        try:
            result = FeatureExtractionResult(
                product_identifier=str(product_identifier),
                features=best_result
            )
            self._save_result(result, product_identifier, product_text)
            return result
            
        except ValidationError as e:
            print(f"Validation error for product {product_identifier}: {e}")
            logging.error(f"Validation error for product {product_identifier}: {e}")
            return None
    
    def _create_prompt(self, product_text: str, features_to_extract: List[Dict], examples, use_vision: bool = True) -> str:
        """Create the prompt for the model."""
        additional_context = {}
        if use_vision:
            self.prompt.set_current_version("vision_version")
        else:
            self.prompt.set_current_version("default_version")

        messages = self.prompt.format(
            product_text=product_text,
            features_to_extract=features_to_extract,
            examples=examples,
            additional_context=additional_context
        )

        prompt_content = "\n".join([msg["content"] for msg in messages])
        with open("prompt.txt", "w") as f:
            f.write(prompt_content)
        return prompt_content
    
    def _get_model_response(self, assistant, prompt_text: str) -> Optional[Dict]:
        """Get and parse the model's response."""
        if self.use_langchain:
            from langchain import chains
            chain = chains.LLMChain(llm=llm, prompt=prompts.PromptTemplate(
                template=prompt_text, input_variables=[]))
            output = chain.run()
        else:
            answers = assistant.predict(prompt_text)
            output = answers[0] if answers else None
            if output:
                output = output[output.find("{"):output.rfind("}")+1]
        
        try:
            return json.loads(output) if output else None
        except json.JSONDecodeError:
            return None

    def get_unstored_products(self) -> List[str]:
        """Get a list of product identifiers that have not been processed yet."""
        processed_files = glob.glob(f'{self.output_dir}/*.json')
        processed_ids = {os.path.splitext(os.path.basename(f))[0] for f in processed_files}
        all_product_ids = set(self.product_features.keys())
        unstored_products = list(all_product_ids - processed_ids)
        return unstored_products