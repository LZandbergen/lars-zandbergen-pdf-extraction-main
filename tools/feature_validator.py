import json
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import logging
import os
import weave
from weave import Evaluation, Model, Scorer
from tools.product_processor import ProductProcessor

class FeatureValidator:
    def __init__(self, correct_features_path: str):
        """
        Initialize the validator with the path to the correct features JSON file.
        
        Args:
            correct_features_path (str): Path to the JSON file containing correct feature values
        """
        self.correct_features_path = correct_features_path
        # Check if the file exists, otherwise create it
        if not os.path.exists(correct_features_path):
            self.correct_features = {}
        else:
            with open(correct_features_path, 'r', encoding='utf-8') as f:
                all_features = json.load(f)
                # Filter out products with empty 'values' fields
                self.correct_features = {k: v for k, v in all_features.items() if v.get('values')}

        # Set up logging
        os.makedirs('logs', exist_ok=True)
        
        # Feature validation logger
        feature_logger = logging.getLogger('feature_validator')
        feature_handler = logging.FileHandler('logs/missing_features.log')
        feature_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
        feature_logger.addHandler(feature_handler)
        feature_logger.setLevel(logging.INFO)
        
        # API request logger
        api_logger = logging.getLogger('api')
        api_handler = logging.FileHandler('logs/api_requests.log')
        api_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
        api_logger.addHandler(api_handler)
        api_logger.setLevel(logging.INFO)
        
        # Track statistics
        self.total_comparisons = 0
        self.feature_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'missing': 0, 'total': 0})
        
    def normalize_value(self, value: Any) -> str:
        """
        Normalize feature values for comparison.
        
        Args:
            value: The value to normalize
            
        Returns:
            str: Normalized value
        """
        if value is None:
            return None
        
        # Convert to string and normalize
        value = str(value).lower().strip()
        # Remove common variations
        value = value.replace('-', ' ')
        value = value.replace('_', ' ')
        # Remove multiple spaces
        value = ' '.join(value.split())
        
        return value
    
    def normalize_product_id(self, product_id: Any) -> str:
        """
        Normalize product ID to ensure consistent comparison.
        
        Args:
            product_id: The product ID to normalize
            
        Returns:
            str: Normalized product ID
        """
        return str(product_id).strip()
    
    def calculate_feature_match(self, predicted: str, correct: str) -> Tuple[bool, float]:
        """
        Calculate if two feature values match and their similarity score.
        
        Args:
            predicted (str): Predicted feature value
            correct (str): Correct feature value
            
        Returns:
            Tuple[bool, float]: (exact_match, similarity_score)
        """
        if predicted is None and correct is None:
            return True, 1.0
        if predicted is None or correct is None:
            return False, 0.0
            
        predicted = self.normalize_value(predicted)
        correct = self.normalize_value(correct)
        
        # Exact match
        if predicted == correct:
            return True, 1.0
            
        # Partial match (one is substring of other)
        if predicted in correct or correct in predicted:
            return False, 0.5
            
        # No match
        return False, 0.0
    
    @weave.op()
    def validate_product(self, product_id: str, output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate predicted features against correct features for a single product.
        
        Args:
            product_id (str): Product identifier
            output (Dict[str, Any]): Predicted feature values
            
        Returns:
            Optional[Dict[str, Any]]: Validation results or None if correct features are not available
        """
        normalized_id = self.normalize_product_id(product_id)
        
        # Check if product ID exists in correct features
        if normalized_id not in self.correct_features:
            # Attempt to normalize ID further
            normalized_id_int = normalized_id.lstrip('0')
            if normalized_id_int and normalized_id_int in self.correct_features:
                normalized_id = normalized_id_int
            else:
                print(f"Correct features not found for product ID {product_id}. Skipping validation.")
                return None  # Skip validation if not found
        
        correct_values = self.correct_features[normalized_id]['values']
        results = {
            'product_id': normalized_id,
            'features': {},
            'metrics': {
                'exact_matches': 0,
                'partial_matches': 0,
                'mismatches': 0,
                'missing_predictions': 0,
                'total_features': len(correct_values)
            }
        }
        
        # Compare each feature
        for feature_name, correct_value in correct_values.items():
            predicted_value = output.get(feature_name)
            
            # Track feature-specific stats
            self.feature_stats[feature_name]['total'] += 1
            
            if predicted_value is None:
                results['features'][feature_name] = {
                    'status': 'missing',
                    'predicted': None,
                    'correct': correct_value,
                    'similarity': 0.0
                }
                results['metrics']['missing_predictions'] += 1
                self.feature_stats[feature_name]['missing'] += 1
                continue
                
            exact_match, similarity = self.calculate_feature_match(predicted_value, correct_value)
            
            if exact_match:
                status = 'exact_match'
                results['metrics']['exact_matches'] += 1
                self.feature_stats[feature_name]['correct'] += 1
            elif similarity > 0:
                status = 'partial_match'
                results['metrics']['partial_matches'] += 1
                self.feature_stats[feature_name]['incorrect'] += 1
            else:
                status = 'mismatch'
                results['metrics']['mismatches'] += 1
                self.feature_stats[feature_name]['incorrect'] += 1
            
            results['features'][feature_name] = {
                'status': status,
                'predicted': predicted_value,
                'correct': correct_value,
                'similarity': similarity
            }
            
        self.total_comparisons += 1
        return results
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get overall validation statistics.
        
        Returns:
            Dict[str, Any]: Overall statistics
        """
        stats = {
            'total_products_validated': self.total_comparisons,
            'feature_performance': {},
            'overall_accuracy': 0.0
        }
        
        total_correct = 0
        total_features = 0
        
        for feature_name, feature_stats in self.feature_stats.items():
            accuracy = feature_stats['correct'] / feature_stats['total'] if feature_stats['total'] > 0 else 0
            stats['feature_performance'][feature_name] = {
                'accuracy': accuracy,
                'correct': feature_stats['correct'],
                'incorrect': feature_stats['incorrect'],
                'missing': feature_stats['missing'],
                'total': feature_stats['total']
            }
            total_correct += feature_stats['correct']
            total_features += feature_stats['total']
        
        stats['overall_accuracy'] = total_correct / total_features if total_features > 0 else 0
        return stats

    @weave.op()
    def generate_report(self, results: List[Dict[str, Any]], output_file: str = None) -> str:
        """
        Generate a detailed validation report.
        
        Args:
            results (List[Dict[str, Any]]): List of validation results
            output_file (str, optional): Path to save the report
            
        Returns:
            str: Report text
        """
        report = []
        report.append("=== Feature Extraction Validation Report ===\n")
        
        # Overall statistics
        overall_stats = self.get_overall_stats()
        report.append(f"Total Products Validated: {overall_stats['total_products_validated']}")
        report.append(f"Overall Accuracy: {overall_stats['overall_accuracy']:.2%}\n")
        
        # Feature-level performance
        report.append("Feature-level Performance:")
        for feature, stats in overall_stats['feature_performance'].items():
            report.append(f"\n{feature}:")
            report.append(f"  Accuracy: {stats['accuracy']:.2%}")
            report.append(f"  Correct: {stats['correct']}/{stats['total']}")
            report.append(f"  Incorrect: {stats['incorrect']}")
            report.append(f"  Missing: {stats['missing']}")
        
        # Sample of errors
        report.append("\nSample of Errors:")
        error_samples = []
        for result in results:
            for feature, details in result['features'].items():
                if details['status'] != 'exact_match':
                    error_samples.append(
                        f"\nProduct {result['product_id']} - {feature}:"
                        f"\n  Predicted: {details['predicted']}"
                        f"\n  Correct: {details['correct']}"
                        f"\n  Status: {details['status']}"
                    )
                if len(error_samples) >= 10:  # Limit to 10 examples
                    break
            if len(error_samples) >= 10:
                break
        
        report.extend(error_samples)
        
        report_text = '\n'.join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text

    def log_validation_results(self, product_id: str, validation_results: Dict[str, Any]):
        """
        Log the validation results for a product.
        
        Args:
            product_id (str): Product identifier
            validation_results (Dict[str, Any]): Validation results to log
        """
        total_string = ""
        for feature, details in validation_results['features'].items():
            temp_string = f"{feature}: {details['status']}: {details['predicted']} (correct: {details['correct']})\n"
            total_string += temp_string

        log_entry = {
            'product_id': product_id,
            'validation_results': total_string,
        }
        logging.info(json.dumps(log_entry, indent=4))

    def parse_validation_results(self, results: dict) -> float:
        """
        Parse validation results and return success rate as percentage.
        
        Args:
            results (dict): Validation results dictionary
            
        Returns:
            float: Success rate for exact matches as percentage (0-100)
        """
        if not results:
            logging.warning("No validation results available.")
            return 0.0
            
        matches = results["metrics"]["exact_matches"]
        total = results["metrics"]["total_features"]
        logging.info(f"Validation results: {matches} exact matches out of {total} total features.")
        
        if total == 0:
            return 0.0
            
        return (matches / total) * 100.0

    async def run_evaluation(self, test_examples: List[Dict[str, Any]], output_file: str = None) -> None:
        """
        Run evaluation on a set of test examples using Weave.
        
        Args:
            test_examples (List[Dict[str, Any]]): List of test examples with product_id and features
            output_file (str, optional): Path to save evaluation results
        """
        model = ModelFeatureValidator(self.correct_features_path)
        evaluation = Evaluation(
            dataset=test_examples,
            scorers=[feature_extraction_score],
            name="Feature Extraction Evaluation"
        )

        weave.init('feature-extraction')
        results = await evaluation.evaluate(model)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)

        return results

    def create_evaluation_examples(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create evaluation examples from product data.
        
        Args:
            products (List[Dict[str, Any]]): List of products with features
            
        Returns:
            List[Dict[str, Any]]: Formatted examples for evaluation
        """
        examples = []
        for product in products:
            product_id = self.normalize_product_id(product['product_id'])
            if product_id in self.correct_features:
                examples.append({
                    'product_id': product_id,
                    'features': product['features'],
                    'expected': self.correct_features[product_id]['values']
                })
        return examples

@weave.op()
def feature_extraction_score(expected: dict, output: dict) -> dict:
    """Score the feature extraction results."""
    matches = 0
    total = len(expected['values'])

    for feature, correct_value in expected['values'].items():
        predicted = output.features.get(feature)
        logging.debug(f"Feature: {feature}, Predicted: {predicted}, Correct: {correct_value}")
        if predicted == correct_value:
            matches += 1
            
    return {
        'exact_matches': matches,
        'total': total,
        'accuracy': matches / total if total > 0 else 0
    }

class ModelFeatureValidator(Model):
    """Model wrapper for feature validation with PDF support."""
    processor: ProductProcessor
    correct_features_path: str
    llm_model: Any
    validator: FeatureValidator

    def __init__(self, processor: ProductProcessor, correct_features_path: str, 
                 llm_model: Any):
        super().__init__(
            processor=processor,
            correct_features_path=correct_features_path,
            llm_model=llm_model,
            validator=FeatureValidator(correct_features_path),
        )
        self.processor = processor
        self.validator = FeatureValidator(correct_features_path)
        self.llm_model = llm_model

    @weave.op()
    def predict(self, product_id: str) -> dict:
        """Predict features for a given product."""
        # product_id = product_data['product_id']
        
        # Get prediction using processor
        result = self.processor.process_product(
            product_identifier=product_id,
            assistant=self.llm_model,
            examples=[]
        )
        
        if not result:
            return {}
        return result
        # return self.validator.validate_product(product_id, result.features)

class CustomModelFeatureValidator(Model):
    """
    Example custom validator if you do not wish to directly subclass
    or alter ModelFeatureValidator from feature_validator. 
    """
    def __init__(self, processor: ProductProcessor, correct_features_path: str, llm_model: Any):
        super().__init__()
        self.processor = processor
        self.correct_features_path = correct_features_path
        self.llm_model = llm_model
        self.validator = FeatureValidator(correct_features_path)

    @weave.op()
    def predict(self, product_id: str) -> dict:
        # Prepare prompt & generate
        result = self.processor.process_product(
            product_identifier=product_id,
            assistant=self.llm_model,
            examples=[]
        )
        if not result:
            return {}
        return {"features": result.features}