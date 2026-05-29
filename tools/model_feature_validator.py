"""
model_feature_validator.py
"""

import weave
from weave import Model
from typing import Any
from tools.product_processor import ProductProcessor
from tools.feature_validator import FeatureValidator

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