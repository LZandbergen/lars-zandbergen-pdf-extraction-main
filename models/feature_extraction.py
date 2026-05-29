from typing import Dict, Optional
from pydantic import BaseModel, field_validator

class FeatureValue(BaseModel):
    """
    Represents a single feature value extracted from a product.
    
    Attributes:
        feature_name (str): Name of the feature
        value (Optional[str]): The extracted value, can be None if not found
    """
    feature_name: str
    value: Optional[str]

class FeatureExtractionResult(BaseModel):
    """
    Represents the complete set of features extracted for a product.
    
    Attributes:
        product_identifier (str): Identifier of the product
        features (Dict[str, Optional[str]]): Mapping of feature names to their values
    """
    product_identifier: str
    features: Dict[str, Optional[str]]  # Feature name to value mapping

    @field_validator('features', mode='before')
    def ensure_strings(cls, v: Dict[str, any]) -> Dict[str, Optional[str]]:
        """
        Ensures all feature values are converted to strings or None.
        
        Args:
            v: Dictionary of feature values
            
        Returns:
            Dictionary with all values converted to strings or None
        """
        return {k: cls.convert_to_string(v) for k, v in v.items()}

    def convert_to_string(value: any) -> Optional[str]:
        """
        Converts a value to string format, handling special cases.
        
        Args:
            value: Value to convert
            
        Returns:
            Converted string value or None
        """
        if value is None or value == "None":
            return None
        if isinstance(value, list):
            return ', '.join(map(str, value))
        return str(value)