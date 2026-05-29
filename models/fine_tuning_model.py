from weave import Model
import weave
from dataclasses import dataclass
from typing import Dict, Any

class LlamaFineTuningModel(Model):
    # Training configuration
    base_model: str
    subset_size: int
    num_epochs: int
    learning_rate: float
    lora_config: Dict[str, Any]
    
    # Model configuration
    quantization_config: Dict[str, Any]
    max_length: int
    
    @weave.op()
    def predict(self, input_text: str) -> dict:
        # For monitoring inference
        return {'prediction': self.model(input_text)}
    
    @weave.op()
    def log_training_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        # For monitoring training progress
        return metrics
