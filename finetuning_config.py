"""
Configuration settings for fine-tuning models.
Contains default configurations and constants used throughout the training process.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Constants - fixed values that don't change based on runtime parameters
RESPONSE_TEMPLATE = "\n### Answer: \n"

TARGET_MODULES = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
SUBSET_SIZE = 10000
DEFAULT_SEED = 42
MAX_INPUT_LENGTH = 896  # Depends on dataset. 768 for MAVE, 896 (95th for OA mine)
MAX_RESPONSE_LENGTH = 100
WANDB_PROJECT_NAME = "oa-finetuning"
DEBUG_LEVEL = logging.DEBUG

DATA_PATH = "/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/data/OA-mine/dataset/stratified_split"
# DATA_PATH = "/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/data/OA-mine/dataset/category_split"
# DATA_PATH = "/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/data/MAVE/dataset"

# BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
# BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# System messages templates
SYSTEM_MSG_JSON = """You are an expert in extracting product features from technical documents.

The user will provide a template JSON showing the features they'd like to extract.
Your task is to fill in the "values" for each feature from the provided product information.
If a feature cannot be found, fill in "None" for its value. 
Only answer in JSON-formatted responses."""

SYSTEM_MSG_NEWLINES = """You are an expert in extracting product features from technical documents.

The user will provide a template showing the features they'd like to extract.
Your task is to fill in the "values" for each feature from the provided product information.
If a feature cannot be found, fill in "None" for its value. 
Only answer in newline-formatted responses."""

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning runs."""
    # Model parameters
    base_model: str = BASE_MODEL
    max_input_length: int = MAX_INPUT_LENGTH
    max_response_length: int = MAX_RESPONSE_LENGTH
    
    # Dataset parameters
    data_path: str = DATA_PATH
    subset_size: int = SUBSET_SIZE
    test_size: int = 100
    
    # Training parameters
    load_in_4bit: bool = False
    load_in_8bit: bool = True
    json_mode: bool = False
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    
    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Runtime parameters
    run_name: Optional[str] = None
    job_id: Optional[str] = None
    timestamp: Optional[str] = None
    quant_config: Optional[str] = None
    
    # Paths
    model_save_path: Optional[str] = None
    
    def __post_init__(self):
        """Set dependent parameters after initialization."""
        # Set quantization configuration
        if self.load_in_4bit:
            self.quant_config = "4bit"
        elif self.load_in_8bit:
            self.quant_config = "8bit"
        else:
            self.quant_config = "no_quant"
            
        # Get job ID from environment
        if not self.job_id:
            self.job_id = os.getenv("SLURM_JOB_ID", "local")
            
        # Set system message based on JSON mode
        self.system_message = SYSTEM_MSG_JSON if self.json_mode else SYSTEM_MSG_NEWLINES

def get_config_from_args(args):
    """Create a configuration object from parsed arguments."""
    config = TrainingConfig(
        base_model=args.base_model if hasattr(args, 'base_model') else TrainingConfig.base_model,
        subset_size=args.subset_size,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        run_name=args.run_name,
        data_path=args.data_path if hasattr(args, 'data_path') else TrainingConfig.data_path,
    )
    
    # Set JSON mode if specified in args
    if hasattr(args, 'JSON_mode'):
        config.json_mode = args.JSON_mode
    
    # Process data path for tracking
    config.data_path_wandb = config.data_path.split("data/", 1)[1] if "data/" in config.data_path else config.data_path
    
    return config

def prepare_wandb_config(config):
    """
    Prepare configuration dictionary for wandb tracking.
    Centralizes all parameters to track in wandb.
    
    Args:
        config: TrainingConfig object with configuration parameters
        
    Returns:
        Dictionary with parameters to track in wandb
    """
    # Hardware information (safely imported here to avoid circular imports)
    gpu_info = {}
    try:
        import torch
        gpu_info = {
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "None"
        }
    except ImportError:
        pass
    
    # Create dictionary with all relevant config parameters to track
    wandb_config = {
        # Model parameters
        "base_model": config.base_model,
        "max_input_length": config.max_input_length,
        "max_response_length": config.max_response_length,
        
        # Dataset parameters
        "subset_size": config.subset_size,
        "test_size": config.test_size,
        "data_path": config.data_path_wandb,
        
        # Training parameters
        "quant_config": config.quant_config,
        "load_in_4bit": config.load_in_4bit,
        "load_in_8bit": config.load_in_8bit,
        "json_mode": config.json_mode,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        
        # LoRA parameters
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        
        # Runtime parameters
        "job_id": config.job_id,
        "timestamp": config.timestamp,
        "model_save_path": config.model_save_path,
        
        # Hardware info (if available)
        **gpu_info
    }
    
    return wandb_config