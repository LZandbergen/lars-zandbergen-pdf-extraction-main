import torch, wandb, weave
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from dotenv import load_dotenv
import json
from pydantic import BaseModel, ValidationError
load_dotenv()
import sys, os

USE_ADAPTERS = False
JSON_MODE = True
FORCE_A6000 = False 

if FORCE_A6000:
    # Check for RTX A6000 GPU and ensure it is idling
    def get_idle_rtx_a6000_device():
        for i in range(torch.cuda.device_count()):
            if "RTX A6000" in torch.cuda.get_device_name(i):
                memory_allocated = torch.cuda.memory_allocated(i)
                if memory_allocated == 0:
                    return i
        return None

    device_index = get_idle_rtx_a6000_device()
    if device_index is None:
        print("Idle RTX A6000 GPU not found. Exiting.")
        sys.exit(1)

    device = torch.device(f"cuda:{device_index}")

device = torch.device(f"cuda")

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from data.MAVE.sample_mave import load_index, get_record_by_id, get_combined_text

wandb.login(key=os.getenv("WANDB_KEY"))
os.environ["WANDB_SILENT"] = "true"

index = load_index('data/MAVE/index/merged_positives_index.json')
record = get_record_by_id('data/MAVE/merged/merged_positives.jsonl', index, '6318962387')
combined_text = get_combined_text(record)

tokenizer = AutoTokenizer.from_pretrained("/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/models/meta-llama/Llama-3.1-8B-Instruct")
adapter_dir = "/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/models/fine-tuned/llama-3-8b-amazon-200000-samples-15-38-17-12-15:38-4bit-lora/checkpoint-5440"

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    # "meta-llama/Llama-3.2-3B-Instruct",
    "/vol/csedu-nobackup/project/lzandbergen-storage/lars-zandbergen-pdf-extraction/models/meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map={"": device}
)

if USE_ADAPTERS:
    # Load and apply the PEFT adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        torch_dtype=torch.bfloat16
    )

    # Merge the adapter weights with the base model
    model = model.merge_and_unload()
else:
    model = base_model

pipe = pipeline(
    "text-generation",
    model=model,  # Use the merged model instead of PeftModel
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    max_new_tokens=200,
    truncation=True,
    num_return_sequences=1,
    temperature=0.5,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

feature_names = [f'    "{item["key"]}" : "value_{i+1}"' for i, item in enumerate(record['labels'])]
feature_names = "\n".join(feature_names)


if JSON_MODE:
    system_message = """You are an expert in extracting product features from technical documents.
You must respond ONLY with valid JSON, no other text.
The JSON must exactly match the template structure provided.
For any features that cannot be found, use null instead of "None".
Do not include any explanations or additional text outside the JSON structure."""
    user_message = f"""Here is the product information:
\n\n{combined_text}

Provide the output in new lines with the feature names as keys and their corresponding "values",
Respond with only the features that are requested.

Requested features:
{{
{feature_names}
}}
"""
else:
    system_message = """You are an expert in extracting product features from technical documents.
For any features that cannot be found, use null instead of "None".
Please provide the output with a new line for each feature, using the following formatting:
    feature_name: feature_value
Do not include any explanations or additional text besides this information
"""

    user_message = f"""Here is the product information:
    \n\n{combined_text}

Provide the output in new lines with the feature names as keys and their corresponding "values",
Respond with only the features that are requested.
Requested features:
{feature_names}
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False)

# Initialize weave project
weave.init('Mave-local-llama-3b-testing')

@weave.op()
def generate_features(prompt: str, pipe) -> dict:
    chat = pipe(prompt)
    response = chat[0]["generated_text"]
    
    # Debug print
    print("\n====== Raw Response ======")
    print(response)
    print("========================\n")

    
    actual_response = response[-1]

    result = {"response": actual_response, "prompt": prompt}

    if JSON_MODE:
        try:
            features = json.loads(actual_response)
            result["features"] = features
        except json.JSONDecodeError as e:
            result["error"] =  f"Error parsing JSON: {e}"
        except ValidationError as e:
            result["error"] =  f"Error validating JSON: {e}"
    else:
        split_response = actual_response.split("\n")
        if len(split_response) == len(record['labels']):
            result["features"] = {}
            for i, line in enumerate(split_response):
                key, value = line.split(":")
                result["features"][key.strip()] = value.strip()
        else:
            result["error"] = "Number of features does not match the number of labels"

    return result

result = generate_features(prompt, pipe)
if JSON_MODE and "features" in result:
    print(f"######### Parsed JSON #########: \n{json.dumps(result['features'], indent=2)}\n")
elif "error" in result:
    print(f"######### Error parsing response #########: \n{result['error']}\n")
    print(f"######### Raw Response #########: \n{result['response']}\n")
elif "features" in result:
    print(f"######### Features #########: \n{result['features']}\n")
else:
    print("Failed all other checks")
print(f"MAVE Label: {record['labels']}")