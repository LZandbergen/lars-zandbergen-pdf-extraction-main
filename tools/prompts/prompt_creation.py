from .prompt_class import ProductPrompt

DEFAULT_SYSTEM = """You are an expert in extracting product features from technical documents."""

DEFAULT_USER = """Extract the following features for the product:

{features_description}

If a feature cannot be found, return "None" for its value.

Provide the output in JSON format with the feature names as keys and their corresponding values:

{{
    "feature_name_1": "value_1",
    "feature_name_2": "value_2",
    ...
}}

Here is the product information:

Identifier: {product_text}

And here are some examples of previous extractions:
{examples}"""

VISION_SYSTEM = """You are an expert in extracting product features from technical documents using both text and the images in the document."""

VISION_USER = """Extract the following features for the product:

{features_description}

Always first try to answer the question using the text. If a feature cannot be found, use the image to extract the feature.

Provide the output in JSON format with the feature names as keys and their corresponding values:

{{
    "feature_name_1": "value_1",
    "feature_name_2": "value_2",
    ...
}}

Here is the product information:

Identifier: {product_text}

And here are some examples of previous extractions:
{examples}"""

def get_default_prompt() -> ProductPrompt:
    prompt = ProductPrompt("product_extraction")
    prompt.add_version(
        system_content=DEFAULT_SYSTEM,
        user_template=DEFAULT_USER,
        description="Baseline prompt",
        version_name="default_version"
    )
    prompt.add_version(
        system_content=VISION_SYSTEM,
        user_template=VISION_USER,
        description="Vision-based prompt",
        version_name="vision_version"
    )
    return prompt