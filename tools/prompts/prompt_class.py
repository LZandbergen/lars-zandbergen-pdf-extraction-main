from dataclasses import dataclass
from typing import Dict, List, Optional
import weave
from datetime import datetime
import hashlib

@dataclass
class PromptVersion:
    system_content: str
    user_template: str
    version: str
    created_at: datetime
    description: str
    metrics: Optional[Dict] = None

class ProductPrompt:
    def __init__(self, name: str):
        weave.init('product-extraction')
        self.name = name
        self.prompt = weave.MessagesPrompt(name=name)
        self._current_version = None
        self.versions = {}
    
    def add_version(self, system_content: str, user_template: str, description: str, version_name: Optional[str] = None) -> str:
        """Add new prompt version with a specified version name"""
        # Generate a hash of the prompt contents
        prompt_content = system_content + user_template
        version_hash = hashlib.md5(prompt_content.encode('utf-8')).hexdigest()
        version_id = version_name or version_hash

        if version_id in self.versions:
            # Prompt version already exists
            self._current_version = self.versions[version_id]
            return version_id

        # Create new prompt version
        prompt_version = PromptVersion(
            system_content=system_content,
            user_template=user_template,
            version=version_id,
            created_at=datetime.now(),
            description=description
        )
        self.versions[version_id] = prompt_version
        self._current_version = prompt_version
        weave.publish(
            self.prompt,
            name=f"{self.name}_v{version_id}",
        )
        return version_id

    def set_current_version(self, version_name: str):
        """Set the current prompt version by name"""
        if version_name in self.versions:
            self._current_version = self.versions[version_name]
        else:
            raise ValueError(f"Prompt version '{version_name}' does not exist.")

    def format(self, product_text: str, features_to_extract: List[Dict], examples: str, additional_context: Optional[Dict] = None) -> List[Dict]:
        """Format prompt with the current version and additional context"""
        features_description = self._format_features(features_to_extract)
        user_content = self._current_version.user_template.format(
            features_description=features_description,
            product_text=product_text,
            examples=examples,
            **(additional_context or {})
        )
        return [
            {"role": "system", "content": self._current_version.system_content},
            {"role": "user", "content": user_content}
        ]

    def _format_features(self, features_to_extract: List[Dict]) -> str:
        """Format features description"""
        features_description = ""
        for feature in features_to_extract:
            # If there is a strict selection of options for the feature, add them to the description
            if feature['options']:
                options_str = ', '.join(f'"{opt}"' for opt in feature['options'])
                features_description += f"- {feature['feature_name']}: Choose from options: {options_str}.\n"
            else:
                features_description += f"- {feature['feature_name']}: \n"
        return features_description