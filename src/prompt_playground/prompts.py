import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any


class PromptTemplate:
    """
    A template for prompts with variable substitution support.
    """

    def __init__(self, template: str, variables: List[str]):
        """
        Initialize a prompt template.

        Args:
            template: Template string with placeholders like {variable_name}.
            variables: List of variable names used in the template.
        """
        self.template = template
        self.variables = variables

    def fill(self, **kwargs) -> str:
        """
        Fill the template with provided variable values.

        Args:
            **kwargs: Variable values to substitute into the template.

        Returns:
            Filled template string.

        Raises:
            ValueError: If required variables are missing.
        """
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {', '.join(missing)}")

        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template contains undefined variable: {e}")

    def validate(self) -> bool:
        """
        Ensure all required variables are present in the template.

        Returns:
            True if template is valid, False otherwise.
        """
        pattern = r"\{([^}]+)\}"
        found_vars = set(re.findall(pattern, self.template))
        expected_vars = set(self.variables)

        return found_vars == expected_vars

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert template to dictionary for serialization.

        Returns:
            Dictionary representation of the template.
        """
        return {
            "template": self.template,
            "variables": self.variables,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """
        Create template from dictionary.

        Args:
            data: Dictionary containing template and variables.

        Returns:
            PromptTemplate instance.
        """
        return cls(template=data["template"], variables=data["variables"])


class PromptLibrary:
    """
    Manages a collection of prompt templates with file persistence.
    """

    def __init__(self, storage_path: str = "prompts.json"):
        """
        Initialize the prompt library.

        Args:
            storage_path: Path to JSON file for storing templates.
        """
        self.storage_path = Path(storage_path)
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_from_file()

    def _load_from_file(self) -> None:
        """
        Load templates from storage file.
        """
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                for name, template_data in data.items():
                    self._templates[name] = PromptTemplate.from_dict(template_data)
        except (json.JSONDecodeError, IOError) as e:
            raise Exception(f"Failed to load templates from {self.storage_path}: {e}")

    def _save_to_file(self) -> None:
        """
        Save templates to storage file.
        """
        try:
            data = {name: template.to_dict() for name, template in self._templates.items()}
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            raise Exception(f"Failed to save templates to {self.storage_path}: {e}")

    def save(self, name: str, template: PromptTemplate) -> None:
        """
        Save a template to the library.

        Args:
            name: Name to identify the template.
            template: PromptTemplate instance to save.

        Raises:
            Exception: If saving fails.
        """
        self._templates[name] = template
        self._save_to_file()

    def load(self, name: str) -> PromptTemplate:
        """
        Load a template by name.

        Args:
            name: Name of the template to load.

        Returns:
            PromptTemplate instance.

        Raises:
            KeyError: If template not found.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found in library")
        return self._templates[name]

    def list_all(self) -> List[str]:
        """
        List all saved template names.

        Returns:
            List of template names.
        """
        return list(self._templates.keys())

    def delete(self, name: str) -> None:
        """
        Remove a template from the library.

        Args:
            name: Name of the template to delete.

        Raises:
            KeyError: If template not found.
            Exception: If saving fails.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found in library")

        del self._templates[name]
        self._save_to_file()


def validate_prompt(prompt: str) -> Dict[str, Any]:
    """
    Check prompt structure and provide suggestions.

    Args:
        prompt: The prompt string to validate.

    Returns:
        Dictionary with validation results and suggestions.
    """
    issues = []
    suggestions = []

    if len(prompt.strip()) == 0:
        issues.append("Prompt is empty")

    if len(prompt) < 10:
        issues.append("Prompt is very short")
        suggestions.append("Consider providing more context or details")

    if len(prompt) > 10000:
        issues.append("Prompt is very long")
        suggestions.append("Consider breaking into smaller, focused prompts")

    if not any(char in prompt for char in ".!?"):
        suggestions.append("Add punctuation for better clarity")

    if prompt.isupper():
        suggestions.append("Avoid all caps; use normal capitalization")

    if prompt.islower() and len(prompt) > 50:
        suggestions.append("Consider proper capitalization for readability")

    pattern = r"\{([^}]+)\}"
    variables = re.findall(pattern, prompt)
    if variables:
        suggestions.append(f"Template variables detected: {', '.join(set(variables))}")

    word_count = len(prompt.split())
    if word_count < 3:
        issues.append("Prompt has very few words")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "suggestions": suggestions,
        "word_count": word_count,
        "char_count": len(prompt),
        "has_variables": len(variables) > 0,
        "variables": list(set(variables)) if variables else [],
    }
