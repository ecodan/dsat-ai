"""
Prompt management system with versioned TOML storage.
"""

import tomlkit
from pathlib import Path
from typing import Dict, List, Optional


class PromptManager:
    """Manages versioned prompts stored as TOML files."""

    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def create_prompt(self, name: str, prompt_text: str) -> Path:
        """Create a new prompt with v1."""
        prompt_file = self.prompts_dir / f"{name}.toml"

        # Create a tomlkit document for proper formatting
        doc = tomlkit.document()
        doc["v1"] = tomlkit.string(prompt_text, literal=True, multiline=True)

        with open(prompt_file, "w") as f:
            f.write(tomlkit.dumps(doc))

        return prompt_file

    def add_version(self, name: str, prompt_text: str) -> str:
        """Add a new version to an existing prompt."""
        prompt_file = self.prompts_dir / f"{name}.toml"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt '{name}' does not exist")

        # Load existing prompt data
        with open(prompt_file, "r") as f:
            doc = tomlkit.load(f)

        # Find next version number
        versions = [k for k in doc.keys() if k.startswith("v")]
        version_numbers = [int(v[1:]) for v in versions if v[1:].isdigit()]
        next_version = max(version_numbers) + 1 if version_numbers else 1

        version_key = f"v{next_version}"
        doc[version_key] = tomlkit.string(prompt_text, literal=True, multiline=True)

        # Save updated data
        with open(prompt_file, "w") as f:
            f.write(tomlkit.dumps(doc))

        return version_key

    def get_prompt(self, name: str, version: Optional[str] = None) -> Optional[str]:
        """Get a prompt by name and optional version."""
        prompt_file = self.prompts_dir / f"{name}.toml"

        if not prompt_file.exists():
            return None

        with open(prompt_file, "r") as f:
            prompt_data = tomlkit.load(f)

        if version:
            return prompt_data.get(version)
        else:
            # Return latest version
            versions = [k for k in prompt_data.keys() if k.startswith("v")]
            if not versions:
                return None

            version_numbers = [int(v[1:]) for v in versions if v[1:].isdigit()]
            if not version_numbers:
                return None

            latest_version = f"v{max(version_numbers)}"
            return prompt_data.get(latest_version)

    def list_prompts(self) -> List[str]:
        """List all available prompt names."""
        return [f.stem for f in self.prompts_dir.glob("*.toml")]

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a prompt."""
        prompt_file = self.prompts_dir / f"{name}.toml"

        if not prompt_file.exists():
            return []

        with open(prompt_file, "r") as f:
            prompt_data = tomlkit.load(f)

        versions = [k for k in prompt_data.keys() if k.startswith("v")]
        version_numbers = [int(v[1:]) for v in versions if v[1:].isdigit()]
        return [f"v{num}" for num in sorted(version_numbers)]

    def get_all_prompt_data(self) -> Dict[str, Dict[str, str]]:
        """Get all prompt data for copying to milestone runs."""
        all_prompts = {}

        for prompt_file in self.prompts_dir.glob("*.toml"):
            prompt_name = prompt_file.stem
            with open(prompt_file, "r") as f:
                prompt_data = tomlkit.load(f)
            all_prompts[prompt_name] = dict(prompt_data)

        return all_prompts

    def copy_prompts_to_directory(self, target_dir: Path) -> None:
        """Copy all prompt files to target directory."""
        target_dir.mkdir(parents=True, exist_ok=True)

        for prompt_file in self.prompts_dir.glob("*.toml"):
            target_file = target_dir / prompt_file.name
            with open(prompt_file, "r") as src:
                doc = tomlkit.load(src)

            with open(target_file, "w") as dst:
                dst.write(tomlkit.dumps(doc))
