"""
Tests for the PromptManager class.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from src.agents.prompts import PromptManager


class TestPromptManager:
    """Test cases for PromptManager class."""

    @pytest.fixture
    def temp_prompts_dir(self):
        """Create a temporary directory for prompt files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def prompt_manager(self, temp_prompts_dir):
        """Create a PromptManager instance with temp directory."""
        return PromptManager(temp_prompts_dir)

    def test_init_creates_directory(self):
        """Test that PromptManager creates the prompts directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompts_dir = Path(tmp_dir) / "prompts"
            assert not prompts_dir.exists()
            
            PromptManager(prompts_dir)
            assert prompts_dir.exists()

    def test_create_prompt(self, prompt_manager, temp_prompts_dir):
        """Test creating a new prompt."""
        prompt_text = "You are a helpful assistant. Please respond clearly and concisely."
        
        result_path = prompt_manager.create_prompt("test_prompt", prompt_text)
        
        # Check file was created
        expected_path = temp_prompts_dir / "test_prompt.toml"
        assert result_path == expected_path
        assert expected_path.exists()
        
        # Check content
        with open(expected_path, 'r') as f:
            content = f.read()
            assert "v1 = '''" in content or 'v1 = """' in content  # Either format is OK
            assert prompt_text in content

    def test_add_version_to_existing_prompt(self, prompt_manager, temp_prompts_dir):
        """Test adding a new version to an existing prompt."""
        # Create initial prompt
        initial_text = "Version 1 prompt"
        prompt_manager.create_prompt("versioned_prompt", initial_text)
        
        # Add second version
        v2_text = "Version 2 prompt with improvements"
        version_key = prompt_manager.add_version("versioned_prompt", v2_text)
        
        assert version_key == "v2"
        
        # Verify both versions exist
        prompt_file = temp_prompts_dir / "versioned_prompt.toml"
        with open(prompt_file, 'r') as f:
            content = f.read()
            assert ("v1 = '''" in content or 'v1 = """' in content)
            assert ("v2 = '''" in content or 'v2 = """' in content)
            assert initial_text in content
            assert v2_text in content

    def test_add_version_nonexistent_prompt(self, prompt_manager):
        """Test adding version to non-existent prompt raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Prompt 'nonexistent' does not exist"):
            prompt_manager.add_version("nonexistent", "some text")

    def test_get_prompt_latest_version(self, prompt_manager, temp_prompts_dir):
        """Test getting the latest version of a prompt."""
        # Create prompt with multiple versions
        prompt_manager.create_prompt("multi_version", "Version 1")
        prompt_manager.add_version("multi_version", "Version 2")
        prompt_manager.add_version("multi_version", "Version 3")
        
        # Get latest version (should be v3)
        latest_prompt = prompt_manager.get_prompt("multi_version")
        assert latest_prompt == "Version 3"

    def test_get_prompt_specific_version(self, prompt_manager, temp_prompts_dir):
        """Test getting a specific version of a prompt."""
        prompt_manager.create_prompt("specific_version", "Version 1")
        prompt_manager.add_version("specific_version", "Version 2")
        
        # Get specific version
        v1_prompt = prompt_manager.get_prompt("specific_version", "v1")
        v2_prompt = prompt_manager.get_prompt("specific_version", "v2")
        
        assert v1_prompt == "Version 1"
        assert v2_prompt == "Version 2"

    def test_get_prompt_nonexistent_prompt(self, prompt_manager):
        """Test getting non-existent prompt returns None."""
        result = prompt_manager.get_prompt("nonexistent")
        assert result is None

    def test_get_prompt_nonexistent_version(self, prompt_manager):
        """Test getting non-existent version returns None."""
        prompt_manager.create_prompt("test", "Version 1")
        
        result = prompt_manager.get_prompt("test", "v99")
        assert result is None

    def test_list_prompts(self, prompt_manager):
        """Test listing all available prompts."""
        # Create several prompts
        prompt_manager.create_prompt("prompt1", "Text 1")
        prompt_manager.create_prompt("prompt2", "Text 2")
        prompt_manager.create_prompt("prompt3", "Text 3")
        
        prompts = prompt_manager.list_prompts()
        assert set(prompts) == {"prompt1", "prompt2", "prompt3"}

    def test_list_prompts_empty_directory(self, prompt_manager):
        """Test listing prompts in empty directory."""
        prompts = prompt_manager.list_prompts()
        assert prompts == []

    def test_list_versions(self, prompt_manager):
        """Test listing all versions of a prompt."""
        # Create prompt with multiple versions
        prompt_manager.create_prompt("versioned", "V1")
        prompt_manager.add_version("versioned", "V2")
        prompt_manager.add_version("versioned", "V3")
        
        versions = prompt_manager.list_versions("versioned")
        assert versions == ["v1", "v2", "v3"]

    def test_list_versions_nonexistent_prompt(self, prompt_manager):
        """Test listing versions of non-existent prompt."""
        versions = prompt_manager.list_versions("nonexistent")
        assert versions == []

    def test_get_all_prompt_data(self, prompt_manager):
        """Test getting all prompt data."""
        # Create test prompts
        prompt_manager.create_prompt("prompt1", "Text 1")
        prompt_manager.create_prompt("prompt2", "Text 2")
        prompt_manager.add_version("prompt2", "Text 2 v2")
        
        all_data = prompt_manager.get_all_prompt_data()
        
        assert "prompt1" in all_data
        assert "prompt2" in all_data
        assert all_data["prompt1"]["v1"] == "Text 1"
        assert all_data["prompt2"]["v1"] == "Text 2"
        assert all_data["prompt2"]["v2"] == "Text 2 v2"

    def test_copy_prompts_to_directory(self, prompt_manager, temp_prompts_dir):
        """Test copying prompts to another directory."""
        # Create test prompts
        prompt_manager.create_prompt("prompt1", "Content 1")
        prompt_manager.create_prompt("prompt2", "Content 2")
        
        # Create target directory
        with tempfile.TemporaryDirectory() as target_tmp:
            target_dir = Path(target_tmp) / "copied_prompts"
            
            prompt_manager.copy_prompts_to_directory(target_dir)
            
            # Verify files were copied
            assert target_dir.exists()
            assert (target_dir / "prompt1.toml").exists()
            assert (target_dir / "prompt2.toml").exists()
            
            # Verify content is preserved
            with open(target_dir / "prompt1.toml", 'r') as f:
                content = f.read()
                assert "Content 1" in content

    def test_prompt_with_special_characters(self, prompt_manager):
        """Test handling prompts with special characters and formatting."""
        complex_prompt = '''You are an AI assistant. 

Follow these rules:
1. Be helpful and harmless
2. Use {placeholder} for variables
3. Handle "quotes" and 'apostrophes'
4. Support multi-line responses

Example output:
```python
def hello():
    print("Hello, world!")
```
'''
        
        prompt_manager.create_prompt("complex", complex_prompt)
        retrieved = prompt_manager.get_prompt("complex")
        
        assert retrieved == complex_prompt
        assert "{placeholder}" in retrieved
        assert 'print("Hello, world!")' in retrieved

    @patch('builtins.open', side_effect=OSError("Permission denied"))
    def test_create_prompt_permission_error(self, mock_open_func, prompt_manager):
        """Test handling permission errors when creating prompts."""
        with pytest.raises(OSError, match="Permission denied"):
            prompt_manager.create_prompt("test", "content")

    def test_version_number_parsing(self, prompt_manager, temp_prompts_dir):
        """Test that version numbers are parsed correctly even with gaps."""
        # Create initial prompt
        prompt_manager.create_prompt("gaps", "V1")
        
        # Manually create a file with non-sequential versions
        prompt_file = temp_prompts_dir / "gaps.toml"
        with open(prompt_file, 'w') as f:
            f.write('v1 = """Version 1"""\n')
            f.write('v3 = """Version 3"""\n')  # Skip v2
            f.write('v7 = """Version 7"""\n')  # Big gap
        
        # Add new version - should be v8
        version_key = prompt_manager.add_version("gaps", "Version 8")
        assert version_key == "v8"
        
        # Latest should be v8
        latest = prompt_manager.get_prompt("gaps")
        assert latest == "Version 8"

    def test_empty_prompt_handling(self, prompt_manager):
        """Test handling of empty prompt content."""
        prompt_manager.create_prompt("empty", "")
        
        result = prompt_manager.get_prompt("empty")
        assert result == ""

    def test_prompt_with_toml_special_syntax(self, prompt_manager):
        """Test prompts containing TOML-sensitive content."""
        # Content that could break TOML parsing if not handled properly
        tricky_content = '''Here's a prompt with:
- Triple quotes: """
- Array syntax: [1, 2, 3]
- Key-value pairs: key = "value"
- Multiple lines with = signs'''
        
        prompt_manager.create_prompt("tricky", tricky_content)
        result = prompt_manager.get_prompt("tricky")
        
        assert result == tricky_content
        assert '"""' in result
        assert '[1, 2, 3]' in result