"""
Unit tests for the chat CLI functionality.
"""

import json
import tempfile
import pytest
import requests
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Import the modules we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dsat.cli.chat import ChatInterface, ChatSession, create_parser
from dsat.agents.agent import AgentConfig


class TestChatSession:
    """Test the ChatSession class."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Create a mock agent
        self.mock_agent = Mock()
        self.mock_agent.config = AgentConfig(
            agent_name="test_agent",
            model_provider="test",
            model_family="test",
            model_version="test-model",
            prompt="test:v1"
        )
        self.session = ChatSession(self.mock_agent)
    
    def test_chat_session_initialization(self):
        """Test ChatSession initialization."""
        assert self.session.agent == self.mock_agent
        assert self.session.history == []
        assert self.session.start_time is not None
    
    def test_add_message(self):
        """Test adding messages to conversation history."""
        self.session.add_message("user", "Hello")
        self.session.add_message("assistant", "Hi there!")
        
        assert len(self.session.history) == 2
        assert self.session.history[0]["role"] == "user"
        assert self.session.history[0]["content"] == "Hello"
        assert self.session.history[1]["role"] == "assistant"
        assert self.session.history[1]["content"] == "Hi there!"
        assert "timestamp" in self.session.history[0]
        assert "timestamp" in self.session.history[1]
    
    def test_export_conversation(self):
        """Test exporting conversation to JSON file."""
        # Add some messages
        self.session.add_message("user", "Hello")
        self.session.add_message("assistant", "Hi there!")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Export conversation
            self.session.export_conversation(temp_path)
            
            # Read and verify the exported data
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            assert "session_start" in exported_data
            assert "agent_config" in exported_data
            assert "conversation" in exported_data
            assert len(exported_data["conversation"]) == 2
            assert exported_data["agent_config"]["agent_name"] == "test_agent"
            
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()


class TestChatInterface:
    """Test the ChatInterface class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.chat = ChatInterface()
    
    def test_chat_interface_initialization(self):
        """Test ChatInterface initialization."""
        assert self.chat.current_session is None
        assert isinstance(self.chat.available_agents, dict)
        assert len(self.chat.available_agents) == 0
        assert self.chat.logger is not None
        assert self.chat.prompts_dir is None
    
    @patch('requests.get')
    def test_check_ollama_health_success(self, mock_get):
        """Test Ollama health check when service is running."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = self.chat._check_ollama_health()
        assert result is True
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5)
    
    @patch('requests.get')
    def test_check_ollama_health_failure(self, mock_get):
        """Test Ollama health check when service is not running."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
        
        result = self.chat._check_ollama_health()
        assert result is False
    
    @patch('requests.get')
    def test_get_ollama_models_success(self, mock_get):
        """Test getting Ollama models successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "qwen2:7b"},
                {"name": "qwen2:13b"}
            ]
        }
        mock_get.return_value = mock_response
        
        models = self.chat._get_ollama_models()
        assert models == ["llama3.2", "qwen2"]  # Should remove duplicates and tags
    
    @patch('requests.get')
    def test_get_ollama_models_failure(self, mock_get):
        """Test getting Ollama models when request fails."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
        
        models = self.chat._get_ollama_models()
        assert models == []
    
    def test_infer_model_family_llama(self):
        """Test inferring model family for Llama models."""
        assert self.chat._infer_model_family("llama3.2") == "llama"
        assert self.chat._infer_model_family("codellama") == "llama"
        assert self.chat._infer_model_family("custom-llama-model") == "llama"
    
    def test_infer_model_family_qwen(self):
        """Test inferring model family for Qwen models."""
        assert self.chat._infer_model_family("qwen2.5") == "qwen"
        assert self.chat._infer_model_family("qwen") == "qwen"
    
    def test_infer_model_family_generic(self):
        """Test inferring model family for unknown models."""
        assert self.chat._infer_model_family("unknown-model") == "llm"
        assert self.chat._infer_model_family("custom") == "llm"
    
    @patch('builtins.input')
    def test_prompt_user_for_ollama_model_single(self, mock_input):
        """Test prompting user for Ollama model when only one available."""
        available_models = ["llama3.2"]
        
        model, family = self.chat._prompt_user_for_ollama_model(available_models)
        assert model == "llama3.2"
        assert family == "llama"
        # Should not prompt if only one model
        mock_input.assert_not_called()
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_prompt_user_for_ollama_model_selection(self, mock_print, mock_input):
        """Test prompting user to select from multiple Ollama models."""
        available_models = ["llama3.2", "qwen2", "mistral"]
        mock_input.return_value = "2"  # Select qwen2
        
        model, family = self.chat._prompt_user_for_ollama_model(available_models)
        assert model == "qwen2"
        assert family == "qwen"
        mock_input.assert_called_once()
    
    @patch('builtins.input')
    def test_prompt_user_for_ollama_model_empty(self, mock_input):
        """Test prompting user for Ollama model when no models available."""
        available_models = []
        
        model, family = self.chat._prompt_user_for_ollama_model(available_models)
        assert model == "llama3.2"  # Fallback
        assert family == "llama"
        mock_input.assert_not_called()
    
    def test_resolve_prompts_directory_cli_priority(self):
        """Test prompts directory resolution with CLI argument (highest priority)."""
        cli_path = Path("/cli/prompts")
        config_path = Path("/config/dir/agents.json")
        agent_config = AgentConfig(
            agent_name="test",
            model_provider="test",
            model_family="test",
            model_version="test",
            prompt="test:v1",
            prompts_dir="/agent/prompts"
        )
        
        result = self.chat._resolve_prompts_directory(cli_path, agent_config, config_path)
        assert result == cli_path
    
    def test_resolve_prompts_directory_agent_config_priority(self):
        """Test prompts directory resolution with agent config field."""
        config_path = Path("/config/dir/agents.json")
        agent_config = AgentConfig(
            agent_name="test",
            model_provider="test",
            model_family="test",
            model_version="test",
            prompt="test:v1",
            prompts_dir="./agent/prompts"
        )
        
        result = self.chat._resolve_prompts_directory(None, agent_config, config_path)
        expected = config_path.parent / "agent/prompts"
        assert result == expected
    
    def test_resolve_prompts_directory_absolute_agent_path(self):
        """Test prompts directory resolution with absolute path in agent config."""
        config_path = Path("/config/dir/agents.json")
        agent_config = AgentConfig(
            agent_name="test",
            model_provider="test",
            model_family="test",
            model_version="test",
            prompt="test:v1",
            prompts_dir="/absolute/agent/prompts"
        )
        
        result = self.chat._resolve_prompts_directory(None, agent_config, config_path)
        expected = Path("/absolute/agent/prompts")
        assert result == expected
    
    def test_resolve_prompts_directory_config_file_fallback(self):
        """Test prompts directory resolution falling back to config file relative."""
        config_path = Path("/config/dir/agents.json")
        
        # Mock the exists method to make it return False for the config relative path
        with patch('pathlib.Path.exists', return_value=False):
            result = self.chat._resolve_prompts_directory(None, None, config_path)
            expected = config_path.parent / "prompts"
            assert result == expected
    
    def test_resolve_prompts_directory_current_dir_fallback(self):
        """Test prompts directory resolution falling back to current directory."""
        result = self.chat._resolve_prompts_directory(None, None, None)
        expected = Path("prompts")
        assert result == expected
    
    def test_create_default_agent_anthropic(self):
        """Test creating default Anthropic agent configuration."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            config = self.chat._create_default_agent("anthropic")
            
            assert config is not None
            assert config.agent_name == "default_claude"
            assert config.model_provider == "anthropic"
            assert config.model_family == "claude"
            assert config.provider_auth["api_key"] == "test-key"
    
    def test_create_default_agent_anthropic_no_key(self):
        """Test creating default Anthropic agent without API key."""
        with patch.dict(os.environ, {}, clear=True):
            config = self.chat._create_default_agent("anthropic")
            assert config is None
    
    @patch('dsat.cli.chat.ChatInterface._check_ollama_health')
    @patch('dsat.cli.chat.ChatInterface._get_ollama_models')  
    @patch('dsat.cli.chat.ChatInterface._prompt_user_for_ollama_model')
    def test_create_default_agent_ollama(self, mock_prompt, mock_get_models, mock_health):
        """Test creating default Ollama agent configuration."""
        # Mock Ollama being available with models
        mock_health.return_value = True
        mock_get_models.return_value = ["llama3.2", "qwen2"]
        mock_prompt.return_value = ("qwen2", "qwen")  # User selects qwen2
        
        config = self.chat._create_default_agent("ollama")
        
        assert config is not None
        assert config.agent_name == "default_ollama"
        assert config.model_provider == "ollama"
        assert config.model_version == "qwen2"  # Should use user selection
        assert config.model_family == "qwen"
        assert config.provider_auth["base_url"] == "http://localhost:11434"
    
    @patch('dsat.cli.chat.ChatInterface._check_ollama_health')
    def test_create_default_agent_ollama_not_running(self, mock_health):
        """Test creating default Ollama agent when Ollama is not running."""
        mock_health.return_value = False
        
        config = self.chat._create_default_agent("ollama")
        assert config is None
    
    @patch('dsat.cli.chat.ChatInterface._check_ollama_health')
    @patch('dsat.cli.chat.ChatInterface._get_ollama_models')
    def test_create_default_agent_ollama_no_models(self, mock_get_models, mock_health):
        """Test creating default Ollama agent when no models are available."""
        mock_health.return_value = True
        mock_get_models.return_value = []
        
        config = self.chat._create_default_agent("ollama")
        assert config is None
    
    def test_create_default_agent_google(self):
        """Test creating default Google agent configuration."""
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}):
            config = self.chat._create_default_agent("google")
            
            assert config is not None
            assert config.agent_name == "default_gemini"
            assert config.model_provider == "google"
            assert config.model_family == "gemini"
            assert config.provider_auth["project_id"] == "test-project"
    
    def test_load_agents_from_config(self):
        """Test loading agents from a configuration file."""
        # Create a temporary config file
        config_data = {
            "test_agent": {
                "model_provider": "test",
                "model_family": "test",
                "model_version": "test-model",
                "prompt": "test:v1"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            agents = self.chat._load_agents_from_config(temp_path)
            
            assert "test_agent" in agents
            assert agents["test_agent"].agent_name == "test_agent"
            assert agents["test_agent"].model_provider == "test"
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_load_agents_from_config_file_not_found(self):
        """Test loading agents from non-existent config file."""
        non_existent_path = Path("/non/existent/config.json")
        agents = self.chat._load_agents_from_config(non_existent_path)
        assert agents == {}
    
    def test_handle_command_help(self):
        """Test handling the /help command."""
        with patch('builtins.print') as mock_print:
            result = self.chat._handle_command("/help")
            assert result is True
            mock_print.assert_called()
    
    def test_handle_command_quit(self):
        """Test handling the /quit command."""
        result = self.chat._handle_command("/quit")
        assert result is False
        
        result = self.chat._handle_command("/exit")
        assert result is False
    
    def test_handle_command_agents(self):
        """Test handling the /agents command."""
        # Add some test agents
        self.chat.available_agents = {
            "agent1": AgentConfig(
                agent_name="agent1",
                model_provider="test",
                model_family="test", 
                model_version="test",
                prompt="test:v1"
            )
        }
        
        with patch('builtins.print') as mock_print:
            result = self.chat._handle_command("/agents")
            assert result is True
            mock_print.assert_called()
    
    def test_handle_command_switch_valid_agent(self):
        """Test switching to a valid agent."""
        # Setup available agents
        test_config = AgentConfig(
            agent_name="test_agent",
            model_provider="test",
            model_family="test",
            model_version="test",
            prompt="test:v1"
        )
        self.chat.available_agents = {"test_agent": test_config}
        self.chat.cli_prompts_dir = None
        self.chat.config_file = None
        
        with patch('dsat.agents.agent.Agent.create') as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            
            with patch('builtins.print') as mock_print:
                result = self.chat._handle_command("/switch test_agent")
                assert result is True
                mock_create.assert_called_once()
                assert self.chat.current_session is not None
                assert self.chat.current_session.agent == mock_agent
    
    def test_handle_command_switch_invalid_agent(self):
        """Test switching to an invalid agent."""
        with patch('builtins.print') as mock_print:
            result = self.chat._handle_command("/switch invalid_agent")
            assert result is True
            mock_print.assert_called()
    
    def test_handle_command_clear_history(self):
        """Test clearing conversation history."""
        # Create a session with history
        mock_agent = Mock()
        self.chat.current_session = ChatSession(mock_agent)
        self.chat.current_session.add_message("user", "test")
        
        assert len(self.chat.current_session.history) == 1
        
        with patch('builtins.print') as mock_print:
            result = self.chat._handle_command("/clear")
            assert result is True
            assert len(self.chat.current_session.history) == 0
            mock_print.assert_called()
    
    def test_handle_command_export_conversation(self):
        """Test exporting conversation."""
        # Create a session with history
        mock_agent = Mock()
        mock_agent.config = AgentConfig(
            agent_name="test",
            model_provider="test",
            model_family="test",
            model_version="test",
            prompt="test:v1"
        )
        self.chat.current_session = ChatSession(mock_agent)
        self.chat.current_session.add_message("user", "test")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            with patch('builtins.print') as mock_print:
                result = self.chat._handle_command(f"/export {temp_path}")
                assert result is True
                mock_print.assert_called()
                
                # Verify file was created
                assert Path(temp_path).exists()
                
        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()
    
    def test_handle_command_unknown(self):
        """Test handling unknown command."""
        with patch('builtins.print') as mock_print:
            result = self.chat._handle_command("/unknown")
            assert result is True
            mock_print.assert_called()
    
    @patch('dsat.agents.agent.Agent.create')
    def test_initialize_agents_inline_creation(self, mock_create):
        """Test initializing agents with inline creation."""
        mock_agent = Mock()
        mock_create.return_value = mock_agent
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            result = self.chat.initialize_agents(
                provider="anthropic", 
                model="claude-3-5-haiku-latest"
            )
            
            assert result is True
            assert self.chat.current_session is not None
            assert self.chat.current_session.agent == mock_agent
            mock_create.assert_called_once()
    
    @patch('dsat.agents.agent.Agent.create')
    def test_initialize_agents_from_config_file(self, mock_create):
        """Test initializing agents from config file."""
        mock_agent = Mock()
        mock_create.return_value = mock_agent
        
        # Create temporary config file
        config_data = {
            "test_agent": {
                "model_provider": "test",
                "model_family": "test",
                "model_version": "test",
                "prompt": "test:v1"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            result = self.chat.initialize_agents(
                config_file=temp_path,
                agent_name="test_agent"
            )
            
            assert result is True
            assert self.chat.current_session is not None
            mock_create.assert_called_once()
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_initialize_agents_no_agents_available(self):
        """Test initializing agents when none are available."""
        # Clear environment variables and mock auto-detect to return empty
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(self.chat, '_auto_detect_providers', return_value={}):
                with patch.object(self.chat, '_discover_agent_configs', return_value={}):
                    with patch('builtins.print') as mock_print:
                        result = self.chat.initialize_agents()
                        assert result is False
                        mock_print.assert_called()


class TestCLIArgumentParser:
    """Test the CLI argument parser."""
    
    def test_create_parser(self):
        """Test creating the argument parser."""
        parser = create_parser()
        
        # Test help generation doesn't crash
        help_text = parser.format_help()
        assert "--config" in help_text
        assert "--agent" in help_text
        assert "--provider" in help_text
        assert "--model" in help_text
        assert "--prompts-dir" in help_text
        assert "--no-colors" in help_text
    
    def test_parser_config_argument(self):
        """Test parsing config argument."""
        parser = create_parser()
        args = parser.parse_args(["--config", "test.json"])
        assert args.config == Path("test.json")
        
        args = parser.parse_args(["-c", "test.json"])
        assert args.config == Path("test.json")
    
    def test_parser_agent_argument(self):
        """Test parsing agent argument."""
        parser = create_parser()
        args = parser.parse_args(["--agent", "test_agent"])
        assert args.agent == "test_agent"
        
        args = parser.parse_args(["-a", "test_agent"])
        assert args.agent == "test_agent"
    
    def test_parser_provider_argument(self):
        """Test parsing provider argument."""
        parser = create_parser()
        args = parser.parse_args(["--provider", "anthropic"])
        assert args.provider == "anthropic"
        
        args = parser.parse_args(["-p", "google"])
        assert args.provider == "google"
    
    def test_parser_model_argument(self):
        """Test parsing model argument."""
        parser = create_parser()
        args = parser.parse_args(["--model", "claude-3-5-haiku"])
        assert args.model == "claude-3-5-haiku"
        
        args = parser.parse_args(["-m", "gpt-4"])
        assert args.model == "gpt-4"
    
    def test_parser_prompts_dir_argument(self):
        """Test parsing prompts directory argument."""
        parser = create_parser()
        args = parser.parse_args(["--prompts-dir", "/path/to/prompts"])
        assert args.prompts_dir == Path("/path/to/prompts")
        
        args = parser.parse_args(["-d", "/another/path"])
        assert args.prompts_dir == Path("/another/path")
    
    def test_parser_no_colors_flag(self):
        """Test parsing no colors flag."""
        parser = create_parser()
        args = parser.parse_args(["--no-colors"])
        assert args.no_colors is True
        
        args = parser.parse_args([])
        assert args.no_colors is False
    
    def test_parser_invalid_provider(self):
        """Test parsing invalid provider raises error."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["--provider", "invalid"])


class TestIntegration:
    """Integration tests for chat CLI components."""
    
    def test_end_to_end_agent_creation_and_session(self):
        """Test complete flow from config to chat session."""
        # Create temporary config and prompts
        config_data = {
            "test_agent": {
                "model_provider": "test",
                "model_family": "test", 
                "model_version": "test-model",
                "prompt": "assistant:v1"
            }
        }
        
        prompt_content = '''v1 = "You are a helpful test assistant."'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create config file
            config_file = temp_path / "agents.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
            
            # Create prompts directory and file
            prompts_dir = temp_path / "prompts"
            prompts_dir.mkdir()
            prompt_file = prompts_dir / "assistant.toml"
            with open(prompt_file, 'w') as f:
                f.write(prompt_content)
            
            # Test the integration
            chat = ChatInterface()
            
            # Mock Agent.create to avoid actual LLM calls
            with patch('dsat.agents.agent.Agent.create') as mock_create:
                mock_agent = Mock()
                test_config_dict = config_data["test_agent"].copy()
                test_config_dict["agent_name"] = "test_agent"  # Add required field
                mock_agent.config = AgentConfig.from_dict(test_config_dict)
                mock_create.return_value = mock_agent
                
                # Initialize with the test config
                result = chat.initialize_agents(
                    config_file=config_file,
                    agent_name="test_agent"
                )
                
                assert result is True
                assert chat.current_session is not None
                assert chat.current_session.agent == mock_agent
                
                # Test prompts directory resolution
                resolved_prompts = chat._resolve_prompts_directory(
                    None, 
                    chat.available_agents["test_agent"], 
                    config_file
                )
                assert resolved_prompts == prompts_dir


if __name__ == "__main__":
    pytest.main([__file__])