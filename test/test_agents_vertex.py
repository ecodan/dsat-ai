"""
Tests for the GoogleVertexAIAgent class.
"""

import logging
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.agents.agent import AgentConfig
from src.agents.vertex_agent import GoogleVertexAIAgent, VERTEX_AI_AVAILABLE


class TestGoogleVertexAIAgent:
    """Test cases for GoogleVertexAIAgent class."""

    @pytest.fixture
    def vertex_config(self):
        """Return a valid Vertex AI agent config."""
        return AgentConfig(
            agent_name="vertex_test",
            model_provider="google",
            model_family="gemini",
            model_version="gemini-2.0-flash",
            prompt="assistant:v1",
            model_parameters={"temperature": 0.5, "max_output_tokens": 8192},
            provider_auth={"project_id": "test-project-123", "location": "us-central1"}
        )

    @pytest.fixture
    def logger(self):
        """Return a mock logger."""
        return Mock(spec=logging.Logger)

    @pytest.fixture
    def temp_prompts_dir(self):
        """Create temporary directory for prompts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def mock_vertex_model(self):
        """Return a mock Vertex AI GenerativeModel."""
        return Mock()

    def test_vertex_available_flag(self):
        """Test that VERTEX_AI_AVAILABLE flag is set correctly."""
        assert isinstance(VERTEX_AI_AVAILABLE, bool)

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_initialization_with_config(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test GoogleVertexAIAgent initialization with AgentConfig."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)
        
        assert agent.config == vertex_config
        assert agent.logger == logger
        assert agent.client == mock_model
        
        mock_vertexai.init.assert_called_once_with(project="test-project-123", location="us-central1")
        mock_model_class.assert_called_once_with("gemini-2.0-flash")

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_initialization_backward_compatibility(self, mock_model_class, mock_vertexai, logger, temp_prompts_dir):
        """Test GoogleVertexAIAgent initialization with backward compatibility."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(
            project_id="legacy-project",
            location="us-west1",
            model="gemini-pro",
            logger=logger,
            prompts_dir=temp_prompts_dir
        )
        
        assert agent.config.agent_name == "vertex"
        assert agent.config.model_provider == "google"
        assert agent.config.model_version == "gemini-pro"
        assert agent.config.provider_auth["project_id"] == "legacy-project"
        assert agent.config.provider_auth["location"] == "us-west1"
        
        mock_vertexai.init.assert_called_once_with(project="legacy-project", location="us-west1")
        mock_model_class.assert_called_once_with("gemini-pro")

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_initialization_config_with_overrides(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test initialization with config and parameter overrides."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(
            config=vertex_config,
            project_id="override-project",
            location="europe-west1",
            logger=logger,
            prompts_dir=temp_prompts_dir
        )
        
        # Should use override values
        assert agent.config.provider_auth["project_id"] == "override-project"
        assert agent.config.provider_auth["location"] == "europe-west1"
        
        mock_vertexai.init.assert_called_once_with(project="override-project", location="europe-west1")

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_initialization_default_location(self, mock_model_class, mock_vertexai, logger, temp_prompts_dir):
        """Test initialization uses default location when not specified."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(
            project_id="test-project",
            model="gemini-pro",
            logger=logger,
            prompts_dir=temp_prompts_dir
        )
        
        # Should use default location
        assert agent.config.provider_auth["location"] == "us-central1"
        mock_vertexai.init.assert_called_once_with(project="test-project", location="us-central1")

    def test_initialization_missing_project_id(self, logger, temp_prompts_dir):
        """Test initialization fails without project_id."""
        with pytest.raises(ValueError, match="Either config must be provided, or both project_id and model must be provided"):
            GoogleVertexAIAgent(model="gemini-pro", logger=logger, prompts_dir=temp_prompts_dir)

    def test_initialization_missing_model(self, logger, temp_prompts_dir):
        """Test initialization fails without model."""
        with pytest.raises(ValueError, match="Either config must be provided, or both project_id and model must be provided"):
            GoogleVertexAIAgent(project_id="test-project", logger=logger, prompts_dir=temp_prompts_dir)

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    def test_initialization_config_missing_project_id(self, logger, temp_prompts_dir):
        """Test initialization fails with config but no project_id."""
        config = AgentConfig(
            agent_name="vertex_no_project",
            model_provider="google",
            model_family="gemini",
            model_version="gemini-pro",
            prompt="test:v1",
            provider_auth={}  # No project_id
        )
        
        with pytest.raises(ValueError, match="project_id is required in provider_auth for GoogleVertexAIAgent"):
            GoogleVertexAIAgent(config=config, logger=logger, prompts_dir=temp_prompts_dir)

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', False)
    def test_initialization_vertex_not_available(self, vertex_config, logger, temp_prompts_dir):
        """Test initialization fails when Vertex AI package not available."""
        with pytest.raises(ImportError, match="google-cloud-aiplatform package is required for Google Vertex AI support"):
            GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)

    def test_initialization_default_logger(self, vertex_config, temp_prompts_dir):
        """Test initialization creates default logger when none provided."""
        with patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True):
            with patch('src.agents.vertex_agent.vertexai'):
                with patch('src.agents.vertex_agent.GenerativeModel'):
                    agent = GoogleVertexAIAgent(config=vertex_config, prompts_dir=temp_prompts_dir)
                    
                    assert isinstance(agent.logger, logging.Logger)
                    assert agent.logger.name == 'src.agents.vertex_agent'

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_invoke_with_explicit_system_prompt(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test invoke method with explicit system prompt."""
        # Setup mock response
        mock_response = Mock()
        mock_response.text = "Vertex AI response"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)
        
        result = agent.invoke("Hello", "You are a helpful assistant")
        
        assert result == "Vertex AI response"
        
        # Should combine system and user prompts
        expected_prompt = "You are a helpful assistant\n\nHello"
        mock_model.generate_content.assert_called_once_with(
            expected_prompt,
            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 8192
            }
        )

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_invoke_with_auto_system_prompt(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test invoke method with automatic system prompt loading."""
        # Setup prompt file
        prompt_file = temp_prompts_dir / "assistant.toml"
        with open(prompt_file, 'w') as f:
            f.write('v1 = """You are a Gemini AI assistant."""')
        
        # Setup mock response
        mock_response = Mock()
        mock_response.text = "Gemini response"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)
        
        result = agent.invoke("What can you help with?")
        
        assert result == "Gemini response"
        
        # Should combine auto-loaded system prompt with user prompt
        expected_prompt = "You are a Gemini AI assistant.\n\nWhat can you help with?"
        mock_model.generate_content.assert_called_once_with(
            expected_prompt,
            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 8192
            }
        )

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_invoke_without_system_prompt(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test invoke method without system prompt."""
        # Setup mock response
        mock_response = Mock()
        mock_response.text = "Direct response"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)
        
        result = agent.invoke("Just the user prompt")
        
        assert result == "Direct response"
        
        # Should use just the user prompt
        mock_model.generate_content.assert_called_once_with(
            "Just the user prompt",
            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 8192
            }
        )

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_invoke_with_default_parameters(self, mock_model_class, mock_vertexai, logger, temp_prompts_dir):
        """Test invoke uses default parameters when not specified in config."""
        config = AgentConfig(
            agent_name="vertex_defaults",
            model_provider="google",
            model_family="gemini",
            model_version="gemini-pro",
            prompt="test:v1",
            provider_auth={"project_id": "test-project"}
            # No model_parameters specified
        )
        
        mock_response = Mock()
        mock_response.text = "Default response"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=config, logger=logger, prompts_dir=temp_prompts_dir)
        
        agent.invoke("Test")
        
        # Should use default parameters
        call_args = mock_model.generate_content.call_args[1]
        assert call_args['generation_config']['temperature'] == 0.3  # Default
        assert call_args['generation_config']['max_output_tokens'] == 20000  # Default

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_invoke_api_error(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test invoke handles API errors."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Vertex AI API error")
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)
        
        with pytest.raises(Exception, match="Vertex AI API error"):
            agent.invoke("Test")
        
        logger.error.assert_called_once()
        assert "Vertex AI API error" in logger.error.call_args[0][0]

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_invoke_logging(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test that invoke method logs responses correctly."""
        mock_response = Mock()
        mock_response.text = "Test response from Vertex AI"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)
        
        result = agent.invoke("Hello")
        
        # Check debug logging calls
        assert logger.debug.call_count == 2
        debug_calls = [call.args[0] for call in logger.debug.call_args_list]
        
        # Should log raw response
        assert any("Vertex AI raw response:" in call for call in debug_calls)
        
        # Should log response stats
        assert any("response: " in call and "bytes" in call and "words" in call for call in debug_calls)

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_model_property(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test the model property returns correct model version."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)
        
        assert agent.model == "gemini-2.0-flash"

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_invoke_with_custom_model_parameters(self, mock_model_class, mock_vertexai, logger, temp_prompts_dir):
        """Test invoke with custom model parameters."""
        config = AgentConfig(
            agent_name="vertex_custom",
            model_provider="google",
            model_family="gemini",
            model_version="gemini-ultra",
            prompt="test:v1",
            model_parameters={
                "temperature": 0.8,
                "max_output_tokens": 16384
            },
            provider_auth={"project_id": "test-project"}
        )
        
        mock_response = Mock()
        mock_response.text = "Custom response"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=config, logger=logger, prompts_dir=temp_prompts_dir)
        
        agent.invoke("Test with custom params")
        
        # Verify custom parameters were used
        call_args = mock_model.generate_content.call_args[1]
        assert call_args['generation_config']['temperature'] == 0.8
        assert call_args['generation_config']['max_output_tokens'] == 16384

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_config_with_location_from_auth(self, mock_model_class, mock_vertexai, logger, temp_prompts_dir):
        """Test that location is read from provider_auth correctly."""
        config = AgentConfig(
            agent_name="vertex_location",
            model_provider="google",
            model_family="gemini",
            model_version="gemini-pro",
            prompt="test:v1",
            provider_auth={"project_id": "test-project", "location": "asia-southeast1"}
        )
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=config, logger=logger, prompts_dir=temp_prompts_dir)
        
        # Should use location from config
        mock_vertexai.init.assert_called_once_with(project="test-project", location="asia-southeast1")

    def test_backward_compatibility_missing_model(self, logger, temp_prompts_dir):
        """Test backward compatibility mode requires both project_id and model."""
        with pytest.raises(ValueError, match="Either config must be provided, or both project_id and model must be provided"):
            GoogleVertexAIAgent(project_id="test-project", logger=logger, prompts_dir=temp_prompts_dir)  # Missing model

    def test_backward_compatibility_missing_project_id(self, logger, temp_prompts_dir):
        """Test backward compatibility mode requires both project_id and model."""
        with pytest.raises(ValueError, match="Either config must be provided, or both project_id and model must be provided"):
            GoogleVertexAIAgent(model="gemini-pro", logger=logger, prompts_dir=temp_prompts_dir)  # Missing project_id

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_prompt_with_formatting(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test that prompts with special formatting are handled correctly."""
        # Setup prompt file with placeholder formatting
        prompt_file = temp_prompts_dir / "assistant.toml"
        with open(prompt_file, 'w') as f:
            f.write('v1 = """You are {role}. Follow these rules: {rules}"""')
        
        mock_response = Mock()
        mock_response.text = "Formatted response"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)
        
        result = agent.invoke("Test prompt")
        
        # Should preserve the formatting in the prompt
        expected_prompt = "You are {role}. Follow these rules: {rules}\n\nTest prompt"
        mock_model.generate_content.assert_called_once_with(
            expected_prompt,
            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 8192
            }
        )

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_invoke_with_conversation_history(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test invoke method with conversation history."""
        from src.cli.memory import ConversationMessage
        
        mock_response = Mock()
        mock_response.text = "I can see you asked about Machine Learning earlier."
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)
        
        # Create conversation history
        history = [
            ConversationMessage("user", "What is Machine Learning?", "2023-01-01T00:00:00", 12),
            ConversationMessage("assistant", "ML is a subset of AI that uses algorithms.", "2023-01-01T00:01:00", 18)
        ]
        
        result = agent.invoke("What did I ask about?", history=history)
        
        assert result == "I can see you asked about Machine Learning earlier."
        
        # Verify context was built with history
        call_args = mock_model.generate_content.call_args[0]
        full_context = call_args[0]
        
        # Context should include system prompt, history, and current question
        expected_context = (
            "You are {role}. Follow these rules: {rules}\n\n"
            "Human: What is Machine Learning?\n\n"
            "Assistant: ML is a subset of AI that uses algorithms.\n\n"
            "Human: What did I ask about?"
        )
        assert full_context == expected_context

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_invoke_without_history_backward_compatibility(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test invoke method still works without history parameter (backward compatibility)."""
        mock_response = Mock()
        mock_response.text = "Hello! How can I assist you today?"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)
        
        # Call without history parameter
        result = agent.invoke("Hello")
        
        assert result == "Hello! How can I assist you today?"
        
        # Verify only current message with system prompt was sent
        call_args = mock_model.generate_content.call_args[0]
        full_context = call_args[0]
        
        expected_context = "You are {role}. Follow these rules: {rules}\n\nHuman: Hello"
        assert full_context == expected_context

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_invoke_with_empty_history(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test invoke method with empty history list."""
        mock_response = Mock()
        mock_response.text = "Hello there!"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        agent = GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)
        
        # Call with empty history
        result = agent.invoke("Hello", history=[])
        
        assert result == "Hello there!"
        
        # Verify only current message with system prompt was sent
        call_args = mock_model.generate_content.call_args[0]
        full_context = call_args[0]
        
        expected_context = "You are {role}. Follow these rules: {rules}\n\nHuman: Hello"
        assert full_context == expected_context

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    @patch('src.agents.vertex_agent.vertexai')
    @patch('src.agents.vertex_agent.GenerativeModel')
    def test_context_building_method(self, mock_model_class, mock_vertexai, vertex_config, logger, temp_prompts_dir):
        """Test the _build_conversation_context method directly."""
        from src.cli.memory import ConversationMessage
        
        agent = GoogleVertexAIAgent(config=vertex_config, logger=logger, prompts_dir=temp_prompts_dir)
        
        # Test with history
        history = [
            ConversationMessage("user", "Hello", "2023-01-01T00:00:00", 3),
            ConversationMessage("assistant", "Hi there!", "2023-01-01T00:01:00", 5)
        ]
        
        context = agent._build_conversation_context("You are helpful.", history, "How are you?")
        
        expected = (
            "You are helpful.\n\n"
            "Human: Hello\n\n"
            "Assistant: Hi there!\n\n"
            "Human: How are you?"
        )
        
        assert context == expected
        
        # Test without history
        context_no_history = agent._build_conversation_context("You are helpful.", None, "Hello")
        expected_no_history = "You are helpful.\n\nHuman: Hello"
        
        assert context_no_history == expected_no_history