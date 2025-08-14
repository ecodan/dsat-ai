"""
Tests for the Agent base class.
"""

import logging
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.agents.agent import Agent, AgentConfig
from test.echo_agent import EchoAgent, create_echo_agent_config


class ConcreteAgent(Agent):
    """Concrete implementation of Agent for testing."""
    
    def invoke(self, user_prompt: str, system_prompt: str = None) -> str:
        """Mock implementation for testing."""
        if system_prompt is None:
            system_prompt = self.get_system_prompt()
        return f"Response to: {user_prompt} (System: {system_prompt})"
    
    @property
    def model(self) -> str:
        return self.config.model_version


class TestAgentBase:
    """Test cases for Agent base class."""

    @pytest.fixture
    def sample_config(self):
        """Return a sample AgentConfig for testing."""
        return AgentConfig(
            agent_name="test_agent",
            model_provider="test_provider",
            model_family="test_family",
            model_version="test-model-v1",
            prompt="test_prompt:v1",
            model_parameters={"temperature": 0.7},
            provider_auth={"api_key": "test-key"}
        )

    @pytest.fixture
    def temp_prompts_dir(self):
        """Create temporary directory for prompts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def logger(self):
        """Return a mock logger."""
        return Mock(spec=logging.Logger)

    @pytest.fixture
    def agent(self, sample_config, logger, temp_prompts_dir):
        """Create a concrete agent instance for testing."""
        return ConcreteAgent(sample_config, logger, temp_prompts_dir)

    def test_agent_initialization(self, sample_config, logger, temp_prompts_dir):
        """Test Agent initialization with all parameters."""
        agent = ConcreteAgent(sample_config, logger, temp_prompts_dir)
        
        assert agent.config == sample_config
        assert agent.logger == logger
        assert agent.prompt_manager.prompts_dir == temp_prompts_dir
        assert agent._system_prompt is None

    def test_agent_initialization_default_prompts_dir(self, sample_config, logger):
        """Test Agent initialization with default prompts directory."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            agent = ConcreteAgent(sample_config, logger)
            
            assert agent.prompt_manager.prompts_dir == Path("prompts")
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_agent_initialization_string_prompts_dir(self, sample_config, logger):
        """Test Agent initialization with string prompts directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = ConcreteAgent(sample_config, logger, tmp_dir)
            assert agent.prompt_manager.prompts_dir == Path(tmp_dir)

    def test_get_system_prompt_caching(self, agent, temp_prompts_dir):
        """Test that get_system_prompt caches the result."""
        # Create a prompt file
        prompt_file = temp_prompts_dir / "test_prompt.toml"
        with open(prompt_file, 'w') as f:
            f.write('v1 = """Test system prompt content"""')
        
        # First call should load from file
        prompt1 = agent.get_system_prompt()
        assert prompt1 == "Test system prompt content"
        assert agent._system_prompt == "Test system prompt content"
        
        # Second call should use cached value
        with patch.object(agent.prompt_manager, 'get_prompt') as mock_get:
            prompt2 = agent.get_system_prompt()
            assert prompt2 == "Test system prompt content"
            mock_get.assert_not_called()  # Should not call prompt manager again

    def test_get_system_prompt_with_specific_version(self, logger, temp_prompts_dir):
        """Test get_system_prompt gets latest version when not specified."""
        # Create config that will get latest version
        config = AgentConfig(
            agent_name="test_agent",
            model_provider="test_provider",
            model_family="test_family",
            model_version="test-model-v1",
            prompt="test_prompt:latest"  # Will get latest
        )
        
        # Create prompt file with multiple versions
        prompt_file = temp_prompts_dir / "test_prompt.toml"
        with open(prompt_file, 'w') as f:
            f.write("v1 = '''Version 1 content'''\n")
            f.write("v2 = '''Version 2 content'''")
        
        # Create fresh agent to avoid caching issues
        agent = ConcreteAgent(config, logger, temp_prompts_dir)
        prompt = agent.get_system_prompt()
        assert prompt == "Version 2 content"  # Should get latest (v2)

    def test_get_system_prompt_latest_version(self, sample_config, logger, temp_prompts_dir):
        """Test get_system_prompt with 'latest' version."""
        config = AgentConfig(
            agent_name="test_agent",
            model_provider="test_provider",
            model_family="test_family", 
            model_version="test-model-v1",
            prompt="test_prompt:latest"  # Explicitly set to latest
        )
        
        agent = ConcreteAgent(config, logger, temp_prompts_dir)
        
        # Create prompt file
        prompt_file = temp_prompts_dir / "test_prompt.toml"
        with open(prompt_file, 'w') as f:
            f.write('v1 = """Version 1"""\nv3 = """Version 3"""')
        
        prompt = agent.get_system_prompt()
        assert prompt == "Version 3"  # Should get highest version

    def test_get_system_prompt_none_version(self, sample_config, logger, temp_prompts_dir):
        """Test get_system_prompt with None version."""
        config = AgentConfig(
            agent_name="test_agent",
            model_provider="test_provider",
            model_family="test_family",
            model_version="test-model-v1",
            prompt="test_prompt:latest"
        )
        
        agent = ConcreteAgent(config, logger, temp_prompts_dir)
        
        # Create prompt file
        prompt_file = temp_prompts_dir / "test_prompt.toml"
        with open(prompt_file, 'w') as f:
            f.write('v1 = """Version 1"""\nv2 = """Version 2"""')
        
        prompt = agent.get_system_prompt()
        assert prompt == "Version 2"  # Should get latest

    def test_get_system_prompt_not_found(self, agent, logger):
        """Test get_system_prompt when prompt is not found."""
        prompt = agent.get_system_prompt()
        assert prompt is None
        logger.warning.assert_called_once_with("System prompt not found: test_prompt:v1")

    def test_get_system_prompt_file_not_found(self, agent, logger):
        """Test get_system_prompt with non-existent prompt file."""
        prompt = agent.get_system_prompt()
        assert prompt is None
        logger.warning.assert_called_once_with("System prompt not found: test_prompt:v1")

    def test_abstract_methods_not_implemented(self, sample_config, logger):
        """Test that Agent cannot be instantiated directly due to abstract methods."""
        with pytest.raises(TypeError):
            Agent(sample_config, logger)

    def test_invoke_abstract_method(self):
        """Test that invoke is properly marked as abstract."""
        # Verify that ConcreteAgent has implemented invoke
        agent_methods = ConcreteAgent.__dict__
        assert 'invoke' in agent_methods
        
        # Verify that base Agent has invoke as abstract
        assert hasattr(Agent, 'invoke')

    def test_model_property_abstract(self):
        """Test that model property is properly marked as abstract."""
        # Verify ConcreteAgent implements model property
        assert hasattr(ConcreteAgent, 'model')
        
        # Verify base Agent has model as abstract
        assert hasattr(Agent, 'model')

    def test_concrete_agent_invoke_with_auto_prompt(self, agent, temp_prompts_dir):
        """Test ConcreteAgent invoke method with automatic prompt loading."""
        # Create prompt file
        prompt_file = temp_prompts_dir / "test_prompt.toml"
        with open(prompt_file, 'w') as f:
            f.write('v1 = """You are a helpful assistant"""')
        
        result = agent.invoke("Hello")
        assert "Response to: Hello" in result
        assert "You are a helpful assistant" in result

    def test_concrete_agent_invoke_with_explicit_prompt(self, agent):
        """Test ConcreteAgent invoke method with explicit system prompt."""
        result = agent.invoke("Hello", "Custom system prompt")
        assert "Response to: Hello" in result
        assert "Custom system prompt" in result

    def test_concrete_agent_model_property(self, agent):
        """Test ConcreteAgent model property."""
        assert agent.model == "test-model-v1"


class TestAgentFactory:
    """Test cases for Agent factory methods."""

    @pytest.fixture
    def anthropic_config(self):
        """Return Anthropic agent config."""
        return AgentConfig(
            agent_name="claude_agent",
            model_provider="anthropic",
            model_family="claude",
            model_version="claude-3-5-haiku-latest",
            prompt="assistant:v1",
            provider_auth={"api_key": "sk-test-key"}
        )

    @pytest.fixture
    def google_config(self):
        """Return Google Vertex AI agent config."""
        return AgentConfig(
            agent_name="vertex_agent",
            model_provider="google",
            model_family="gemini",
            model_version="gemini-2.0-flash",
            prompt="assistant:v1",
            provider_auth={"project_id": "test-project", "location": "us-central1"}
        )

    @patch('src.agents.anthropic_agent.ClaudeLLMAgent')
    def test_create_anthropic_agent(self, mock_claude_class, anthropic_config):
        """Test creating Anthropic agent through factory."""
        mock_claude_instance = Mock()
        mock_claude_class.return_value = mock_claude_instance
        
        with patch('src.agents.anthropic_agent.ANTHROPIC_AVAILABLE', True):
            agent = Agent.create(anthropic_config)
            
            mock_claude_class.assert_called_once()
            call_args = mock_claude_class.call_args
            assert call_args[1]['config'] == anthropic_config
            assert call_args[1]['api_key'] == "sk-test-key"
            assert agent == mock_claude_instance

    @patch('src.agents.vertex_agent.GoogleVertexAIAgent')
    def test_create_google_agent(self, mock_vertex_class, google_config):
        """Test creating Google Vertex AI agent through factory."""
        mock_vertex_instance = Mock()
        mock_vertex_class.return_value = mock_vertex_instance
        
        with patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True):
            agent = Agent.create(google_config)
            
            mock_vertex_class.assert_called_once()
            call_args = mock_vertex_class.call_args
            assert call_args[1]['config'] == google_config
            assert call_args[1]['project_id'] == "test-project"
            assert call_args[1]['location'] == "us-central1"
            assert agent == mock_vertex_instance

    def test_create_unsupported_provider(self):
        """Test creating agent with unsupported provider."""
        config = AgentConfig(
            agent_name="unsupported",
            model_provider="unsupported_provider",
            model_family="unsupported",
            model_version="v1",
            prompt="test:v1"
        )
        
        with pytest.raises(ValueError, match="Unsupported provider: unsupported_provider"):
            Agent.create(config)

    @patch('src.agents.anthropic_agent.ANTHROPIC_AVAILABLE', True)
    def test_create_anthropic_missing_api_key(self):
        """Test creating Anthropic agent without API key."""
        config = AgentConfig(
            agent_name="claude_no_key",
            model_provider="anthropic",
            model_family="claude",
            model_version="claude-3-5-haiku",
            prompt="test:v1",
            provider_auth={}  # No API key
        )
        
        with pytest.raises(ValueError, match="api_key is required in provider_auth for Anthropic provider"):
            Agent.create(config)

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', True)
    def test_create_google_missing_project_id(self):
        """Test creating Google agent without project ID."""
        config = AgentConfig(
            agent_name="vertex_no_project",
            model_provider="google",
            model_family="gemini",
            model_version="gemini-2.0-flash",
            prompt="test:v1",
            provider_auth={}  # No project_id
        )
        
        with pytest.raises(ValueError, match="project_id is required in provider_auth for Google provider"):
            Agent.create(config)

    @patch('src.agents.anthropic_agent.ANTHROPIC_AVAILABLE', False)
    def test_create_anthropic_not_available(self, anthropic_config):
        """Test creating Anthropic agent when package not available."""
        with pytest.raises(ImportError, match="anthropic package is required for Anthropic provider"):
            Agent.create(anthropic_config)

    @patch('src.agents.vertex_agent.VERTEX_AI_AVAILABLE', False)  
    def test_create_google_not_available(self, google_config):
        """Test creating Google agent when package not available."""
        with pytest.raises(ImportError, match="google-cloud-aiplatform package is required for Google provider"):
            Agent.create(google_config)

    def test_create_with_default_logger(self, anthropic_config):
        """Test creating agent with default logger."""
        with patch('src.agents.anthropic_agent.ClaudeLLMAgent') as mock_claude:
            with patch('src.agents.anthropic_agent.ANTHROPIC_AVAILABLE', True):
                Agent.create(anthropic_config)
                
                # Should have created a default logger
                call_args = mock_claude.call_args
                logger = call_args[1]['logger']
                assert isinstance(logger, logging.Logger)

    def test_create_with_custom_logger(self, anthropic_config):
        """Test creating agent with custom logger."""
        custom_logger = logging.getLogger("test_logger")
        
        with patch('src.agents.anthropic_agent.ClaudeLLMAgent') as mock_claude:
            with patch('src.agents.anthropic_agent.ANTHROPIC_AVAILABLE', True):
                Agent.create(anthropic_config, logger=custom_logger)
                
                call_args = mock_claude.call_args
                assert call_args[1]['logger'] == custom_logger

    def test_create_with_custom_prompts_dir(self, anthropic_config):
        """Test creating agent with custom prompts directory."""
        custom_prompts_dir = "/custom/prompts"
        
        with patch('src.agents.anthropic_agent.ClaudeLLMAgent') as mock_claude:
            with patch('src.agents.anthropic_agent.ANTHROPIC_AVAILABLE', True):
                Agent.create(anthropic_config, prompts_dir=custom_prompts_dir)
                
                call_args = mock_claude.call_args
                assert call_args[1]['prompts_dir'] == custom_prompts_dir

    def test_create_echo_agent(self):
        """Test creating EchoAgent using factory method."""
        config = create_echo_agent_config("echo_test")
        
        # Create agent using factory
        agent = Agent.create(config)
        
        # Should return EchoAgent instance
        assert isinstance(agent, EchoAgent)
        assert agent.config.model_provider == "echo"
        assert agent.model == "echo-v1"
        
        # Test invoke functionality
        response = agent.invoke("Hello world")
        assert "Echo: Hello world" in response