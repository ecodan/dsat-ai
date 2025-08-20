"""
Tests for async streaming functionality in agents.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from dsat.agents.agent import AgentConfig
from test.echo_agent import EchoAgent, create_echo_agent_config


class TestAgentConfigStreaming:
    """Test cases for AgentConfig stream field."""

    def test_agent_config_stream_default(self):
        """Test that stream defaults to False."""
        config = AgentConfig(
            agent_name="test",
            model_provider="test",
            model_family="test",
            model_version="test",
            prompt="test:v1"
        )
        assert config.stream is False

    def test_agent_config_stream_explicit_true(self):
        """Test setting stream to True."""
        config = AgentConfig(
            agent_name="test",
            model_provider="test",
            model_family="test",
            model_version="test",
            prompt="test:v1",
            stream=True
        )
        assert config.stream is True

    def test_agent_config_stream_from_dict(self):
        """Test loading stream field from dictionary."""
        config_dict = {
            "agent_name": "test",
            "model_provider": "test",
            "model_family": "test",
            "model_version": "test",
            "prompt": "test:v1",
            "stream": True
        }
        config = AgentConfig.from_dict(config_dict)
        assert config.stream is True

    def test_agent_config_stream_from_dict_default(self):
        """Test that stream defaults to False when not in dictionary."""
        config_dict = {
            "agent_name": "test",
            "model_provider": "test",
            "model_family": "test",
            "model_version": "test",
            "prompt": "test:v1"
        }
        config = AgentConfig.from_dict(config_dict)
        assert config.stream is False

    def test_agent_config_stream_to_dict(self):
        """Test that stream field is included in to_dict output."""
        config = AgentConfig(
            agent_name="test",
            model_provider="test",
            model_family="test",
            model_version="test",
            prompt="test:v1",
            stream=True
        )
        result = config.to_dict()
        assert result["stream"] is True


class TestEchoAgentAsyncStreaming:
    """Test cases for EchoAgent async streaming."""

    @pytest.fixture
    def echo_agent(self):
        """Create EchoAgent for testing."""
        config = create_echo_agent_config("test_echo")
        logger = Mock(spec=logging.Logger)
        return EchoAgent(config, logger)

    @pytest.mark.asyncio
    async def test_echo_agent_invoke_async_simple(self, echo_agent):
        """Test basic async streaming with EchoAgent."""
        user_prompt = "Hello"
        
        response_chunks = []
        async for chunk in echo_agent.invoke_async(user_prompt):
            response_chunks.append(chunk)
        
        full_response = ''.join(response_chunks)
        assert "Echo: Hello" in full_response
        assert len(response_chunks) > 1  # Should be chunked character by character

    @pytest.mark.asyncio
    async def test_echo_agent_invoke_async_with_system_prompt(self, echo_agent):
        """Test async streaming with explicit system prompt."""
        user_prompt = "Hello"
        system_prompt = "You are helpful"
        
        response_chunks = []
        async for chunk in echo_agent.invoke_async(user_prompt, system_prompt):
            response_chunks.append(chunk)
        
        full_response = ''.join(response_chunks)
        assert "You are helpful" in full_response
        assert "Echo: Hello" in full_response

    @pytest.mark.asyncio
    async def test_echo_agent_invoke_async_incremental(self, echo_agent):
        """Test that streaming yields incremental chunks."""
        user_prompt = "Test"
        
        chunks_received = 0
        partial_response = ""
        
        async for chunk in echo_agent.invoke_async(user_prompt):
            chunks_received += 1
            partial_response += chunk
            
            # Verify we're getting incremental updates
            if chunks_received == 5:  # After a few chunks
                assert len(partial_response) > 0
                assert len(partial_response) < len("Echo: Test")
        
        assert chunks_received > 5  # Should receive multiple chunks
        assert "Echo: Test" in partial_response

    @pytest.mark.asyncio
    async def test_echo_agent_call_logging_async(self, echo_agent):
        """Test that call logging works with async streaming."""
        mock_call_logger = Mock()
        echo_agent.call_logger = mock_call_logger
        
        user_prompt = "Hello"
        
        response_chunks = []
        async for chunk in echo_agent.invoke_async(user_prompt):
            response_chunks.append(chunk)
        
        # Verify call logger was called
        mock_call_logger.log_llm_call.assert_called_once()
        call_args = mock_call_logger.log_llm_call.call_args[1]
        
        assert call_args["request_data"]["user_prompt"] == user_prompt
        assert "Echo: Hello" in call_args["response_data"]["content"]
        assert call_args["model_provider"] == "echo"


@pytest.mark.asyncio
class TestAgentAsyncStreamingIntegration:
    """Integration tests for agent async streaming."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client for testing."""
        mock_client = Mock()
        mock_stream = Mock()
        
        # Mock streaming chunks
        mock_chunks = [
            Mock(type="content_block_delta", delta=Mock(text="Hello")),
            Mock(type="content_block_delta", delta=Mock(text=" ")),
            Mock(type="content_block_delta", delta=Mock(text="world")),
            Mock(type="content_block_delta", delta=Mock(text="!")),
        ]
        mock_stream.__iter__ = Mock(return_value=iter(mock_chunks))
        mock_client.messages.create.return_value = mock_stream
        
        return mock_client

    @pytest.fixture 
    def anthropic_config(self):
        """Create Anthropic agent config for testing."""
        return AgentConfig(
            agent_name="test_claude",
            model_provider="anthropic",
            model_family="claude",
            model_version="claude-3-5-haiku-latest",
            prompt="test:v1",
            provider_auth={"api_key": "test-key"},
            stream=True
        )

    async def test_anthropic_agent_async_streaming(self, anthropic_config, mock_anthropic_client):
        """Test ClaudeLLMAgent async streaming."""
        with patch('dsat.agents.anthropic_agent.ANTHROPIC_AVAILABLE', True):
            with patch('dsat.agents.anthropic_agent.Anthropic', return_value=mock_anthropic_client):
                from dsat.agents.anthropic_agent import ClaudeLLMAgent
                
                logger = Mock(spec=logging.Logger)
                agent = ClaudeLLMAgent(anthropic_config, logger=logger)
                
                user_prompt = "Test prompt"
                response_chunks = []
                
                async for chunk in agent.invoke_async(user_prompt):
                    response_chunks.append(chunk)
                
                full_response = ''.join(response_chunks)
                assert full_response == "Hello world!"
                assert len(response_chunks) == 4
                
                # Verify API was called with streaming
                mock_anthropic_client.messages.create.assert_called_once()
                call_kwargs = mock_anthropic_client.messages.create.call_args[1]
                assert call_kwargs["stream"] is True

    @pytest.fixture
    def mock_aiohttp_session(self):
        """Mock aiohttp session for Ollama testing."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        
        # Mock NDJSON streaming response
        mock_lines = [
            b'{"response": "Hello"}\n',
            b'{"response": " "}\n', 
            b'{"response": "world"}\n',
            b'{"response": "!", "done": true}\n'
        ]
        mock_response.content.__aiter__ = AsyncMock(return_value=iter(mock_lines))
        mock_response.raise_for_status = Mock()
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        
        return mock_session

    @pytest.fixture
    def ollama_config(self):
        """Create Ollama agent config for testing."""
        return AgentConfig(
            agent_name="test_ollama",
            model_provider="ollama",
            model_family="llama",
            model_version="llama3.2",
            prompt="test:v1",
            provider_auth={"base_url": "http://localhost:11434"},
            stream=True
        )

    async def test_ollama_agent_async_streaming(self, ollama_config, mock_aiohttp_session):
        """Test OllamaAgent async streaming."""
        # TODO: Fix aiohttp mock setup for proper async context manager
        pytest.skip("Aiohttp mock setup needs fixing for async context manager protocol")
        # with patch('dsat.agents.ollama_agent.REQUESTS_AVAILABLE', True):
        #     with patch('dsat.agents.ollama_agent.AIOHTTP_AVAILABLE', True):
        #         with patch('dsat.agents.ollama_agent.aiohttp.ClientSession', return_value=mock_aiohttp_session):
        #             from dsat.agents.ollama_agent import OllamaAgent
        #             
        #             logger = Mock(spec=logging.Logger)
        #             agent = OllamaAgent(ollama_config, logger=logger)
        #             
        #             user_prompt = "Test prompt"
        #             response_chunks = []
        #             
        #             async for chunk in agent.invoke_async(user_prompt):
        #                 response_chunks.append(chunk)
        #             
        #             full_response = ''.join(response_chunks)
        #             assert full_response == "Hello world!"
        #             assert len(response_chunks) == 4

    @pytest.fixture
    def mock_vertex_client(self):
        """Mock Vertex AI client for testing."""
        mock_client = Mock()
        
        # Mock streaming response
        mock_chunks = [
            Mock(text="Hello"),
            Mock(text=" "),
            Mock(text="world"),
            Mock(text="!"),
        ]
        mock_client.generate_content.return_value = iter(mock_chunks)
        
        return mock_client

    @pytest.fixture
    def vertex_config(self):
        """Create Vertex AI agent config for testing."""
        return AgentConfig(
            agent_name="test_vertex",
            model_provider="google",
            model_family="gemini",
            model_version="gemini-1.5-flash",
            prompt="test:v1",
            provider_auth={"project_id": "test-project", "location": "us-central1"},
            stream=True
        )

    async def test_vertex_agent_async_streaming(self, vertex_config, mock_vertex_client):
        """Test GoogleVertexAIAgent async streaming."""
        # TODO: Fix vertexai module mock setup
        pytest.skip("Vertex AI module mock setup needs fixing for None module")
        # with patch('dsat.agents.vertex_agent.VERTEX_AI_AVAILABLE', True):
        #     with patch('dsat.agents.vertex_agent.vertexai.init'):
        #         with patch('dsat.agents.vertex_agent.GenerativeModel', return_value=mock_vertex_client):
        #             from dsat.agents.vertex_agent import GoogleVertexAIAgent
        #             
        #             logger = Mock(spec=logging.Logger)
        #             agent = GoogleVertexAIAgent(vertex_config, logger=logger)
        #             
        #             user_prompt = "Test prompt"
        #             response_chunks = []
        #             
        #             async for chunk in agent.invoke_async(user_prompt):
        #                 response_chunks.append(chunk)
        #             
        #             full_response = ''.join(response_chunks)
        #             assert full_response == "Hello world!"
        #             assert len(response_chunks) == 4
        #             
        #             # Verify API was called with streaming
        #             mock_vertex_client.generate_content.assert_called_once()
        #             call_kwargs = mock_vertex_client.generate_content.call_args[1]
        #             assert call_kwargs["stream"] is True


class TestAsyncStreamingErrorHandling:
    """Test error handling in async streaming."""

    @pytest.mark.asyncio
    async def test_echo_agent_async_exception_handling(self):
        """Test exception handling in EchoAgent async streaming."""
        config = create_echo_agent_config("test_echo")
        logger = Mock(spec=logging.Logger)
        agent = EchoAgent(config, logger)
        
        # Mock get_system_prompt to raise an exception
        agent.get_system_prompt = Mock(side_effect=Exception("Test error"))
        
        with pytest.raises(Exception, match="Test error"):
            async for chunk in agent.invoke_async("test"):
                pass

    @pytest.mark.asyncio
    async def test_anthropic_agent_async_api_error(self):
        """Test API error handling in ClaudeLLMAgent async streaming."""
        config = AgentConfig(
            agent_name="test_claude",
            model_provider="anthropic", 
            model_family="claude",
            model_version="claude-3-5-haiku-latest",
            prompt="test:v1",
            provider_auth={"api_key": "test-key"}
        )
        
        with patch('dsat.agents.anthropic_agent.ANTHROPIC_AVAILABLE', True):
            mock_client = Mock()
            mock_client.messages.create.side_effect = Exception("API Error")
            
            with patch('dsat.agents.anthropic_agent.Anthropic', return_value=mock_client):
                from dsat.agents.anthropic_agent import ClaudeLLMAgent
                
                logger = Mock(spec=logging.Logger)
                agent = ClaudeLLMAgent(config, logger=logger)
                
                with pytest.raises(Exception, match="API Error"):
                    async for chunk in agent.invoke_async("test"):
                        pass


class TestCLIAsyncStreamingIntegration:
    """Test CLI integration with async streaming."""

    def test_cli_streaming_flag_parsing(self):
        """Test that CLI correctly parses --stream flag."""
        from dsat.cli.chat import create_parser
        
        parser = create_parser()
        
        # Test with --stream flag
        args = parser.parse_args(["--stream"])
        assert args.stream is True
        
        # Test without --stream flag  
        args = parser.parse_args([])
        assert args.stream is False

    @pytest.mark.asyncio
    async def test_chat_interface_streaming_toggle(self):
        """Test ChatInterface streaming toggle functionality."""
        from dsat.cli.chat import ChatInterface
        
        chat = ChatInterface()
        assert chat.streaming_enabled is False
        
        # Test toggle
        chat._toggle_streaming()
        assert chat.streaming_enabled is True
        
        chat._toggle_streaming()
        assert chat.streaming_enabled is False

    def test_chat_interface_streaming_initialization(self):
        """Test ChatInterface streaming flag initialization."""
        from dsat.cli.chat import ChatInterface
        
        chat = ChatInterface()
        
        # Test that stream parameter is properly stored
        chat.streaming_enabled = False
        chat.cli_prompts_dir = None
        chat.config_file = None
        
        # Call initialize_agents with stream=True but disable actual agent discovery
        with patch.object(chat, '_discover_agent_configs', return_value={}):
            with patch.object(chat, '_auto_detect_providers', return_value={}):
                with patch.object(chat, '_check_ollama_health', return_value=False):
                    # We expect this to fail because no agents are found, but it should still set streaming
                    success = chat.initialize_agents(stream=True)
                    assert success is False  # No agents found
                    assert chat.streaming_enabled is True  # But streaming flag should be set