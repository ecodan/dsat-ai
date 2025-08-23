# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2025-08-23

### Changed
- **Chat command terminology**: Renamed `/compact` command to `/prune` for better accuracy since it removes older messages rather than compacting them

## [0.3.1] - 2025-08-23

### Added

#### 🌐 LiteLLM Integration - Access to 100+ LLM Providers
- **LiteLLMAgent provider** - New `litellm` provider supporting 100+ LLM providers through unified interface
- **Universal model access** - Access OpenAI, Anthropic, Google, Azure, AWS Bedrock, Cohere, HuggingFace, Groq, Replicate, and many more
- **Unified API** - Single configuration format works across all LiteLLM-supported providers
- **Streaming support** - Full streaming functionality with `invoke_async()` for all supported providers
- **Authentication flexibility** - Support for various authentication methods including API keys, OAuth, and service accounts
- **Model format** - Use `provider/model_name` format (e.g., `openai/gpt-4o`, `anthropic/claude-3-5-sonnet-20241022`)

### Changed

#### Enhanced Provider Support
- **Extended provider ecosystem** - From 3 native providers (Anthropic, Google, Ollama) to 100+ via LiteLLM
- **Dependency management** - Added `litellm` as optional dependency with `uv sync --extra litellm`
- **Documentation updates** - Comprehensive LiteLLM integration guide with model discovery resources

### Installation

```bash
# Install with LiteLLM support
uv sync --extra litellm

# Or all providers
uv sync --extra all
```

### Example Usage

```python
from dsat.agents.agent import Agent, AgentConfig

# OpenAI via LiteLLM
config = AgentConfig(
    agent_name="openai_assistant",
    model_provider="litellm",
    model_family="openai",
    model_version="openai/gpt-4o",
    provider_auth={"api_key": "sk-..."},
    stream=True
)

agent = Agent.create(config)
response = agent.invoke("Hello from OpenAI!")

# Anthropic via LiteLLM  
anthropic_config = AgentConfig(
    model_provider="litellm",
    model_version="anthropic/claude-3-5-sonnet-20241022",
    provider_auth={"api_key": "sk-ant-..."}
)
```

### Provider Coverage
- **Direct providers**: Anthropic, Google Vertex AI, Ollama (unchanged)
- **LiteLLM providers**: OpenAI, Azure OpenAI, AWS Bedrock, Cohere, HuggingFace, Groq, Replicate, Together AI, Fireworks AI, XAI Grok, and 90+ more
- **Model discovery**: https://models.litellm.ai/ for complete provider and model listings

## [0.3.0] - 2025-08-20

### Added

#### =� Conversational Memory System
- **Intelligent conversation history** - Agents now maintain context across conversation turns
- **Configurable memory limits** - Set `max_memory_tokens` to control conversation length
- **Automatic memory management** - Smart compaction removes older messages while preserving context
- **Response truncation** - Configure `response_truncate_length` to limit long responses
- **Persistent storage** - Conversations automatically saved to `~/.dsat/chat_history/`
- **Memory commands** - New chat commands: `/memory`, `/compact`, `/clear`, `/history`
- **Memory status display** - Color-coded memory usage indicators in chat interface

#### <
 Real-time Streaming Support  
- **Token-by-token streaming** - Real-time response streaming with `invoke_async()` method
- **Multi-provider streaming** - Anthropic Claude, Google Vertex AI, and Ollama streaming support
- **Interactive streaming control** - `/stream` command to toggle streaming on/off during chat
- **CLI streaming flag** - `--stream` flag to enable streaming from command line
- **Streaming integration** - Memory system works seamlessly with streaming responses

#### <� Enhanced Agent Configuration
- **Model definitions** - New `_models` section for reusable model configurations
- **Model references** - Use `model_id` to reference shared model definitions  
- **Memory configuration** - New agent config fields:
  - `memory_enabled: bool` (default: true)
  - `max_memory_tokens: int` (default: 8000) 
  - `response_truncate_length: int` (default: 1000)
- **Backward compatibility** - All existing agent configurations continue to work

#### =' Developer Improvements
- **Enhanced logging** - Added `full_prompt_tokens` field to track conversation context size
- **Comprehensive testing** - Complete test coverage for memory functionality across all providers
- **Method signature updates** - All agent `invoke()` methods now support optional `history` parameter
- **Documentation updates** - Extensive memory system documentation with examples

### Changed

#### Agent Configuration Format
- **Extended config schema** - Agent configs now support memory and model definition fields
- **Flexible model references** - Agents can inherit from model definitions or define models inline
- **Improved validation** - Better error messages for invalid configurations

#### Chat Interface Enhancements  
- **Memory-aware status** - Chat interface shows memory usage and message count
- **Enhanced help system** - Updated `/help` command with memory and streaming information
- **Improved user experience** - Better visual feedback for streaming and memory operations

### Technical Details

#### Memory System Architecture
- **ConversationMessage** - Structured message storage with role, content, timestamp, and token count
- **TokenCounter** - Efficient token estimation using 4-chars-per-token heuristic  
- **MemoryManager** - Centralized memory operations including compaction and persistence
- **Provider-specific context building** - Each LLM provider formats conversation history appropriately:
  - **Anthropic**: Messages array format with role/content objects
  - **Vertex AI**: Conversational context strings with Human:/Assistant: labels
  - **Ollama**: Similar conversational format optimized for local models

#### Streaming Implementation
- **AsyncGenerator support** - All agents implement `invoke_async()` for streaming responses
- **Concurrent processing** - Memory management works during streaming without blocking
- **Provider-specific streaming** - Native streaming APIs used for each provider
- **Graceful fallback** - Streaming gracefully falls back to regular responses when needed

#### Backward Compatibility
- **Method signatures** - `history` parameter is optional with `None` default
- **Existing configurations** - All current agent.json files work without modification  
- **API compatibility** - Existing code using `invoke()` continues to work unchanged

### Example Configuration

```json
{
  "_models": {
    "local-qwen": {
      "model_provider": "ollama",
      "model_family": "qwen", 
      "model_version": "qwen",
      "provider_auth": {"base_url": "http://localhost:11434"}
    }
  },
  "assistant": {
    "model_id": "local-qwen",
    "prompt": "assistant:v1",
    "memory_enabled": true,
    "max_memory_tokens": 8000,
    "response_truncate_length": 1000,
    "custom_configs": {
      "logging": {"enabled": true, "mode": "jsonl_file"}
    }
  }
}
```

### Usage Examples

```bash
# Enable streaming and memory
dsat chat --stream --config agents.json --agent assistant

# Memory management commands
/memory          # Show memory statistics
/compact         # Reduce memory usage
/clear           # Clear conversation history
/history         # Show conversation with memory stats
```

```python
# Programming with conversation history
from dsat.agents.agent import Agent, AgentConfig
from dsat.cli.memory import ConversationMessage

config = AgentConfig(
    agent_name="assistant",
    model_provider="anthropic",
    model_family="claude",
    model_version="claude-3-5-haiku-latest", 
    memory_enabled=True,
    max_memory_tokens=8000
)

agent = Agent.create(config)

# Build conversation
history = [
    ConversationMessage("user", "What is Python?", "2023-01-01T00:00:00", 8),
    ConversationMessage("assistant", "Python is a programming language.", "2023-01-01T00:01:00", 12)
]

# Continue conversation with context
response = agent.invoke("Tell me more about its features", history=history)

# Or use streaming
async for chunk in agent.invoke_async("What are its main uses?", history=history):
    print(chunk, end='', flush=True)
```