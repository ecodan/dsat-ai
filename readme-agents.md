# Agents System Documentation

The DSAT agents system provides a unified interface for working with multiple LLM providers through a configuration-driven approach. It supports Anthropic Claude and Google Vertex AI models with extensible prompt management.

## Quick Start

### Installation

Install with the required dependencies:

```bash
# Basic installation
uv sync

# With Anthropic support
pip install anthropic

# With Google Vertex AI support  
pip install google-cloud-aiplatform
```

### Basic Usage

```python
from src.agents.agent import Agent, AgentConfig

# Create a configuration
config = AgentConfig(
    agent_name="my_assistant",
    model_provider="anthropic",
    model_family="claude", 
    model_version="claude-3-5-haiku-latest",
    prompt="assistant:v1",
    provider_auth={"api_key": "your-api-key"},
    model_parameters={"temperature": 0.7, "max_tokens": 4096}
)

# Create and use the agent
agent = Agent.create(config)
response = agent.invoke("Hello, how are you?")
print(response)
```

## Core Components

### AgentConfig

The `AgentConfig` class defines the configuration for an agent:

```python
config = AgentConfig(
    agent_name="assistant",           # Required: Unique identifier
    model_provider="anthropic",       # Required: "anthropic" or "google" 
    model_family="claude",           # Required: Model family
    model_version="claude-3-5-haiku-latest",  # Required: Specific model
    prompt="assistant:v1",         # Required: Prompt in format "name:version" or "name:latest"
    model_parameters={              # Optional: Model-specific parameters
        "temperature": 0.7,
        "max_tokens": 4096
    },
    provider_auth={                 # Optional: Authentication credentials
        "api_key": "your-key"
    },
    custom_configs={},              # Optional: Custom configuration
    tools=[]                        # Optional: Available tools
)
```

### Agent Factory

Use `Agent.create()` to instantiate agents:

```python
# Create from config
agent = Agent.create(config)

# With custom logger and prompts directory
agent = Agent.create(
    config, 
    logger=my_logger,
    prompts_dir="./my_prompts"
)
```

## Supported Providers

### Anthropic Claude

```python
config = AgentConfig(
    agent_name="claude_agent",
    model_provider="anthropic",
    model_family="claude",
    model_version="claude-3-5-haiku-latest",  # or claude-3-5-sonnet-latest, etc.
    prompt="assistant:v1",
    provider_auth={"api_key": "sk-ant-..."},
    model_parameters={
        "temperature": 0.7,
        "max_tokens": 4096
    }
)
```

**Required auth fields:** `api_key`

### Google Vertex AI

```python
config = AgentConfig(
    agent_name="vertex_agent", 
    model_provider="google",
    model_family="gemini",
    model_version="gemini-2.0-flash",  # or gemini-pro, etc.
    prompt="assistant:v1", 
    provider_auth={
        "project_id": "your-gcp-project",
        "location": "us-central1"  # Optional, defaults to us-central1
    },
    model_parameters={
        "temperature": 0.3,
        "max_output_tokens": 20000
    }
)
```

**Required auth fields:** `project_id`  
**Optional auth fields:** `location` (defaults to "us-central1")

## Prompt Management

The system includes a sophisticated prompt management system that supports versioning and templates.

### Creating Prompts

```python
from src.agents.prompts import PromptManager

# Initialize prompt manager
pm = PromptManager("./prompts")

# Create a new prompt
pm.create_prompt("assistant", "You are a helpful AI assistant.")

# Add a new version
pm.add_version("assistant", "You are a helpful AI assistant with advanced reasoning.")
```

### Prompt File Format

Prompts are stored as TOML files:

```toml
# prompts/assistant.toml
v1 = """You are a helpful AI assistant."""
v2 = """You are a helpful AI assistant with advanced reasoning capabilities."""
v3 = """You are a helpful AI assistant. Be concise and accurate."""
```

### Prompt Format

The `prompt` field uses the format `"name:version"`:

- **Specific version**: `"assistant:v1"` or `"assistant:2"`
- **Latest version**: `"assistant:latest"`
- **Number versions**: `"assistant:1"`, `"assistant:2"`, etc.

Examples:
```python
config = AgentConfig(
    # ... other fields
    prompt="assistant:v1",     # Use version 1
    # or
    prompt="assistant:latest", # Use latest version
    # or  
    prompt="assistant:3",      # Use version 3
)
```

### Using Prompts

```python
# Get latest version (automatic with "latest")
config = AgentConfig(prompt="assistant:latest", ...)
agent = Agent.create(config)
response = agent.invoke("Hello")  # Uses latest prompt version

# Get specific version  
config = AgentConfig(prompt="assistant:v1", ...)
agent = Agent.create(config)
response = agent.invoke("Hello")  # Uses v1

# Use explicit system prompt (overrides config)
response = agent.invoke("Hello", "Custom system prompt")
```

## Configuration Management

### Save and Load Configurations

```python
# Save single configuration
config.save_to_file("config.json")

# Save multiple configurations
configs = {"agent1": config1, "agent2": config2}
config1.save_to_file("multi_config.json", configs)

# Load configurations
loaded_configs = AgentConfig.load_from_file("config.json")
agent = Agent.create(loaded_configs["agent1"])
```

### Configuration Files

**JSON format:**
```json
{
  "assistant": {
    "agent_name": "assistant",
    "model_provider": "anthropic", 
    "model_family": "claude",
    "model_version": "claude-3-5-haiku-latest",
    "prompt": "assistant:v1",
    "model_parameters": {
      "temperature": 0.7,
      "max_tokens": 4096
    },
    "provider_auth": {
      "api_key": "sk-ant-..."
    }
  }
}
```

**TOML format:**
```toml
[assistant]
agent_name = "assistant"
model_provider = "anthropic"
model_family = "claude" 
model_version = "claude-3-5-haiku-latest"
prompt = "assistant:v1"

[assistant.model_parameters]
temperature = 0.7
max_tokens = 4096

[assistant.provider_auth]  
api_key = "sk-ant-..."
```

## Backward Compatibility

The system supports legacy initialization patterns:

### Anthropic Legacy API

```python
from src.agents.anthropic_agent import ClaudeLLMAgent

# Legacy initialization
agent = ClaudeLLMAgent(
    api_key="sk-ant-...",
    model="claude-3-5-haiku-latest",
    logger=logger,
    prompts_dir="./prompts"
)
```

### Google Vertex AI Legacy API

```python  
from src.agents.vertex_agent import GoogleVertexAIAgent

# Legacy initialization
agent = GoogleVertexAIAgent(
    project_id="your-project",
    location="us-central1", 
    model="gemini-2.0-flash",
    logger=logger,
    prompts_dir="./prompts"
)
```

## Error Handling

The system provides comprehensive error handling:

```python
try:
    agent = Agent.create(config)
    response = agent.invoke("Hello")
except ImportError as e:
    print(f"Missing dependency: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"API error: {e}")
```

**Common errors:**
- `ImportError`: Missing optional dependencies (anthropic, google-cloud-aiplatform)
- `ValueError`: Invalid configuration or missing required fields
- `FileNotFoundError`: Prompt file not found
- API-specific exceptions for network/auth issues

## Advanced Usage

### Custom Logging

```python
import logging

# Create custom logger
logger = logging.getLogger("my_agent")
logger.setLevel(logging.DEBUG)

# Use with agent
agent = Agent.create(config, logger=logger)
```

### Multiple Agents

```python
# Load multiple agent configurations
configs = AgentConfig.load_from_file("agents.json")

# Create multiple agents
claude_agent = Agent.create(configs["claude"])
vertex_agent = Agent.create(configs["vertex"])

# Use different agents for different tasks
creative_response = claude_agent.invoke("Write a poem")
factual_response = vertex_agent.invoke("What is the capital of France?")
```

### Prompt Templating

```python
# Create parameterized prompts
pm = PromptManager("./prompts")
pm.create_prompt("tutor", "You are a {subject} tutor. Help students learn {topic}.")

# Use with string formatting (manual)
system_prompt = "You are a math tutor. Help students learn algebra."
response = agent.invoke("Explain quadratic equations", system_prompt)
```

### Environment Variables

Set environment variables for default configurations:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_CLOUD_PROJECT="your-project-id"
export PROMPTS_DIR="./custom_prompts"
```

## Project Structure

```
src/agents/
├── __init__.py
├── agent.py              # Base Agent class and AgentConfig
├── anthropic_agent.py    # Anthropic Claude implementation  
├── vertex_agent.py       # Google Vertex AI implementation
└── prompts.py           # Prompt management system

test/
├── test_agents_base.py      # Base agent tests
├── test_agents_config.py    # Configuration tests
├── test_agents_anthropic.py # Anthropic agent tests
├── test_agents_vertex.py    # Vertex AI agent tests
└── test_agents_prompts.py   # Prompt management tests
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest test/ -v

# Run specific test files
python -m pytest test/test_agents_base.py -v
python -m pytest test/test_agents_config.py -v

# Test with coverage
python -m pytest test/ --cov=src/agents
```

The test suite includes 105+ tests covering:
- Agent initialization and configuration
- Factory method creation
- Prompt management and versioning
- Error handling and edge cases
- Backward compatibility
- Mock-based testing (no real API calls)

## Best Practices

1. **Use configuration files** for production deployments
2. **Version your prompts** to track changes and enable rollbacks  
3. **Handle errors gracefully** with try-catch blocks
4. **Use appropriate logging levels** for debugging
5. **Test with mock objects** to avoid API costs during development
6. **Store credentials securely** using environment variables or secret management
7. **Use factory methods** (`Agent.create()`) instead of direct instantiation
8. **Cache agent instances** for repeated use to avoid re-initialization overhead

## License

MIT License - see LICENSE file for details.