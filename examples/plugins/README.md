# DSAT Plugin System Examples

This directory contains examples of how to create custom LLM provider plugins for DSAT.

## Plugin Structure

A DSAT plugin is a Python package that:

1. **Extends the Agent base class** with a custom provider implementation
2. **Registers itself via entry points** in `pyproject.toml`
3. **Follows the Agent interface** for consistency

## Example: OpenAI Plugin

Here's how you would create an OpenAI plugin:

### 1. Package Structure

```
my-dsat-openai/
├── pyproject.toml
├── src/
│   └── dsat_openai/
│       ├── __init__.py
│       └── openai_agent.py
```

### 2. Agent Implementation (`src/dsat_openai/openai_agent.py`)

```python
import logging
from typing import Optional, Union
from pathlib import Path

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from dsat.agents.agent import Agent, AgentConfig


class OpenAIAgent(Agent):
    """OpenAI GPT agent implementation."""
    
    def __init__(self, config: AgentConfig, logger: logging.Logger = None, 
                 prompts_dir: Optional[Union[str, Path]] = None):
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for OpenAI provider")
            
        super().__init__(config, logger, prompts_dir)
        
        # Get API key from config
        api_key = config.provider_auth.get("api_key")
        if not api_key:
            raise ValueError("api_key required in provider_auth for OpenAI provider")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        
    def invoke(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send prompts to OpenAI and return response."""
        
        # Get system prompt from config if not provided
        if system_prompt is None:
            system_prompt = self.get_system_prompt()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            # Log the call if logger is configured
            if self.call_logger:
                self.call_logger.log_call(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=self.model,
                    config=self.config.to_dict()
                )
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.config.model_version,
                messages=messages,
                **self.config.model_parameters
            )
            
            result = response.choices[0].message.content
            
            # Log the response
            if self.call_logger:
                self.call_logger.log_response(result)
                
            return result
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    @property
    def model(self) -> str:
        """Return the model name."""
        return self.config.model_version
```

### 3. Package Exports (`src/dsat_openai/__init__.py`)

```python
from .openai_agent import OpenAIAgent

__all__ = ["OpenAIAgent"]
```

### 4. Entry Point Registration (`pyproject.toml`)

```toml
[project]
name = "dsat-openai"
version = "0.1.0"
description = "OpenAI provider plugin for DSAT"
dependencies = [
    "dsat-ai>=0.1.0",
    "openai>=1.0.0"
]

# Register the plugin
[project.entry-points."dsat.providers"]
openai = "dsat_openai:OpenAIAgent"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Usage

### 1. Install the Plugin

```bash
pip install dsat-openai
```

### 2. Configure Agent

Create an `agents.json` file:

```json
{
  "my_gpt": {
    "model_provider": "openai",
    "model_family": "gpt",
    "model_version": "gpt-4o",
    "prompt": "assistant:latest",
    "model_parameters": {
      "temperature": 0.7,
      "max_tokens": 1000
    },
    "provider_auth": {
      "api_key": "sk-..."
    }
  }
}
```

### 3. Use in Chat

```bash
# Plugin is automatically discovered
dsat chat --agent my_gpt

# Or inline
dsat chat --provider openai --model gpt-4o
```

## Available Commands

Check what providers are available:

```bash
dsat chat
# Then use /providers command to see:
# Available LLM Providers:
#   Built-in Providers:
#     anthropic
#     google  
#     ollama
#   Plugin Providers:
#     openai
```

## Plugin Development Tips

1. **Always check for required packages** in your `__init__.py` or agent class
2. **Follow the same patterns** as built-in agents for consistency
3. **Handle authentication** through `config.provider_auth`
4. **Support model parameters** via `config.model_parameters`
5. **Integrate with logging** using `self.call_logger`
6. **Document your configuration requirements** clearly

## Testing Your Plugin

```python
# test_openai_plugin.py
from dsat.agents.agent import Agent, AgentConfig

def test_openai_plugin():
    config = AgentConfig(
        agent_name="test_openai",
        model_provider="openai",
        model_family="gpt", 
        model_version="gpt-4o",
        prompt="assistant:latest",
        provider_auth={"api_key": "sk-test..."}
    )
    
    agent = Agent.create(config)
    response = agent.invoke("Hello, world!")
    assert response is not None
```