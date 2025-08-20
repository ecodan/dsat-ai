# Agents Framework Examples

This directory contains examples demonstrating the DSAT agents framework, which provides an abstraction for working with different LLM providers through a unified interface.

## Overview

The agents framework supports:
- **Multiple LLM providers**: Anthropic, Google Vertex AI, Ollama, etc.
- **Real-time streaming**: Token-by-token async streaming with `invoke_async()`
- **Flexible configuration**: JSON-based agent configs with TOML prompts
- **Comprehensive logging**: Standard Python logging, JSONL files, or custom callbacks
- **Agent factory patterns**: Easy creation from configurations

## Examples

### 1. Agent Logging Examples (`agent_logging_examples.py`)

Demonstrates comprehensive logging configurations for agents:

- **Standard Python logging**: Integration with host app logging
- **Dedicated JSONL files**: Detailed logs for analysis and compliance
- **Environment variable configuration**: Runtime configuration without code changes
- **Custom callback logging**: Integration with databases and monitoring systems
- **Disabled logging**: Zero overhead for privacy-sensitive applications
- **Host app integration**: Full integration with existing logging infrastructure

**Usage:**
```bash
cd /Users/dan/dev/code/ai/dsat/examples/agents
python agent_logging_examples.py
```

**Key Features Shown:**
- Six different logging configuration patterns
- Environment variable overrides
- Custom callback functions for advanced logging
- Production-ready logging setup examples

### 2. Pirate vs Shakespeare Conversation (`conversation.py`)

Interactive demo showing two character agents in conversation:

- **Character agents**: A literature-hating pirate and Shakespeare-inspired poet
- **Multi-turn conversation**: 5-round debate about literature's value
- **Agent loading**: Configuration from JSON files and TOML prompts
- **Real-time interaction**: Live conversation between two agents

**Usage:**
```bash
cd /Users/dan/dev/code/ai/dsat/examples/agents
python conversation.py
```

**Prerequisites:**
- Ollama running locally (`ollama serve`)
- Qwen model available (`ollama pull qwen`)

**Example Output:**
```
🏴‍☠️  PIRATE vs SHAKESPEARE: A LITERARY DEBATE  📚
===============================================================================
🔄 ROUND 1
📜 SHAKESPEARE:
Good morrow! I come to speak of literature's might, how books can fill a soul with pure delight!

🏴‍☠️ PIRATE:
Arrr! Books be for landlubbers and scurvy dogs! Give me the open seas and treasure, not dusty tomes!
```

## Configuration Files

### Agent Configuration (`../config/agents.json`)

Defines two character agents:
- **Pirate Agent**: Qwen model with pirate personality
- **Shakespeare Agent**: Qwen model with Shakespearean persona

Both configured with:
- JSONL logging to separate files
- Temperature settings for personality
- Ollama provider configuration

### Prompt Templates (`../config/prompts/`)

- `pirate.toml`: Gruff sea captain who despises reading
- `shakespeare.toml`: Eloquent bard promoting literature's value
- `judge.toml`: Impartial evaluator for scoring (used by scryptorum example)

## Dependencies

Required packages:
```bash
pip install requests  # For agent HTTP communication
```

**Ollama Setup:**
```bash
# Start Ollama service
ollama serve

# Pull required model
ollama pull qwen
```

## Key Concepts Demonstrated

### 1. Agent Factory Pattern
```python
# Load configurations from file
agent_configs = AgentConfig.load_from_file("agents.json")

# Create agent from config
pirate_agent = Agent.create(
    config=agent_configs["pirate"], 
    prompts_dir=prompts_dir
)
```

### 2. Flexible Logging Configuration
```python
"custom_configs": {
    "logging": {
        "enabled": true,
        "mode": "jsonl_file",  # or "standard", "callback"
        "file_path": "./logs/agent_calls.jsonl",
        "level": "standard"  # or "minimal"
    }
}
```

### 3. Multi-Provider Support
The framework supports switching between providers by changing configuration:
```json
{
    "model_provider": "anthropic",  // or "ollama", "openai"
    "model_family": "claude",
    "model_version": "claude-3-5-haiku-latest"
}
```

## Extending the Examples

### Add New Agents
1. Add configuration to `../config/agents.json`
2. Create prompt template in `../config/prompts/`
3. Load and use in your script

### Custom Logging
Implement custom callback functions:
```python
def my_logger(call_data):
    # Send to database, monitoring system, etc.
    process_llm_call(call_data)

config.custom_configs["logging"]["callback"] = my_logger
```

### Different Models
Update configurations to use different models:
- Change `model_provider` for different services
- Update `model_version` for specific model versions
- Adjust `model_parameters` for fine-tuning behavior

These examples provide a foundation for building sophisticated agent-based applications with the DSAT framework.