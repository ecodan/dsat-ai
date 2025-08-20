# Dan's Simple Agent Toolkit (DSAT)

DSAT is a comprehensive Python toolkit for building LLM applications and running experiments. It provides three core components that work independently or together:

## 💬 [Chat CLI](readme-chat.md)

An interactive terminal-based chat interface for testing prompts and having conversations with LLM agents.

**Key Features:**
- **Zero-config mode**: Auto-detects providers via environment variables
- **Real-time streaming**: Token-by-token streaming support for all providers
- **Multiple usage patterns**: Config files, inline creation, or auto-discovery
- **Interactive commands**: `/help`, `/agents`, `/switch`, `/stream`, `/memory`, `/compact`, and more
- **Memory management**: Configurable conversation limits, auto-compaction, and persistent storage
- **Flexible prompts**: Multiple directory search strategies and per-agent overrides
- **Plugin system**: Entry points for custom LLM provider extensions
- **Session management**: History tracking and conversation export

**Quick Start:**
```bash
# Zero-config (with API key in environment)
dsat chat

# Enable real-time streaming
dsat chat --stream

# Use existing agent configuration
dsat chat --config agents.json --agent my_assistant

# Create agent inline
dsat chat --provider anthropic --model claude-3-5-haiku-latest
```

## 🤖 [Agents Framework](readme-agents.md)

A unified interface for working with multiple LLM providers through configuration-driven agents.

**Key Features:**
- **Multi-provider support**: Anthropic Claude, Google Vertex AI, Ollama (local models)
- **Async streaming support**: Real-time token streaming with `invoke_async()` method
- **Configuration-driven**: JSON configs + TOML prompt templates
- **Comprehensive logging**: Standard Python logging, JSONL files, or custom callbacks
- **Prompt versioning**: Versioned prompt management with TOML templates
- **Factory patterns**: Easy agent creation and management

**Quick Example:**
```python
from agents.agent import Agent, AgentConfig

config = AgentConfig(
    agent_name="my_assistant",
    model_provider="anthropic",  # or "google", "ollama"
    model_family="claude", 
    model_version="claude-3-5-haiku-latest",
    prompt="assistant:v1",
    provider_auth={"api_key": "your-api-key"},
    stream=True,  # Enable streaming support
    memory_enabled=True,  # Enable conversation memory
    max_memory_tokens=8000  # Configure memory limit
)

agent = Agent.create(config)

# Traditional response
response = agent.invoke("Hello, how are you?")

# Streaming response
async for chunk in agent.invoke_async("Tell me a story"):
    print(chunk, end='', flush=True)
```

## 📊 [Scryptorum Framework](readme-scryptorum.md)

A modern, annotation-driven framework for running and tracking LLM experiments.

**Key Features:**
- **Dual run types**: Trial runs (logs only) vs Milestone runs (full versioning) 
- **Annotation-driven**: `@experiment`, `@metric`, `@timer`, `@llm_call` decorators
- **CLI-configurable**: Same code runs as trial or milestone based on CLI flags
- **Thread-safe logging**: JSONL format for metrics, timings, and LLM calls
- **Project integration**: Seamlessly integrates with existing Python projects

**Quick Example:**
```python
from scryptorum import experiment, metric, timer

@experiment(name="sentiment_analysis")
def main():
    reviews = load_reviews()
    results = []
    
    for review in reviews:
        sentiment = analyze_sentiment(review)
        results.append(sentiment)
    
    accuracy = calculate_accuracy(results)
    return accuracy

@timer("data_loading")
def load_reviews():
    return ["Great product!", "Terrible service", "Love it!"]

@metric(name="accuracy", metric_type="accuracy")
def calculate_accuracy(results):
    return 0.85
```

## 🔧 Framework Integration

When used together, DSAT provides `AgentExperiment` and `AgentRun` classes that extend Scryptorum's base classes with agent-specific capabilities:

```python
from agents.agent_experiment import AgentExperiment
from scryptorum import metric

@experiment(name="agent_evaluation")
def evaluate_agents():
    # Load agents from configs
    agent1 = Agent.create(config1)
    agent2 = Agent.create(config2)
    
    # Run evaluation with automatic LLM call logging
    score1 = evaluate_agent(agent1)
    score2 = evaluate_agent(agent2) 
    
    return {"agent1": score1, "agent2": score2}
```

## 🚀 Quick Start

### Installation
```bash
# Basic installation
git clone <repository-url>
cd dsat
uv sync

# With optional dependencies
uv sync --extra dev      # Development tools
uv sync --extra server   # HTTP server support
```

### Initialize a Project
```bash
# Initialize scryptorum in your Python project
scryptorum init

# Create your first experiment
scryptorum create-experiment my_experiment
```

### Run Examples
```bash
# Interactive chat interface
dsat chat --config examples/config/agents.json --agent pirate

# Agent conversation demo
python examples/agents/conversation.py

# Agent logging examples  
python examples/agents/agent_logging_examples.py

# Complete experiment with agent evaluation
python examples/scryptorum/literary_evaluation.py
```

## 📁 Examples

The [`examples/`](examples/) directory contains comprehensive demonstrations:

- **[`examples/agents/`](examples/agents/)**: Agent framework examples including logging patterns and character conversations
- **[`examples/scryptorum/`](examples/scryptorum/)**: Experiment tracking examples with literary agent evaluation
- **[`examples/config/`](examples/config/)**: Shared configurations and prompt templates
- **[`examples/flexible-prompts/`](examples/flexible-prompts/)**: Chat CLI examples with flexible prompts directory management

## 🏗️ Architecture

```
your_project/                    ← Your Python Package
├── src/your_package/
│   ├── experiments/             ← Your experiment code
│   └── agents/                  ← Your agent code  
├── .scryptorum                  ← Scryptorum config
└── pyproject.toml              ← Dependencies

~/experiments/                   ← Scryptorum Project (separate location)
├── your_package/               ← Project tracking
│   ├── experiments/            ← Experiment data & results
│   │   └── my_experiment/
│   │       ├── runs/           ← Trial & milestone runs
│   │       ├── config/         ← Agent configs
│   │       └── prompts/        ← Prompt templates
│   └── data/                   ← Shared data
```

## 📖 Documentation

- **[Chat CLI](readme-chat.md)**: Interactive terminal chat interface for agent testing
- **[Agents Framework](readme-agents.md)**: Multi-provider LLM agent system
- **[Scryptorum Framework](readme-scryptorum.md)**: Experiment tracking and management
- **[Examples Documentation](examples/README.md)**: Comprehensive examples and tutorials

## 🛠️ Development

```bash
# Install development dependencies
uv sync --extra dev

# Run tests
python -m pytest test/ -v

# Format code
black src/

# Lint code  
ruff check src/
```

## 📄 License

MIT License - see LICENSE file for details.

---

*DSAT simplifies LLM application development by providing unified agent abstractions and comprehensive experiment tracking with minimal boilerplate.*