# DSAT Examples

This directory contains examples demonstrating the key components of DSAT (Dan's Simple Agent Toolkit):

## 🤖 [Agents Framework](agents/)

Examples showing the unified agent abstraction for working with different LLM providers.

**Key Examples:**
- **Agent Logging**: 6 different logging configurations for production use
- **Character Conversation**: Pirate vs Shakespeare debate about literature

**Features Demonstrated:**
- Multi-provider support (Anthropic, OpenAI, Ollama)
- Flexible logging (standard, JSONL, custom callbacks)
- Configuration-driven agent creation
- Real-time multi-agent conversations

## 📊 [Scryptorum Framework](scryptorum/)

Examples showing the experiment tracking and management framework.

**Key Examples:**
- **Literary Agent Evaluation**: Complete agent comparison experiment
- **Decorator-Based Example**: Transparent trial/milestone execution patterns
- **Class-Based Example**: Structured experiment with lifecycle methods

**Features Demonstrated:**
- Experiment management with `@experiment` decorator
- Metric tracking with `@metric` decorators
- Performance timing with `@timer` decorators  
- Judge agent pattern for evaluation
- JSONL logging and result persistence
- Environment-based configuration
- BaseRunnable class patterns for complex experiments
- Transparent trial vs milestone run modes

## 🔧 [Configuration](config/)

Shared configuration files used across examples:

- **`agents.json`**: Agent configurations for pirate and Shakespeare characters
- **`prompts/`**: TOML prompt templates for different agent personas
  - `pirate.toml`: Literature-hating sea captain
  - `shakespeare.toml`: Eloquent bard promoting reading
  - `judge.toml`: Impartial evaluator for scoring

## Quick Start

### Prerequisites

**Install Dependencies:**
```bash
# From project root
cd /Users/dan/dev/code/ai/dsat
uv sync --extra dev
```

**Setup Ollama (for local models):**
```bash
# Start Ollama service
ollama serve

# Pull required models
ollama pull qwen       # For agent conversations
ollama pull gemma3n    # For scryptorum experiments
```

### Run the Examples

**Agent Conversation:**
```bash
python examples/agents/conversation.py
```

**Agent Logging Demo:**
```bash
python examples/agents/agent_logging_examples.py
```

**Complete Agent Evaluation Experiment:**
```bash
python examples/scryptorum/literary_evaluation.py
```

**Decorator-Based Experiment Example:**
```bash
# Run as trial (lightweight, reuses directory)
scryptorum run transparent_experiment --script examples/scryptorum/example_experiment_script.py

# Run as milestone (versioned, full artifacts)
scryptorum run transparent_experiment --script examples/scryptorum/example_experiment_script.py --milestone

# Or directly as Python script
python examples/scryptorum/example_experiment_script.py
```

**Class-Based Experiment Example:**
```bash
# Run as trial
scryptorum run sentiment_analysis --module examples.scryptorum.example_experiment_runnable.SentimentAnalysisRunnable

# Run as milestone
scryptorum run sentiment_analysis --module examples.scryptorum.example_experiment_runnable.SentimentAnalysisRunnable --milestone

# Or directly as Python script
python examples/scryptorum/example_experiment_runnable.py
```

## Example Output

### Agent Conversation
```
🏴‍☠️  PIRATE vs SHAKESPEARE: A LITERARY DEBATE  📚
===============================================================================
🔄 ROUND 1
📜 SHAKESPEARE:
Good morrow! I come to speak of literature's might, how books can fill a soul with pure delight!

🏴‍☠️ PIRATE:
Arrr! Books be for landlubbers and scurvy dogs! Give me the open seas and treasure, not dusty tomes!
```

### Scryptorum Experiment
```
🏴‍☠️ Starting Literary Agent Evaluation Experiment 📚
📁 Experiments will be saved to: /Users/dan/dev/code/ai/dsat/examples/scryptorum/experiments/
Loading test agents...
Creating judge agent...
🏴‍☠️ Evaluating pirate agent...
🎭 Evaluating shakespeare agent...

📊 Experiment Results:
🏴‍☠️ Pirate Agent Average Score: 3.2/10
🎭 Shakespeare Agent Average Score: 7.8/10
📈 Score Difference: 4.6 (positive = Shakespeare wins)
🏆 Winner: Shakespeare
```

### Decorator-Based Experiment  
```
Running transparent experiment...
Experiment completed with accuracy: 1.0, throughput: 5.0 items/sec
[INFO] Run trial_run completed in 1.34 seconds
```

### Class-Based Experiment
```
[DEBUG] Preparing sentiment analysis experiment...
[DEBUG] Prepared 50 data points in 0.101s
[DEBUG] Running sentiment analysis...
[DEBUG] Processed 50 results in 1.020s
[DEBUG] Evaluating results...
[DEBUG] Final accuracy: 1.000, throughput: 5.0 items/sec
[DEBUG] Cleaning up experiment...
[INFO] Run trial_run completed in 1.15 seconds
```

## Directory Structure

```
examples/
├── README.md                    # This file
├── agents/                      # Agent framework examples
│   ├── README.md
│   ├── agent_logging_examples.py
│   └── conversation.py
├── scryptorum/                  # Experiment framework examples
│   ├── README.md
│   ├── literary_evaluation.py              # Complete agent evaluation
│   ├── example_experiment_script.py        # Decorator-based patterns  
│   ├── example_experiment_runnable.py      # Class-based patterns
│   └── experiments/                        # Generated experiment data
└── config/                     # Shared configurations
    ├── agents.json
    └── prompts/
        ├── pirate.toml
        ├── shakespeare.toml
        └── judge.toml
```

## Architecture Overview

### Agents Framework
- **Multi-provider abstraction**: Uniform interface across LLM providers
- **Configuration-driven**: JSON configs + TOML prompts
- **Production logging**: Multiple logging modes for different use cases
- **Factory patterns**: Easy agent creation and management

### Scryptorum Framework  
- **Experiment management**: Automatic tracking and organization
- **Decorator-driven**: `@experiment`, `@metric`, `@timer` decorators
- **Run types**: Trial runs (logs only) vs Milestone runs (versioned)
- **Data persistence**: JSONL logs + structured result files
- **Environment configuration**: Flexible experiment directory placement

## Use Cases

These examples demonstrate patterns for:

### Production Agent Applications
- Multi-agent conversations and workflows
- Comprehensive logging for compliance and debugging
- Configuration management for different environments
- Error handling and monitoring integration

### Research and Evaluation
- Systematic agent comparison experiments
- Metric tracking and performance analysis
- Reproducible experiment management
- Result persistence and analysis

### Development and Testing
- Agent behavior validation
- Performance measurement and optimization
- Configuration testing across providers
- Integration testing with real LLM services

## Customization

Each example is designed to be easily modified:

- **Add new agents**: Extend `config/agents.json` and create prompt templates
- **Change models**: Update provider configurations to use different models
- **Custom metrics**: Add `@metric` decorators for new evaluation criteria
- **Extended logging**: Implement custom callback functions for specialized logging
- **New experiments**: Create additional scryptorum experiments following the literary evaluation pattern

These examples provide a solid foundation for building sophisticated LLM-powered applications using the DSAT toolkit.