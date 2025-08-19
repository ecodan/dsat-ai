# DSAT Chat CLI

The DSAT Chat CLI provides an interactive terminal interface for testing prompts and having conversations with LLM agents across multiple providers. It's designed for rapid prototyping, prompt testing, and exploring different agent configurations.

## 🚀 Quick Start

### Zero-Config Usage

The easiest way to get started is with environment variables:

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-key-here"

# Start chatting immediately
dsat chat
```

This will auto-detect the available provider and create a default agent.

### With Configuration Files

For more control, use agent configuration files:

```bash
# Use a specific agent from config
dsat chat --config agents.json --agent my_assistant

# Override prompts directory
dsat chat --config agents.json --agent researcher --prompts-dir ./my-prompts
```

### Inline Agent Creation

Create agents on the fly without configuration files:

```bash
# Specify provider and model directly
dsat chat --provider anthropic --model claude-3-5-haiku-latest

# Works with any supported provider
dsat chat --provider ollama --model llama3.2
```

## 🎛️ Command Line Options

```bash
dsat chat [OPTIONS]

Options:
  -c, --config PATH         Path to agent configuration file (JSON/TOML)
  -a, --agent NAME          Name of agent to use from config file
  -p, --provider PROVIDER   LLM provider (anthropic|google|ollama)
  -m, --model MODEL         Model version for inline creation
  -d, --prompts-dir PATH    Directory containing prompt TOML files
  --no-colors               Disable colored output
  -h, --help               Show help message
```

## 💬 Interactive Commands

Once in the chat interface, use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/agents` | List configured agents |
| `/providers` | Show available LLM providers (built-in + plugins) |
| `/switch <agent>` | Switch to a different agent mid-conversation |
| `/history` | Display conversation history |
| `/clear` | Clear conversation history |
| `/export <file>` | Export conversation to JSON file |
| `/quit` or `/exit` | Exit the chat interface |

## 🔧 Agent Configuration

### Basic Agent Config (agents.json)

```json
{
  "my_assistant": {
    "model_provider": "anthropic",
    "model_family": "claude",
    "model_version": "claude-3-5-haiku-latest", 
    "prompt": "assistant:v1",
    "model_parameters": {
      "temperature": 0.7,
      "max_tokens": 1000
    },
    "provider_auth": {
      "api_key": "your-api-key"
    }
  }
}
```

### Advanced Configuration with Custom Prompts

```json
{
  "researcher": {
    "model_provider": "anthropic",
    "model_family": "claude",
    "model_version": "claude-3-5-haiku-latest",
    "prompt": "researcher:v1",
    "prompts_dir": "./research-prompts",
    "provider_auth": {
      "api_key": "your-api-key"
    }
  },
  "creative_writer": {
    "model_provider": "anthropic", 
    "model_family": "claude",
    "model_version": "claude-3-5-haiku-latest",
    "prompt": "creative:latest",
    "prompts_dir": "/path/to/creative-prompts",
    "model_parameters": {
      "temperature": 0.9
    },
    "provider_auth": {
      "api_key": "your-api-key"
    }
  }
}
```

## 📁 Flexible Prompts System

The chat CLI uses a flexible search strategy to find prompt files:

### Search Priority Order

1. **CLI argument** (`--prompts-dir /path/to/prompts`)
2. **Agent config field** (`"prompts_dir": "./custom-prompts"`)
3. **Config file relative** (`config_directory/prompts/`)
4. **Current directory** (`./prompts/`)
5. **User home directory** (`~/.dsat/prompts/`)

### Prompt File Format

Prompts are stored in TOML files with versioned templates:

```toml
# prompts/researcher.toml
v1 = '''You are a thorough research assistant. You excel at finding, analyzing, and synthesizing information from multiple sources...'''

v2 = '''You are a research expert with advanced analytical capabilities...'''

latest = '''You are a research expert with advanced analytical capabilities...'''
```

### Example Directory Structures

#### Project-Specific Structure
```
my-project/
├── agents.json
├── prompts/
│   ├── assistant.toml
│   ├── researcher.toml
│   └── creative.toml
└── data/
```

#### Per-Agent Prompts Structure
```
my-project/
├── agents.json
├── research-prompts/
│   └── researcher.toml
├── creative-prompts/
│   └── creative.toml
└── general-prompts/
    └── assistant.toml
```

#### Global User Prompts
```
~/.dsat/
└── prompts/
    ├── assistant.toml
    ├── helper.toml
    └── general.toml
```

## 🔌 Provider Support

### Built-in Providers

#### Anthropic Claude
```bash
export ANTHROPIC_API_KEY="your-key"
dsat chat --provider anthropic --model claude-3-5-haiku-latest
```

#### Google Vertex AI
```bash
export GOOGLE_CLOUD_PROJECT="your-project"
# Or set up application default credentials
dsat chat --provider google --model gemini-1.5-flash
```

#### Ollama (Local)
```bash
# Make sure Ollama is running locally
ollama serve

# Pull a model if needed (optional - chat CLI auto-detects available models)
ollama pull llama3.2

# Auto-detection (uses best available model)
dsat chat --provider ollama

# Or specify a specific model
dsat chat --provider ollama --model llama3.2
```

**Interactive Model Selection**: The chat CLI automatically detects which models are available in your local Ollama installation and prompts you to select your preferred model from the list. If only one model is available, it will be selected automatically.

### Plugin Providers

The chat CLI supports custom providers via entry points. See [`examples/plugins/README.md`](examples/plugins/README.md) for details on creating custom provider plugins.

Check available providers:
```bash
dsat chat
# Then use: /providers
```

## 📊 Session Management

### Conversation History

The chat interface automatically tracks your conversation:

- **In-memory**: Full conversation history during the session
- **Export capability**: Save conversations to JSON files
- **History commands**: View and manage conversation history

### Exporting Conversations

```bash
# In chat, export current conversation
/export my-conversation.json
```

The exported file contains:
```json
{
  "session_start": "2024-01-15T10:30:00",
  "agent_config": {
    "agent_name": "researcher",
    "model_provider": "anthropic",
    "model_version": "claude-3-5-haiku-latest"
  },
  "conversation": [
    {
      "timestamp": "2024-01-15T10:30:15",
      "role": "user", 
      "content": "What is machine learning?"
    },
    {
      "timestamp": "2024-01-15T10:30:18",
      "role": "assistant",
      "content": "Machine learning is a subset of artificial intelligence..."
    }
  ]
}
```

## 🎨 Terminal Interface

### Visual Features

- **Colored output**: Different colors for user, agent, and system messages
- **Loading indicators**: Shows "🤔 Thinking..." while agent processes
- **Clear formatting**: Organized message display with sender identification
- **Status indicators**: Current agent and model information

### Keyboard Shortcuts

- **Ctrl+C**: Interrupt current response generation
- **Ctrl+D**: Exit chat (same as `/quit`)
- **Up/Down arrows**: Command history (standard terminal behavior)

### Accessibility

- **`--no-colors`**: Disable colored output for accessibility or older terminals
- **Screen reader friendly**: Clean text output without special characters when colors are disabled

## 🔧 Advanced Usage

### Environment Variables

Control behavior with environment variables:

```bash
# Provider-specific API keys
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_CLOUD_PROJECT="your-google-project"

# Agent logging (if supported by agent config)
export DSAT_AGENT_LOGGING_ENABLED="true"
export DSAT_AGENT_LOGGING_MODE="jsonl_file"
export DSAT_AGENT_LOGGING_FILE_PATH="./chat_logs.jsonl"
```

### Multiple Configurations

Organize different configurations for different use cases:

```bash
# Different configs for different projects
dsat chat --config ./research-project/agents.json --agent researcher
dsat chat --config ./creative-project/agents.json --agent storyteller
dsat chat --config ./support-project/agents.json --agent assistant
```

### Batch Testing

Use the chat interface for systematic prompt testing:

1. **Create test agents** with different prompts or parameters
2. **Switch between agents** using `/switch` command  
3. **Test same inputs** against different configurations
4. **Export results** for analysis

## 🐛 Troubleshooting

### Common Issues

#### "No agents available"
- Check that API keys are set in environment variables
- Verify configuration file paths and syntax
- **For Ollama**: Ensure Ollama is running (`ollama serve`) and has models installed (`ollama list`)

#### "Prompt not found" warnings
- Check prompts directory location and structure
- Verify prompt file names match agent configuration
- Use `--prompts-dir` to override default location

#### Connection errors
- **Anthropic**: Verify API key and network connectivity
- **Google**: Check project ID and authentication setup
- **Ollama**: Ensure Ollama service is running and model is pulled

### Debug Mode

Enable detailed logging to troubleshoot issues:

```bash
export DSAT_AGENT_LOGGING_ENABLED="true"
export DSAT_AGENT_LOGGING_LEVEL="standard"
dsat chat --config agents.json --agent my_agent
```

## 📚 Examples

### Basic Usage Examples

```bash
# Quick start with Anthropic
export ANTHROPIC_API_KEY="your-key"
dsat chat

# Interactive Ollama model selection
dsat chat --provider ollama
# Will prompt: "Select a model (1-3): [1] llama3.2 [2] qwen2 [3] mistral"

# Use pirate character from examples
dsat chat --config examples/config/agents.json --agent pirate

# Research assistant with custom prompts
dsat chat --config examples/flexible-prompts/agents-with-custom-prompts.json --agent researcher
```

### Advanced Usage Examples

```bash
# Override prompts for testing
dsat chat --config agents.json --agent assistant --prompts-dir ./test-prompts

# Creative writing with high temperature
dsat chat --provider anthropic --model claude-3-5-haiku-latest
# Then in chat: /switch creative_writer
```

### Integration with Development Workflow

```bash
# Test prompts during development
dsat chat --config ./configs/dev-agents.json --prompts-dir ./prompts-dev

# Production testing
dsat chat --config ./configs/prod-agents.json --prompts-dir ./prompts-prod

# Export test conversations for analysis
# In chat: /export test-session-$(date +%Y%m%d-%H%M).json
```

## 🤝 Contributing

The chat CLI is designed to be extensible:

- **Custom providers**: Create plugins using the entry points system
- **Custom commands**: Extend the command handler system
- **UI improvements**: Enhance the terminal interface
- **Export formats**: Add support for different export formats

See the main [development documentation](README.md#🛠️-development) for setup instructions.

---

The DSAT Chat CLI bridges the gap between agent configuration and interactive testing, making it easy to experiment with different LLM providers, prompts, and configurations in a user-friendly terminal interface.