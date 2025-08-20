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

# Enable real-time streaming for instant responses
dsat chat --stream
```

This will auto-detect the available provider and create a default agent.

### With Configuration Files

For more control, use agent configuration files:

```bash
# Use a specific agent from config
dsat chat --config agents.json --agent my_assistant

# Enable streaming with configuration
dsat chat --config agents.json --agent my_assistant --stream

# Override prompts directory
dsat chat --config agents.json --agent researcher --prompts-dir ./my-prompts
```

### Inline Agent Creation

Create agents on the fly without configuration files:

```bash
# Specify provider and model directly
dsat chat --provider anthropic --model claude-3-5-haiku-latest

# Enable streaming with inline creation
dsat chat --provider anthropic --model claude-3-5-haiku-latest --stream

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
  -s, --stream              Enable real-time token streaming
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
| `/stream` | Toggle real-time streaming mode (ON/OFF) |
| `/history` | Display conversation history with memory usage |
| `/clear` | Clear conversation history |
| `/compact` | Compact memory to reduce token usage |
| `/memory` | Show detailed memory usage statistics |
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
    },
    "stream": true
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
    },
    "stream": false
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
    },
    "stream": true
  }
}
```

### Memory Configuration

The chat CLI includes intelligent memory management to handle long conversations efficiently. Memory settings can be configured per agent:

```json
{
  "my_assistant": {
    "model_provider": "anthropic",
    "model_family": "claude",
    "model_version": "claude-3-5-haiku-latest", 
    "prompt": "assistant:v1",
    "provider_auth": {
      "api_key": "your-api-key"
    },
    
    // Memory Configuration (all optional)
    "memory_enabled": true,           // Enable conversation memory (default: true)
    "max_memory_tokens": 8000,        // Token limit for conversations (default: 8000)
    "response_truncate_length": 1000  // Truncate long responses (default: 1000)
  }
}
```

#### Memory Configuration Options

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `memory_enabled` | boolean | `true` | Enable/disable conversation history persistence |
| `max_memory_tokens` | integer | `8000` | Maximum tokens to keep in conversation memory |
| `response_truncate_length` | integer | `1000` | Truncate assistant responses longer than this |

#### Memory Configuration Examples

**High-Memory Agent (Long Research Sessions):**
```json
{
  "research_assistant": {
    "model_provider": "anthropic",
    "model_family": "claude",
    "model_version": "claude-3-5-sonnet-latest",
    "prompt": "research:v1",
    "memory_enabled": true,
    "max_memory_tokens": 16000,      // Higher limit for long conversations
    "response_truncate_length": 2000  // Allow longer detailed responses
  }
}
```

**Low-Memory Agent (Quick Tasks):**
```json
{
  "quick_helper": {
    "model_provider": "anthropic",
    "model_family": "claude",
    "model_version": "claude-3-5-haiku-latest",
    "prompt": "assistant:v1",
    "memory_enabled": true,
    "max_memory_tokens": 2000,       // Lower limit for simple interactions
    "response_truncate_length": 500   // Concise responses
  }
}
```

**Memory-Disabled Agent (Stateless):**
```json
{
  "stateless_agent": {
    "model_provider": "anthropic",
    "model_family": "claude",
    "model_version": "claude-3-5-haiku-latest",
    "prompt": "assistant:v1",
    "memory_enabled": false          // No conversation history maintained
  }
}
```

#### Memory Guidelines

**Token Limits (`max_memory_tokens`):**
- **Small (1000-2000)**: Quick Q&A, simple tasks
- **Medium (4000-8000)**: Normal conversations, default setting  
- **Large (12000-16000)**: Long research sessions, coding assistance
- **Very Large (20000+)**: Extended analysis, document review

**Response Truncation (`response_truncate_length`):**
- **Short (300-500)**: Mobile-friendly, quick responses
- **Medium (800-1000)**: Standard setting, balanced
- **Long (1500-2000)**: Detailed explanations, technical content
- **Very Long (3000+)**: Documentation, extensive analysis

#### Memory Commands

Use these commands during chat to manage memory:

```bash
# Check current memory usage
/memory

# View conversation with memory stats
/history

# Manually reduce memory usage
/compact

# Clear all conversation history
/clear
```

#### Memory Status Display

The chat interface shows memory status with color-coded indicators:

- 🟢 **Green** (0-60%): Normal usage
- 🟡 **Yellow** (60-80%): High usage
- 🔴 **Red** (80%+): Near/at limit, compaction recommended

Example display:
```
🤖 Active Agent: research_assistant (anthropic/claude-3-5-sonnet-latest)
🌊 Streaming: ON
🧠 Memory: 75% used (42 messages)
💡 Type /help for commands, /quit to exit
```

#### Automatic Memory Management

The chat CLI automatically manages memory by:

1. **Token Counting**: Estimates tokens for all messages
2. **Response Truncation**: Cuts long responses with "..." indicator
3. **Smart Compaction**: Removes older messages while preserving recent context
4. **Persistent Storage**: Saves conversations to `~/.dsat/chat_history/`

#### Memory Storage

Conversations are automatically saved as JSON files:

```json
{
  "session_id": "research_assistant_20250120",
  "timestamp": "2025-01-20T10:30:00",
  "agent_config": {
    "agent_name": "research_assistant",
    "model_provider": "anthropic",
    "model_version": "claude-3-5-sonnet-latest"
  },
  "messages": [...],
  "total_tokens": 1234,
  "memory_stats": {
    "total_messages": 42,
    "memory_usage_percent": 75.0
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

## 🌊 Real-time Streaming

The Chat CLI supports real-time token streaming for immediate response feedback, making conversations feel more natural and responsive.

### Enabling Streaming

Stream mode can be enabled in several ways:

**Command Line Flag:**
```bash
# Enable streaming from start
dsat chat --stream

# Works with any configuration method
dsat chat --config agents.json --agent my_assistant --stream
dsat chat --provider anthropic --model claude-3-5-haiku-latest --stream
```

**Agent Configuration:**
```json
{
  "streaming_agent": {
    "model_provider": "anthropic",
    "model_version": "claude-3-5-haiku-latest",
    "prompt": "assistant:v1",
    "stream": true,
    "provider_auth": {
      "api_key": "your-api-key"
    }
  }
}
```

**Interactive Toggle:**
```bash
# Start chat normally
dsat chat

# Toggle streaming during conversation
> /stream
Streaming mode enabled.

> Hello there!
# Response appears token-by-token in real-time

> /stream  
Streaming mode disabled.
```

### Streaming Behavior

When streaming is enabled:

- **Real-time tokens**: Responses appear character-by-character as they're generated
- **Status indicator**: The chat interface shows streaming is ON/OFF in the header
- **Natural feel**: Conversations feel more interactive and immediate
- **Full compatibility**: Works with all providers (Anthropic, Google Vertex AI, Ollama)
- **Error handling**: Gracefully falls back to traditional mode if streaming fails

### Visual Experience

**Traditional mode:**
```
You: Explain quantum computing
🤔 Thinking...
🤖 Assistant: [Complete response appears at once]
```

**Streaming mode:**
```
You: Explain quantum computing
🤔 Thinking...
🤖 Assistant: Quantum computing is a revolutionary...
                ↑ Text appears progressively in real-time
```

### Performance Benefits

- **Faster perceived response**: First tokens arrive immediately
- **Better engagement**: Visual feedback shows the model is working
- **Interrupt capability**: Can see response developing in real-time
- **Memory efficient**: Processes tokens incrementally

Streaming is supported across all DSAT agent providers and integrates seamlessly with the existing chat interface and logging systems.

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