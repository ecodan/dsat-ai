# Flexible Prompts Directory Examples

This directory demonstrates different ways to organize and configure prompts directories in DSAT.

## Prompts Directory Search Order

DSAT uses a flexible search strategy to find prompts:

1. **CLI argument** (`--prompts-dir` / `-d`)
2. **Agent config field** (`prompts_dir` in agent configuration)
3. **Config file relative** (`<config_dir>/prompts/`)
4. **Current directory** (`./prompts/`)
5. **User home directory** (`~/.dsat/prompts/`)

## Examples

### 1. CLI Override

```bash
# Use specific prompts directory
dsat chat --config agents.json --agent my_agent --prompts-dir /path/to/my/prompts

# Works with any directory structure
dsat chat --prompts-dir ~/my-project/custom-prompts
```

### 2. Per-Agent Prompts Directory

Create `agents-with-custom-prompts.json`:

```json
{
  "researcher": {
    "model_provider": "anthropic",
    "model_family": "claude", 
    "model_version": "claude-3-5-haiku-latest",
    "prompt": "researcher:v1",
    "prompts_dir": "./research-prompts",
    "provider_auth": {
      "api_key": "sk-..."
    }
  },
  "creative_writer": {
    "model_provider": "anthropic",
    "model_family": "claude",
    "model_version": "claude-3-5-haiku-latest", 
    "prompt": "creative:v1",
    "prompts_dir": "./creative-prompts",
    "provider_auth": {
      "api_key": "sk-..."
    }
  },
  "general_assistant": {
    "model_provider": "anthropic",
    "model_family": "claude",
    "model_version": "claude-3-5-haiku-latest",
    "prompt": "assistant:latest",
    "provider_auth": {
      "api_key": "sk-..."
    }
  }
}
```

Directory structure:
```
project/
├── agents-with-custom-prompts.json
├── research-prompts/
│   └── researcher.toml
├── creative-prompts/
│   └── creative.toml
└── prompts/                    # fallback for general_assistant
    └── assistant.toml
```

### 3. Multiple Project Structure

```
~/projects/
├── research-project/
│   ├── agents.json
│   └── prompts/
│       ├── researcher.toml
│       └── analyst.toml
├── creative-project/
│   ├── agents.json  
│   └── prompts/
│       ├── storyteller.toml
│       └── poet.toml
└── shared-prompts/             # shared across projects
    ├── assistant.toml
    └── helper.toml
```

Use shared prompts:
```bash
cd ~/projects/research-project
dsat chat --config agents.json --prompts-dir ../shared-prompts
```

### 4. Global User Prompts

Store commonly used prompts in `~/.dsat/prompts/`:

```bash
mkdir -p ~/.dsat/prompts
cp assistant.toml ~/.dsat/prompts/
cp helper.toml ~/.dsat/prompts/
```

These will be automatically discovered when no other prompts are found.

## Example Prompt Files

### `research-prompts/researcher.toml`
```toml
v1 = '''You are a thorough research assistant. You excel at finding, analyzing, and synthesizing information from multiple sources. You provide well-structured, evidence-based responses with proper citations when possible. You ask clarifying questions to ensure you understand the research scope and objectives.'''

latest = '''You are a thorough research assistant. You excel at finding, analyzing, and synthesizing information from multiple sources. You provide well-structured, evidence-based responses with proper citations when possible. You ask clarifying questions to ensure you understand the research scope and objectives.'''
```

### `creative-prompts/creative.toml`
```toml
v1 = '''You are a creative writing assistant with expertise in storytelling, character development, and narrative structure. You help writers brainstorm ideas, develop plots, create compelling characters, and refine their prose. You offer constructive feedback and suggestions while respecting the writer's unique voice and vision.'''

latest = '''You are a creative writing assistant with expertise in storytelling, character development, and narrative structure. You help writers brainstorm ideas, develop plots, create compelling characters, and refine their prose. You offer constructive feedback and suggestions while respecting the writer's unique voice and vision.'''
```

## Usage Examples

```bash
# Use config-relative prompts (default behavior)
dsat chat --config examples/config/agents.json --agent pirate

# Override with CLI argument  
dsat chat --config agents.json --agent researcher --prompts-dir ./research-prompts

# Use per-agent prompts directories (configured in agent config)
dsat chat --config agents-with-custom-prompts.json --agent researcher

# Auto-discover from current directory
cd project-with-prompts-dir
dsat chat --agent my_agent

# Fall back to user home directory
dsat chat --agent assistant  # uses ~/.dsat/prompts/assistant.toml
```

## Best Practices

1. **Project Structure**: Keep prompts near config files for project-specific setups
2. **Per-Agent Directories**: Use `prompts_dir` in config for agents requiring different prompt sets
3. **CLI Override**: Use `--prompts-dir` for testing or temporary overrides  
4. **Shared Prompts**: Store commonly used prompts in `~/.dsat/prompts/`
5. **Version Control**: Include prompts in your project's version control

## Error Handling

If prompts aren't found, DSAT will:
- Log a warning about the missing prompt
- Continue with no system prompt (agent uses defaults)
- Still function normally for the conversation

You can check which prompts directory is being used with the debug output.