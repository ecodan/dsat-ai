# DSAT Memory Management System

A comprehensive guide to configuring, coding, and running dsat's extensible memory management system.

## Table of Contents

- [Overview](#overview)
- [Built-in Memory Strategies](#built-in-memory-strategies)
- [Configuration Guide](#configuration-guide)
- [Programming Guide](#programming-guide)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

DSAT features an extensible memory management system that allows you to control how chat conversations are stored, managed, and optimized. The system supports:

- **Multiple memory strategies** for different use cases
- **Event-driven hooks** for custom business logic
- **Plugin architecture** for advanced extensions
- **Configuration-driven** strategy selection
- **Full backward compatibility** with existing implementations

### Key Concepts

- **Memory Strategy**: Algorithm that decides how to manage conversation history (pruning, compacting, etc.)
- **Memory Context**: Information passed to strategies and hooks (messages, tokens, metadata)
- **Hooks**: Custom functions that execute on memory events (before/after operations)
- **Plugins**: Packaged extensions that provide new strategies and functionality

## Built-in Memory Strategies

### 1. Pruning Strategy (`pruning`)

**Description**: Removes older messages while preserving recent ones. This is the original `/prune` behavior.

**Best for**: Simple memory management, maintaining recent context only.

**Configuration Options**:
```json
{
  "preserve_recent": 5,     // Number of recent messages to keep (default: 5)
  "force_prune": false      // Force pruning even under token limit (default: false)
}
```

**Behavior**:
- Automatically triggered when exceeding token limit
- Keeps the N most recent messages
- Tries to include older messages if they fit within token limit
- Manual `/prune` forces reduction to recent messages only

### 2. Compacting Strategy (`compacting`)

**Description**: Uses LLM summarization to compress older conversation segments while maintaining context.

**Best for**: Long conversations where context matters, content-heavy discussions.

**Configuration Options**:
```json
{
  "preserve_recent": 5,           // Recent messages to keep uncompacted (default: 5)
  "compaction_ratio": 0.3,        // Target compression ratio (default: 0.3)
  "compaction_threshold": 0.8     // Memory usage % to trigger compaction (default: 0.8)
}
```

**Behavior**:
- Triggers at 80% memory usage (configurable)
- Creates structured summaries of older conversation turns
- Preserves recent messages unchanged
- Replaces older messages with summary message

### 3. Sliding Window Strategy (`sliding_window`)

**Description**: Maintains a sliding window using importance scoring to keep the most relevant messages.

**Best for**: Technical conversations, keyword-focused discussions, maintaining important context.

**Configuration Options**:
```json
{
  "preserve_recent": 3,                           // Always preserved messages (default: 3)
  "important_keywords": ["error", "bug", "fix"]   // Keywords that boost importance (default: [])
}
```

**Behavior**:
- Scores messages by importance (questions, keywords, code, length)
- Preserves recent messages automatically
- Keeps highest-scoring older messages that fit within token limit
- Maintains chronological order in final result

## Configuration Guide

### Agent Configuration

Configure memory strategies in your agent configuration files:

#### Basic Strategy Selection

```json
{
  "my_agent": {
    "model_provider": "anthropic",
    "model_family": "claude",
    "model_version": "claude-3-5-haiku-latest",
    "prompt": "assistant:latest",
    "memory_enabled": true,
    "max_memory_tokens": 8000,
    "memory_config": {
      "strategy": "compacting"
    }
  }
}
```

#### Advanced Configuration with Strategy Options

```json
{
  "technical_assistant": {
    "model_provider": "anthropic",
    "model_family": "claude", 
    "model_version": "claude-3-5-sonnet-latest",
    "prompt": "assistant:latest",
    "memory_enabled": true,
    "max_memory_tokens": 12000,
    "memory_config": {
      "strategy": "sliding_window",
      "strategy_config": {
        "preserve_recent": 4,
        "important_keywords": ["error", "exception", "bug", "fix", "solution", "code", "function"]
      }
    }
  }
}
```

#### Using Model Definitions with Memory Configuration

```json
{
  "_models": {
    "claude_smart": {
      "model_provider": "anthropic",
      "model_family": "claude",
      "model_version": "claude-3-5-sonnet-latest",
      "model_parameters": {"temperature": 0.7}
    }
  },
  "research_agent": {
    "model_id": "claude_smart",
    "prompt": "researcher:v1",
    "max_memory_tokens": 16000,
    "memory_config": {
      "strategy": "compacting",
      "strategy_config": {
        "preserve_recent": 6,
        "compaction_ratio": 0.25,
        "compaction_threshold": 0.75
      }
    }
  }
}
```

### Runtime Configuration

Configure memory strategies programmatically:

```python
from dsat.agents.agent import Agent, AgentConfig
from dsat.cli.chat import ChatSession

# Create agent with memory configuration
config = AgentConfig(
    agent_name="dynamic_agent",
    model_provider="anthropic",
    model_family="claude",
    model_version="claude-3-5-haiku-latest",
    prompt="assistant:latest",
    memory_config={
        "strategy": "pruning",
        "strategy_config": {
            "preserve_recent": 3
        }
    }
)

agent = Agent.create(config)
session = ChatSession(agent)
```

### CLI Commands

Use interactive commands to manage memory:

```bash
# Show memory usage statistics
/memory

# List available strategies and current configuration
/strategies

# Manually prune memory (force reduction to recent messages)
/prune

# Show conversation history with memory stats
/history

# Clear all conversation history
/clear
```

## Programming Guide

### Creating Custom Memory Strategies

Implement the `BaseMemoryStrategy` interface to create custom memory management algorithms:

```python
from dsat.cli.memory_interfaces import BaseMemoryStrategy, MemoryContext
from dsat.cli.memory_registry import register_strategy
from dsat.cli.memory import ConversationMessage

class SemanticMemoryStrategy(BaseMemoryStrategy):
    """Keep messages based on semantic similarity to recent context."""
    
    @property
    def name(self) -> str:
        return "semantic"
    
    @property 
    def description(self) -> str:
        return "Keep messages based on semantic similarity to recent context"
    
    def should_manage_memory(self, context: MemoryContext) -> bool:
        """Trigger when exceeding token limit."""
        return context.total_tokens > context.max_tokens
    
    def manage_memory(self, context: MemoryContext) -> List[ConversationMessage]:
        """Perform semantic-based memory management."""
        preserve_recent = self.config.get('preserve_recent', 3)
        similarity_threshold = self.config.get('similarity_threshold', 0.7)
        
        if len(context.messages) <= preserve_recent:
            return context.messages
        
        # Always preserve recent messages
        recent_messages = context.messages[-preserve_recent:]
        older_messages = context.messages[:-preserve_recent]
        
        # Calculate semantic similarity (simplified example)
        similar_messages = []
        for msg in older_messages:
            if self._calculate_similarity(msg, recent_messages) >= similarity_threshold:
                similar_messages.append(msg)
        
        return similar_messages + recent_messages
    
    def _calculate_similarity(self, message: ConversationMessage, 
                            recent_messages: List[ConversationMessage]) -> float:
        """Calculate semantic similarity (simplified implementation)."""
        # In a real implementation, you would use embeddings
        # This is a simplified keyword-based approach
        msg_words = set(message.content.lower().split())
        recent_words = set()
        for recent_msg in recent_messages:
            recent_words.update(recent_msg.content.lower().split())
        
        if not msg_words or not recent_words:
            return 0.0
        
        intersection = len(msg_words.intersection(recent_words))
        union = len(msg_words.union(recent_words))
        
        return intersection / union if union > 0 else 0.0
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema."""
        return {
            "type": "object",
            "properties": {
                "preserve_recent": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 3,
                    "description": "Number of recent messages to always preserve"
                },
                "similarity_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7,
                    "description": "Minimum similarity score to keep older messages"
                }
            }
        }

# Register the custom strategy
register_strategy("semantic", SemanticMemoryStrategy)
```

### Using Hooks for Business Logic

Add custom hooks to execute business logic during memory operations:

```python
from dsat.cli.memory_interfaces import MemoryContext, MemoryEvent
from dsat.cli.extensible_memory import ExtensibleMemoryManager

def analytics_hook(context: MemoryContext) -> MemoryContext:
    """Log memory usage analytics."""
    print(f"[ANALYTICS] Session: {context.session_id}")
    print(f"[ANALYTICS] Messages: {len(context.messages)}")
    print(f"[ANALYTICS] Tokens: {context.total_tokens}")
    print(f"[ANALYTICS] Usage: {(context.total_tokens/context.max_tokens)*100:.1f}%")
    
    # Send to external analytics service
    # analytics_service.track_memory_usage({
    #     'session_id': context.session_id,
    #     'message_count': len(context.messages),
    #     'token_count': context.total_tokens,
    #     'usage_percent': (context.total_tokens/context.max_tokens)*100
    # })
    
    return context

def content_filter_hook(context: MemoryContext) -> MemoryContext:
    """Filter sensitive content from messages."""
    sensitive_patterns = ['password', 'api_key', 'secret', 'token']
    
    filtered_messages = []
    for msg in context.messages:
        content = msg.content.lower()
        if any(pattern in content for pattern in sensitive_patterns):
            # Create filtered version
            filtered_msg = ConversationMessage(
                role=msg.role,
                content="[FILTERED: Potentially sensitive content removed]",
                timestamp=msg.timestamp,
                tokens=msg.tokens
            )
            filtered_messages.append(filtered_msg)
            print(f"[SECURITY] Filtered sensitive content from message")
        else:
            filtered_messages.append(msg)
    
    context.messages = filtered_messages
    return context

# Register hooks with memory manager
memory_manager = ExtensibleMemoryManager()
memory_manager.register_hook(MemoryEvent.AFTER_MEMORY_OPERATION, analytics_hook)
memory_manager.register_hook(MemoryEvent.BEFORE_MESSAGE_ADD, content_filter_hook)
```

### Building Memory Plugins

Create comprehensive plugins that package strategies, managers, and hooks:

```python
from dsat.cli.memory_interfaces import MemoryPlugin, BaseMemoryStrategy, BaseMemoryManager
from typing import Dict, Optional, Type, Any

class BusinessLogicMemoryPlugin(MemoryPlugin):
    """Plugin providing business-focused memory strategies and analytics."""
    
    @property
    def name(self) -> str:
        return "business_logic"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Business-focused memory strategies with analytics and compliance features"
    
    def get_strategies(self) -> Dict[str, Type[BaseMemoryStrategy]]:
        """Return strategies provided by this plugin."""
        return {
            "semantic": SemanticMemoryStrategy,
            "compliance": ComplianceMemoryStrategy,  # Another custom strategy
            "priority_based": PriorityMemoryStrategy  # Another custom strategy
        }
    
    def get_memory_manager_class(self) -> Optional[Type[BaseMemoryManager]]:
        """Return custom memory manager if provided."""
        return BusinessMemoryManager  # Custom manager with additional features
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        self.config = config
        print(f"Initialized {self.name} plugin v{self.version}")
        
        # Set up any global configuration
        if config.get('enable_analytics', False):
            self._setup_analytics()
        
        if config.get('compliance_mode', False):
            self._setup_compliance_monitoring()
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        print(f"Cleaning up {self.name} plugin")
        # Cleanup any resources, connections, etc.

# Plugin entry point setup (in setup.py or pyproject.toml)
# [project.entry-points."dsat.memory_plugins"]
# business_logic = "my_package.plugins:BusinessLogicMemoryPlugin"
```

## Examples

### Example 1: Basic Strategy Configuration

Configure different strategies for different agents:

```json
{
  "chat_agent": {
    "model_provider": "anthropic",
    "model_family": "claude",
    "model_version": "claude-3-5-haiku-latest",
    "prompt": "assistant:latest", 
    "memory_config": {
      "strategy": "pruning",
      "strategy_config": {
        "preserve_recent": 5
      }
    }
  },
  "research_agent": {
    "model_provider": "anthropic",
    "model_family": "claude",
    "model_version": "claude-3-5-sonnet-latest",
    "prompt": "researcher:v1",
    "max_memory_tokens": 16000,
    "memory_config": {
      "strategy": "compacting",
      "strategy_config": {
        "preserve_recent": 8,
        "compaction_ratio": 0.2,
        "compaction_threshold": 0.8
      }
    }
  },
  "technical_agent": {
    "model_provider": "anthropic", 
    "model_family": "claude",
    "model_version": "claude-3-5-sonnet-latest",
    "prompt": "technical:v1",
    "memory_config": {
      "strategy": "sliding_window",
      "strategy_config": {
        "preserve_recent": 4,
        "important_keywords": ["error", "exception", "bug", "fix", "code", "function", "class", "method"]
      }
    }
  }
}
```

### Example 2: Programmatic Configuration

Set up memory management programmatically:

```python
from dsat.agents.agent import Agent, AgentConfig
from dsat.cli.chat import ChatSession
from dsat.cli.extensible_memory import ExtensibleMemoryManager
from dsat.cli.memory_interfaces import MemoryEvent

# Create agent with custom memory setup
config = AgentConfig(
    agent_name="analytics_agent",
    model_provider="anthropic",
    model_family="claude", 
    model_version="claude-3-5-haiku-latest",
    prompt="assistant:latest",
    max_memory_tokens=10000
)

agent = Agent.create(config)

# Create custom memory manager with hooks
memory_manager = ExtensibleMemoryManager(
    max_tokens=10000,
    strategy_name="sliding_window",
    strategy_config={
        "preserve_recent": 4,
        "important_keywords": ["analytics", "data", "metric", "report"]
    }
)

# Add business logic hooks
memory_manager.register_hook(MemoryEvent.AFTER_MEMORY_OPERATION, analytics_hook)
memory_manager.register_hook(MemoryEvent.BEFORE_MESSAGE_ADD, content_filter_hook)

# Create chat session with custom memory manager
session = ChatSession(agent, memory_manager=memory_manager)
```

### Example 3: Custom Strategy with Registration

Create and use a custom memory strategy:

```python
from dsat.cli.memory_interfaces import BaseMemoryStrategy, MemoryContext
from dsat.cli.memory_registry import register_strategy
from dsat.cli.memory import TokenCounter
import re

class CodeFocusedMemoryStrategy(BaseMemoryStrategy):
    """Memory strategy optimized for coding conversations."""
    
    @property
    def name(self) -> str:
        return "code_focused"
    
    @property
    def description(self) -> str:
        return "Prioritize messages containing code blocks and technical discussions"
    
    def should_manage_memory(self, context: MemoryContext) -> bool:
        return context.total_tokens > context.max_tokens
    
    def manage_memory(self, context: MemoryContext) -> List[ConversationMessage]:
        preserve_recent = self.config.get('preserve_recent', 3)
        
        if len(context.messages) <= preserve_recent:
            return context.messages
        
        # Always preserve recent messages
        recent_messages = context.messages[-preserve_recent:]
        older_messages = context.messages[:-preserve_recent]
        
        # Score messages by code content
        scored_messages = []
        for msg in older_messages:
            score = self._calculate_code_importance(msg)
            scored_messages.append((score, msg))
        
        # Sort by importance and select messages that fit
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        
        selected_messages = []
        tokens_used = sum(TokenCounter.count_message_tokens(msg) for msg in recent_messages)
        
        for score, msg in scored_messages:
            msg_tokens = TokenCounter.count_message_tokens(msg)
            if tokens_used + msg_tokens <= context.max_tokens:
                selected_messages.append(msg)
                tokens_used += msg_tokens
        
        # Maintain chronological order
        all_messages = selected_messages + recent_messages
        all_messages.sort(key=lambda msg: msg.timestamp)
        
        return all_messages
    
    def _calculate_code_importance(self, message: ConversationMessage) -> float:
        """Score messages based on code content and technical relevance."""
        content = message.content
        score = 0.0
        
        # High score for code blocks
        if '```' in content:
            score += 2.0
            # Extra points for specific languages
            if any(lang in content.lower() for lang in ['python', 'javascript', 'java', 'cpp', 'rust']):
                score += 0.5
        
        # Score for inline code
        score += len(re.findall(r'`[^`]+`', content)) * 0.2
        
        # Score for technical keywords
        technical_keywords = ['function', 'class', 'method', 'variable', 'error', 'exception', 
                            'import', 'library', 'framework', 'api', 'database', 'algorithm']
        for keyword in technical_keywords:
            if keyword.lower() in content.lower():
                score += 0.1
        
        # Score for questions (usually important in coding discussions)
        if '?' in content or any(word in content.lower() for word in ['how', 'why', 'what', 'when', 'where']):
            score += 0.3
        
        return score
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "preserve_recent": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 3,
                    "description": "Number of recent messages to always preserve"
                }
            }
        }

# Register the strategy
register_strategy("code_focused", CodeFocusedMemoryStrategy)

# Use in agent configuration
coding_agent_config = {
    "coding_assistant": {
        "model_provider": "anthropic",
        "model_family": "claude",
        "model_version": "claude-3-5-sonnet-latest",
        "prompt": "coding_assistant:v1",
        "memory_config": {
            "strategy": "code_focused",
            "strategy_config": {
                "preserve_recent": 4
            }
        }
    }
}
```

### Example 4: CLI Usage Workflow

Interactive memory management workflow:

```bash
# Start chat with specific agent
dsat chat --agent technical_agent

# During conversation, check memory usage
You: /memory
Memory Statistics:
  Total messages: 15
  Total tokens: 3420
  Memory usage: 42.8%
  Strategy: sliding_window

# View available strategies
You: /strategies
Available Memory Strategies:
  pruning - Remove older messages while preserving recent ones
  compacting - Compress older messages using LLM summarization (current)
  sliding_window - Maintain sliding window with importance-based message selection

Current Strategy Configuration:
  Strategy: sliding_window
  Description: Maintain sliding window with importance-based message selection
  Configuration:
    preserve_recent: 4
    important_keywords: ["error", "exception", "bug", "fix", "code"]

# Manually prune memory if needed
You: /prune
Memory pruned successfully!
Messages: 15 → 4
Tokens: 3420 → 890
Memory usage: 42.8% → 11.1%

# View conversation history with stats
You: /history
Conversation History (4 messages, 890 tokens, 11.1% memory used):
USER: How do I fix this Python error?...
ASSISTANT: This error occurs because...
USER: Thanks! Can you show me the corrected code?...  
ASSISTANT: Here's the corrected version...
```

## API Reference

### Core Interfaces

#### BaseMemoryStrategy

Abstract base class for memory management strategies.

```python
class BaseMemoryStrategy(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    
    @property
    @abstractmethod
    def name(self) -> str
    
    @property  
    @abstractmethod
    def description(self) -> str
    
    @abstractmethod
    def should_manage_memory(self, context: MemoryContext) -> bool
    
    @abstractmethod
    def manage_memory(self, context: MemoryContext) -> List[ConversationMessage]
    
    def validate_config(self) -> bool
    def get_config_schema(self) -> Dict[str, Any]
```

#### BaseMemoryManager

Abstract base class for memory managers.

```python
class BaseMemoryManager(ABC):
    def __init__(self, max_tokens: int = 8000, storage_dir: Optional[Path] = None,
                 strategy: Optional[BaseMemoryStrategy] = None)
    
    @abstractmethod
    def calculate_total_tokens(self, messages: List[ConversationMessage]) -> int
    
    @abstractmethod
    def add_message(self, messages: List[ConversationMessage], 
                   role: str, content: str, **kwargs) -> List[ConversationMessage]
    
    @abstractmethod
    def manage_memory(self, messages: List[ConversationMessage], 
                     session_id: str, agent_name: str,
                     metadata: Optional[Dict[str, Any]] = None) -> List[ConversationMessage]
    
    def register_hook(self, event: MemoryEvent, callback: Callable) -> None
    def unregister_hook(self, event: MemoryEvent, callback: Callable) -> None
    def trigger_hooks(self, event: MemoryEvent, context: MemoryContext) -> MemoryContext
```

#### MemoryPlugin

Abstract base class for memory plugins.

```python
class MemoryPlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str
    
    @property
    @abstractmethod
    def version(self) -> str
    
    @property
    @abstractmethod
    def description(self) -> str
    
    @abstractmethod
    def get_strategies(self) -> Dict[str, Type[BaseMemoryStrategy]]
    
    @abstractmethod
    def get_memory_manager_class(self) -> Optional[Type[BaseMemoryManager]]
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None
    
    @abstractmethod
    def cleanup(self) -> None
```

### Context Objects

#### MemoryContext

Information passed to strategies and hooks:

```python
@dataclass
class MemoryContext:
    messages: List[ConversationMessage]      # Current conversation messages
    total_tokens: int                        # Total token count
    max_tokens: int                          # Maximum allowed tokens
    session_id: str                          # Session identifier
    agent_name: str                          # Agent name
    metadata: Dict[str, Any]                 # Additional context metadata
```

#### ConversationMessage  

Individual message in conversation:

```python
@dataclass
class ConversationMessage:
    role: str                                # "user" or "assistant"
    content: str                             # Message content
    timestamp: str                           # ISO timestamp
    tokens: Optional[int] = None             # Token count for message
```

### Event System

#### MemoryEvent

Events that trigger hooks:

```python
class MemoryEvent(Enum):
    BEFORE_MESSAGE_ADD = "before_message_add"
    AFTER_MESSAGE_ADD = "after_message_add"
    BEFORE_MEMORY_OPERATION = "before_memory_operation"
    AFTER_MEMORY_OPERATION = "after_memory_operation"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_PAUSE = "session_pause"
```

### Registry Functions

Global functions for strategy and plugin management:

```python
# Strategy registration
def register_strategy(name: str, strategy_class: Type[BaseMemoryStrategy]) -> None
def get_strategy(name: str, config: Optional[Dict[str, Any]] = None) -> Optional[BaseMemoryStrategy]
def list_available_strategies() -> List[str]

# Memory manager registration  
def register_memory_manager(name: str, manager_class: Type[BaseMemoryManager]) -> None
def get_memory_manager(name: str, **kwargs) -> Optional[BaseMemoryManager]

# Plugin registration
def register_plugin(plugin: MemoryPlugin) -> None

# Registry access
def get_registry() -> MemoryStrategyRegistry
```

## Best Practices

### Strategy Selection Guidelines

- **Use Pruning** for simple chat scenarios where only recent context matters
- **Use Compacting** for long research or content-heavy conversations where context is important
- **Use Sliding Window** for technical discussions where specific keywords/topics matter
- **Create Custom Strategies** for domain-specific requirements (semantic similarity, business rules, etc.)

### Performance Considerations

- **Token Estimation**: Built-in token counting is approximate (4 chars/token). For production use, consider integrating with tiktoken or similar libraries
- **Strategy Complexity**: More sophisticated strategies (semantic, LLM-based) have higher computational costs
- **Hook Overhead**: Too many hooks can impact performance; use judiciously
- **Memory Limits**: Set appropriate `max_memory_tokens` based on your model's context window and performance requirements

### Memory Usage Optimization

```python
# Optimal configuration for different use cases

# Casual chat - prioritize simplicity and speed
casual_config = {
    "strategy": "pruning",
    "strategy_config": {"preserve_recent": 5}
}

# Research/analysis - maintain context, compress when needed  
research_config = {
    "strategy": "compacting", 
    "strategy_config": {
        "preserve_recent": 8,
        "compaction_ratio": 0.3,
        "compaction_threshold": 0.8
    }
}

# Technical support - keep important technical content
technical_config = {
    "strategy": "sliding_window",
    "strategy_config": {
        "preserve_recent": 4,
        "important_keywords": ["error", "exception", "fix", "solution", "code"]
    }
}
```

### Error Handling and Debugging

```python
# Robust hook implementation with error handling
def robust_analytics_hook(context: MemoryContext) -> MemoryContext:
    try:
        # Analytics logic here
        analytics_service.track_usage(context)
    except Exception as e:
        # Log error but don't fail memory management
        logger.error(f"Analytics hook failed: {e}")
        # Optionally add to context metadata for debugging
        context.metadata['hook_errors'] = context.metadata.get('hook_errors', [])
        context.metadata['hook_errors'].append(f"analytics: {str(e)}")
    
    return context

# Strategy with validation
class ValidatedStrategy(BaseMemoryStrategy):
    def manage_memory(self, context: MemoryContext) -> List[ConversationMessage]:
        # Validate inputs
        if not context.messages:
            return context.messages
        
        if context.total_tokens < 0:
            raise ValueError("Invalid token count")
        
        # Perform memory management
        result = self._do_memory_management(context)
        
        # Validate outputs
        if len(result) > len(context.messages):
            raise ValueError("Strategy cannot increase message count")
        
        return result
```

### Testing Custom Implementations

```python
import pytest
from dsat.cli.memory_interfaces import MemoryContext
from dsat.cli.memory import ConversationMessage

def test_custom_strategy():
    """Test custom memory strategy."""
    strategy = MyCustomStrategy({"preserve_recent": 3})
    
    # Create test messages
    messages = [
        ConversationMessage("user", f"Message {i}", "2023-01-01T00:0{i}:00", 10)
        for i in range(10)
    ]
    
    # Create test context
    context = MemoryContext(
        messages=messages,
        total_tokens=100,
        max_tokens=50,
        session_id="test",
        agent_name="test",
        metadata={}
    )
    
    # Test strategy behavior
    result = strategy.manage_memory(context)
    
    # Validate results
    assert len(result) <= len(messages)
    assert len(result) >= 3  # Should preserve recent messages
    assert result[-3:] == messages[-3:]  # Recent messages preserved

def test_hook_integration():
    """Test hook registration and execution."""
    memory_manager = ExtensibleMemoryManager()
    
    call_count = 0
    def test_hook(context):
        nonlocal call_count
        call_count += 1
        return context
    
    memory_manager.register_hook(MemoryEvent.AFTER_MEMORY_OPERATION, test_hook)
    
    # Trigger memory operation
    context = MemoryContext([], 0, 1000, "test", "test", {})
    memory_manager.trigger_hooks(MemoryEvent.AFTER_MEMORY_OPERATION, context)
    
    assert call_count == 1
```

## Troubleshooting

### Common Issues

#### Strategy Not Working as Expected

**Problem**: Memory strategy doesn't seem to be pruning/compacting messages.

**Solutions**:
1. Check if memory usage exceeds the threshold:
   ```bash
   /memory  # Check current usage percentage
   ```

2. For manual pruning, ensure you're using the `/prune` command:
   ```bash
   /prune   # Forces pruning regardless of usage
   ```

3. Verify strategy configuration:
   ```bash
   /strategies  # Shows current strategy and config
   ```

4. Check agent configuration has correct strategy name:
   ```json
   {
     "memory_config": {
       "strategy": "pruning",  // Ensure this matches available strategies
       "strategy_config": {
         "preserve_recent": 5
       }
     }
   }
   ```

#### Plugin Loading Issues

**Problem**: Custom plugins not being discovered.

**Solutions**:
1. Verify entry point configuration in `setup.py` or `pyproject.toml`:
   ```toml
   [project.entry-points."dsat.memory_plugins"]
   my_plugin = "my_package.plugins:MyMemoryPlugin"
   ```

2. Ensure plugin class implements all required methods:
   ```python
   class MyPlugin(MemoryPlugin):
       # Must implement all abstract methods
       @property
       def name(self) -> str: ...
       # etc.
   ```

3. Check plugin initialization doesn't raise exceptions:
   ```python
   def initialize(self, config: Dict[str, Any]) -> None:
       try:
           # Plugin initialization logic
           pass
       except Exception as e:
           print(f"Plugin init error: {e}")
           raise
   ```

#### Hook Registration Problems

**Problem**: Hooks not being called during memory operations.

**Solutions**:
1. Ensure hooks are registered on the correct memory manager instance:
   ```python
   # Get the actual memory manager from chat session
   session = ChatSession(agent)
   session.memory_manager.register_hook(event, hook_function)
   ```

2. Verify hook function signature:
   ```python
   def correct_hook(context: MemoryContext) -> MemoryContext:
       # Process context
       return context  # Must return context
   ```

3. Check for exceptions in hook functions:
   ```python
   def safe_hook(context: MemoryContext) -> MemoryContext:
       try:
           # Hook logic
           pass
       except Exception as e:
           print(f"Hook error: {e}")  # Will be caught by hook system
       return context
   ```

#### Configuration Errors

**Problem**: Agent fails to start with memory configuration errors.

**Solutions**:
1. Validate JSON syntax in configuration files
2. Ensure strategy names match available strategies:
   ```python
   from dsat.cli.memory_registry import list_available_strategies
   print(list_available_strategies())  # Check available strategy names
   ```

3. Verify strategy-specific configuration options:
   ```python
   from dsat.cli.memory_registry import get_registry
   registry = get_registry()
   info = registry.get_strategy_info("pruning")
   print(info["config_schema"])  # Shows valid config options
   ```

### Debug Mode

Enable detailed logging to troubleshoot memory issues:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("dsat.cli.memory")
logger.setLevel(logging.DEBUG)

# This will show detailed memory operations
```

### Getting Help

1. **Check Configuration**: Use `/strategies` command to verify current setup
2. **Test with Built-ins**: Start with built-in strategies before custom implementations  
3. **Incremental Development**: Build custom strategies step-by-step, testing each component
4. **Community Support**: Check dsat documentation and community resources for additional examples

---

This comprehensive guide covers all aspects of dsat's memory management system. For additional examples and advanced use cases, refer to the `examples/` directory in the dsat repository.