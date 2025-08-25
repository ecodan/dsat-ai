"""
Example demonstrating dsat's extensible memory management system.

This example shows how to:
1. Use different memory strategies
2. Register custom hooks
3. Create custom memory strategies
4. Configure memory strategies via agent configuration
"""

from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Import dsat components
from dsat.agents.agent import Agent, AgentConfig
from dsat.cli.extensible_memory import ExtensibleMemoryManager
from dsat.cli.memory_interfaces import BaseMemoryStrategy, MemoryContext, MemoryEvent
from dsat.cli.memory_registry import register_strategy
from dsat.cli.chat import ChatSession


# Example 1: Custom Memory Strategy
class SemanticMemoryStrategy(BaseMemoryStrategy):
    """
    Custom memory strategy that keeps messages based on semantic similarity.
    
    This is a simplified example - a real implementation would use
    actual embeddings and similarity calculations.
    """
    
    @property
    def name(self) -> str:
        return "semantic"
    
    @property
    def description(self) -> str:
        return "Keep messages based on semantic similarity to recent context"
    
    def should_manage_memory(self, context: MemoryContext) -> bool:
        return context.total_tokens > context.max_tokens
    
    def manage_memory(self, context: MemoryContext) -> List:
        """Keep messages that are semantically similar to recent messages."""
        preserve_recent = self.config.get('preserve_recent', 3)
        similarity_threshold = self.config.get('similarity_threshold', 0.5)
        
        if len(context.messages) <= preserve_recent:
            return context.messages
        
        # Always preserve recent messages
        recent_messages = context.messages[-preserve_recent:]
        older_messages = context.messages[:-preserve_recent]
        
        # For this example, we'll use keyword similarity as a proxy for semantic similarity
        recent_keywords = self._extract_keywords(recent_messages)
        
        # Score older messages by keyword overlap
        scored_messages = []
        for msg in older_messages:
            msg_keywords = self._extract_keywords([msg])
            similarity = self._calculate_similarity(recent_keywords, msg_keywords)
            if similarity >= similarity_threshold:
                scored_messages.append(msg)
        
        # Combine similar messages with recent messages
        return scored_messages + recent_messages
    
    def _extract_keywords(self, messages) -> set:
        """Extract keywords from messages."""
        keywords = set()
        for msg in messages:
            words = msg.content.lower().split()
            # Simple keyword extraction - keep words longer than 3 characters
            keywords.update(word.strip('.,!?') for word in words if len(word) > 3)
        return keywords
    
    def _calculate_similarity(self, keywords1: set, keywords2: set) -> float:
        """Calculate simple Jaccard similarity between keyword sets."""
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_config_schema(self) -> Dict[str, Any]:
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
                    "default": 0.5,
                    "description": "Minimum similarity score to keep older messages"
                }
            }
        }


# Example 2: Custom Business Logic Hook
def analytics_hook(context: MemoryContext) -> MemoryContext:
    """
    Custom hook that logs memory usage analytics.
    
    This could be extended to send metrics to external systems,
    log to databases, etc.
    """
    print(f"[ANALYTICS] Session: {context.session_id}")
    print(f"[ANALYTICS] Messages: {len(context.messages)}")
    print(f"[ANALYTICS] Tokens: {context.total_tokens}")
    print(f"[ANALYTICS] Memory usage: {(context.total_tokens/context.max_tokens)*100:.1f}%")
    
    # Hook could modify context if needed
    context.metadata['analytics_logged'] = True
    
    return context


def content_filter_hook(context: MemoryContext) -> MemoryContext:
    """
    Custom hook that filters sensitive content from messages.
    """
    sensitive_keywords = ['password', 'secret', 'api_key', 'token']
    
    filtered_messages = []
    for msg in context.messages:
        content = msg.content.lower()
        if any(keyword in content for keyword in sensitive_keywords):
            # Create a filtered version of the message
            filtered_msg = msg
            filtered_msg.content = "[FILTERED: Potentially sensitive content removed]"
            filtered_messages.append(filtered_msg)
            print(f"[SECURITY] Filtered sensitive content from message")
        else:
            filtered_messages.append(msg)
    
    context.messages = filtered_messages
    return context


# Example usage functions
def example_1_basic_strategy_usage():
    """Example 1: Using built-in strategies with configuration."""
    print("\n=== Example 1: Basic Strategy Usage ===")
    
    # Agent with compacting memory strategy
    config = AgentConfig(
        agent_name="compacting_agent",
        model_provider="anthropic",
        model_family="claude",
        model_version="claude-3-5-haiku-latest",
        prompt="assistant:latest",
        memory_config={
            "strategy": "compacting",
            "strategy_config": {
                "preserve_recent": 3,
                "compaction_ratio": 0.4,
                "compaction_threshold": 0.7
            }
        }
    )
    
    # Create agent (would need actual API key)
    # agent = Agent.create(config)
    
    print(f"Agent configured with memory strategy: {config.memory_config['strategy']}")
    print(f"Strategy config: {config.memory_config['strategy_config']}")


def example_2_custom_strategy():
    """Example 2: Using a custom memory strategy."""
    print("\n=== Example 2: Custom Memory Strategy ===")
    
    # Register our custom strategy
    register_strategy("semantic", SemanticMemoryStrategy)
    
    # Create memory manager with custom strategy
    memory_manager = ExtensibleMemoryManager(
        max_tokens=1000,
        strategy_name="semantic",
        strategy_config={
            "preserve_recent": 2,
            "similarity_threshold": 0.3
        }
    )
    
    print(f"Created memory manager with custom strategy: {memory_manager.strategy.name}")
    print(f"Strategy description: {memory_manager.strategy.description}")


def example_3_hooks_and_business_logic():
    """Example 3: Adding custom hooks for business logic."""
    print("\n=== Example 3: Custom Hooks ===")
    
    # Create memory manager
    memory_manager = ExtensibleMemoryManager(
        max_tokens=1000,
        strategy_name="pruning"
    )
    
    # Register hooks
    memory_manager.register_hook(MemoryEvent.AFTER_MEMORY_OPERATION, analytics_hook)
    memory_manager.register_hook(MemoryEvent.BEFORE_MESSAGE_ADD, content_filter_hook)
    
    print("Registered analytics and content filter hooks")
    print("These will be triggered during memory operations")


def example_4_agent_configuration():
    """Example 4: Complete agent configuration with memory strategy."""
    print("\n=== Example 4: Complete Agent Configuration ===")
    
    # Example agent configuration JSON that users would create
    agent_config_json = {
        "my_smart_agent": {
            "model_provider": "anthropic",
            "model_family": "claude", 
            "model_version": "claude-3-5-haiku-latest",
            "prompt": "assistant:latest",
            "memory_enabled": True,
            "max_memory_tokens": 4000,
            "memory_config": {
                "strategy": "sliding_window",
                "strategy_config": {
                    "preserve_recent": 4,
                    "important_keywords": ["error", "bug", "fix", "solution", "problem"]
                }
            }
        }
    }
    
    print("Example agent configuration with memory strategy:")
    import json
    print(json.dumps(agent_config_json, indent=2))


if __name__ == "__main__":
    print("DSAT Extensible Memory Management Examples")
    print("=" * 50)
    
    example_1_basic_strategy_usage()
    example_2_custom_strategy()
    example_3_hooks_and_business_logic()
    example_4_agent_configuration()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use these features:")
    print("1. Configure memory strategies in your agent config files")
    print("2. Register custom strategies using register_strategy()")
    print("3. Add hooks using memory_manager.register_hook()")
    print("4. Create custom plugins by implementing MemoryPlugin interface")