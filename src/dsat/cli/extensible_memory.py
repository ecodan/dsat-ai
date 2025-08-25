"""
Extensible memory management implementation.

This module provides the new extensible memory manager that implements
the abstract interfaces while maintaining backward compatibility with
existing code.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .memory import ConversationMessage, TokenCounter
from .memory_interfaces import (
    BaseMemoryManager, BaseMemoryStrategy, BaseSessionManager,
    MemoryContext, MemoryEvent
)
from .memory_registry import get_strategy


class ExtensibleMemoryManager(BaseMemoryManager):
    """
    Extensible memory manager that supports pluggable strategies.
    
    This is the new implementation that supports:
    - Pluggable memory strategies
    - Event hooks for custom business logic
    - Full backward compatibility
    """
    
    def __init__(self, max_tokens: int = 8000, storage_dir: Optional[Path] = None,
                 strategy: Optional[BaseMemoryStrategy] = None,
                 strategy_name: str = "pruning",
                 strategy_config: Optional[Dict[str, Any]] = None):
        """
        Initialize extensible memory manager.
        
        :param max_tokens: Maximum tokens to keep in memory
        :param storage_dir: Directory to store conversation history
        :param strategy: Specific strategy instance to use
        :param strategy_name: Name of strategy to load from registry
        :param strategy_config: Configuration for the strategy
        """
        super().__init__(max_tokens, storage_dir, strategy)
        
        # Initialize strategy
        if strategy is None:
            # Load strategy from registry
            self.strategy = get_strategy(strategy_name, strategy_config)
            if self.strategy is None:
                # Fallback to pruning strategy
                from .memory_strategies import PruningMemoryStrategy
                self.strategy = PruningMemoryStrategy(strategy_config)
        else:
            self.strategy = strategy
    
    def calculate_total_tokens(self, messages: List[ConversationMessage]) -> int:
        """Calculate total tokens for a list of messages."""
        return sum(TokenCounter.count_message_tokens(msg) for msg in messages)
    
    def add_message(self, messages: List[ConversationMessage], 
                   role: str, content: str, **kwargs) -> List[ConversationMessage]:
        """
        Add a message to the conversation with memory management.
        
        :param messages: Current message list
        :param role: Message role (user/assistant)
        :param content: Message content
        :param kwargs: Additional message metadata (session_id, agent_name, etc.)
        :return: Updated message list
        """
        session_id = kwargs.get('session_id', 'unknown')
        agent_name = kwargs.get('agent_name', 'unknown')
        metadata = kwargs.get('metadata', {})
        
        # Create context for hooks
        context = MemoryContext(
            messages=messages.copy(),
            total_tokens=self.calculate_total_tokens(messages),
            max_tokens=self.max_tokens,
            session_id=session_id,
            agent_name=agent_name,
            metadata=metadata
        )
        
        # Trigger before_message_add hooks
        context = self.trigger_hooks(MemoryEvent.BEFORE_MESSAGE_ADD, context)
        
        # Create new message
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            tokens=TokenCounter.estimate_tokens(content)
        )
        
        # Add message to context
        context.messages.append(message)
        context.total_tokens += TokenCounter.count_message_tokens(message)
        
        # Trigger after_message_add hooks
        context = self.trigger_hooks(MemoryEvent.AFTER_MESSAGE_ADD, context)
        
        # Perform memory management if needed
        context.messages = self.manage_memory(
            context.messages, session_id, agent_name, metadata
        )
        
        return context.messages
    
    def manage_memory(self, messages: List[ConversationMessage], 
                     session_id: str, agent_name: str,
                     metadata: Optional[Dict[str, Any]] = None) -> List[ConversationMessage]:
        """
        Perform memory management using the configured strategy.
        
        :param messages: Current message list
        :param session_id: Session identifier
        :param agent_name: Agent name
        :param metadata: Additional context metadata
        :return: Managed message list
        """
        if not self.strategy:
            return messages
        
        # Create memory context
        context = MemoryContext(
            messages=messages.copy(),
            total_tokens=self.calculate_total_tokens(messages),
            max_tokens=self.max_tokens,
            session_id=session_id,
            agent_name=agent_name,
            metadata=metadata or {}
        )
        
        # Check if memory management is needed
        if not self.strategy.should_manage_memory(context):
            return messages
        
        # Trigger before_memory_operation hooks
        context = self.trigger_hooks(MemoryEvent.BEFORE_MEMORY_OPERATION, context)
        
        # Perform memory management
        managed_messages = self.strategy.manage_memory(context)
        
        # Update context with managed messages
        context.messages = managed_messages
        context.total_tokens = self.calculate_total_tokens(managed_messages)
        
        # Trigger after_memory_operation hooks
        context = self.trigger_hooks(MemoryEvent.AFTER_MEMORY_OPERATION, context)
        
        return context.messages
    
    def save_conversation(self, session_id: str, messages: List[ConversationMessage],
                         agent_config: Optional[Dict[str, Any]] = None) -> None:
        """Save conversation to persistent storage."""
        conversation_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "agent_config": agent_config,
            "messages": [msg.to_dict() for msg in messages],
            "total_tokens": self.calculate_total_tokens(messages),
            "strategy_info": {
                "name": self.strategy.name if self.strategy else "none",
                "config": self.strategy.config if self.strategy else {}
            }
        }
        
        file_path = self.storage_dir / f"{session_id}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(conversation_data, f, indent=2)
        except Exception as e:
            # Don't crash if we can't save - just log the error
            print(f"Warning: Could not save conversation: {e}")
    
    def load_conversation(self, session_id: str) -> Tuple[List[ConversationMessage], Optional[Dict[str, Any]]]:
        """Load conversation from persistent storage."""
        file_path = self.storage_dir / f"{session_id}.json"
        
        if not file_path.exists():
            return [], None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            messages = [ConversationMessage.from_dict(msg_data) 
                       for msg_data in data.get("messages", [])]
            agent_config = data.get("agent_config")
            
            return messages, agent_config
            
        except Exception as e:
            print(f"Warning: Could not load conversation {session_id}: {e}")
            return [], None
    
    def get_memory_stats(self, messages: List[ConversationMessage]) -> Dict[str, Any]:
        """Get memory usage statistics."""
        total_tokens = self.calculate_total_tokens(messages)
        total_chars = sum(len(msg.content) for msg in messages)
        
        stats = {
            "total_messages": len(messages),
            "total_tokens": total_tokens,
            "total_characters": total_chars,
            "max_tokens": self.max_tokens,
            "memory_usage_percent": round((total_tokens / self.max_tokens) * 100, 1),
            "tokens_remaining": max(0, self.max_tokens - total_tokens),
            "strategy_name": self.strategy.name if self.strategy else "none"
        }
        
        return stats
    
    def get_session_id(self, agent_config_name: str) -> str:
        """
        Generate a session ID for storing conversation history.
        
        :param agent_config_name: Name of the agent configuration
        :return: Session ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"{agent_config_name}_{timestamp}"
    
    def truncate_response(self, content: str, max_length: int) -> str:
        """
        Truncate a response if it's too long.
        
        :param content: Response content to potentially truncate
        :param max_length: Maximum character length
        :return: Truncated content with "..." indicator if truncated
        """
        if len(content) <= max_length:
            return content
        
        # Find a good break point (end of sentence or word)
        truncated = content[:max_length]
        
        # Try to break at sentence end
        last_sentence = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence > max_length * 0.8:  # If we found a good sentence break
            truncated = content[:last_sentence + 1]
        else:
            # Try to break at word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.9:  # If we found a good word break
                truncated = content[:last_space]
        
        return truncated + "..."
    
    def list_sessions(self) -> List[str]:
        """
        List available conversation sessions.
        
        :return: List of session IDs
        """
        try:
            session_files = list(self.storage_dir.glob("*.json"))
            return [f.stem for f in session_files]
        except Exception:
            return []
    
    # Backward compatibility methods
    def prune_memory(self, messages: List[ConversationMessage], 
                    preserve_recent: int = 5) -> List[ConversationMessage]:
        """
        Backward compatibility method for pruning memory.
        
        This method maintains compatibility with existing code that calls
        prune_memory directly. It forces pruning regardless of current memory usage.
        """
        # Use pruning strategy directly and force pruning
        from .memory_strategies import PruningMemoryStrategy
        pruning_strategy = PruningMemoryStrategy({
            'preserve_recent': preserve_recent,
            'force_prune': True  # Force pruning even if under token limit
        })
        
        context = MemoryContext(
            messages=messages,
            total_tokens=self.calculate_total_tokens(messages),
            max_tokens=self.max_tokens,
            session_id="compat",
            agent_name="compat",
            metadata={}
        )
        
        # Force pruning by calling manage_memory directly (bypassing should_manage_memory check)
        return pruning_strategy.manage_memory(context)


class ExtensibleSessionManager(BaseSessionManager):
    """
    Session manager that integrates with extensible memory management.
    """
    
    def __init__(self, memory_manager: ExtensibleMemoryManager):
        """Initialize session manager."""
        super().__init__(memory_manager)
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, agent_name: str, session_id: Optional[str] = None) -> str:
        """Create a new chat session."""
        if session_id is None:
            session_id = self.memory_manager.get_session_id(agent_name)
        
        # Trigger session_start hooks
        context = MemoryContext(
            messages=[],
            total_tokens=0,
            max_tokens=self.memory_manager.max_tokens,
            session_id=session_id,
            agent_name=agent_name,
            metadata={"action": "session_start"}
        )
        self.memory_manager.trigger_hooks(MemoryEvent.SESSION_START, context)
        
        # Register session
        self._active_sessions[session_id] = {
            "agent_name": agent_name,
            "start_time": datetime.now().isoformat(),
            "status": "active"
        }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        return self._active_sessions.get(session_id)
    
    def pause_session(self, session_id: str) -> None:
        """Pause a session."""
        if session_id in self._active_sessions:
            self._active_sessions[session_id]["status"] = "paused"
            
            # Trigger session_pause hooks
            agent_name = self._active_sessions[session_id]["agent_name"]
            context = MemoryContext(
                messages=[],
                total_tokens=0,
                max_tokens=self.memory_manager.max_tokens,
                session_id=session_id,
                agent_name=agent_name,
                metadata={"action": "session_pause"}
            )
            self.memory_manager.trigger_hooks(MemoryEvent.SESSION_PAUSE, context)
    
    def resume_session(self, session_id: str) -> Optional[List[ConversationMessage]]:
        """Resume a session."""
        messages, _ = self.memory_manager.load_conversation(session_id)
        
        if session_id not in self._active_sessions:
            # Create session info if not exists
            self._active_sessions[session_id] = {
                "agent_name": "unknown",
                "start_time": datetime.now().isoformat(),
                "status": "resumed"
            }
        else:
            self._active_sessions[session_id]["status"] = "active"
        
        return messages if messages else None
    
    def end_session(self, session_id: str) -> None:
        """End a session."""
        if session_id in self._active_sessions:
            agent_name = self._active_sessions[session_id]["agent_name"]
            
            # Trigger session_end hooks
            context = MemoryContext(
                messages=[],
                total_tokens=0,
                max_tokens=self.memory_manager.max_tokens,
                session_id=session_id,
                agent_name=agent_name,
                metadata={"action": "session_end"}
            )
            self.memory_manager.trigger_hooks(MemoryEvent.SESSION_END, context)
            
            # Remove session
            del self._active_sessions[session_id]
    
    def list_sessions(self) -> List[str]:
        """List all available sessions."""
        # Combine active sessions and stored sessions
        stored_sessions = self.memory_manager.list_sessions()
        active_sessions = list(self._active_sessions.keys())
        
        return list(set(stored_sessions + active_sessions))