"""
Abstract interfaces for extensible memory and session management.

This module provides the base classes and interfaces that allow users to extend
dsat's memory management with custom business logic, strategies, and plugins.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from .memory import ConversationMessage


class MemoryEvent(Enum):
    """Events that can trigger memory management hooks."""
    BEFORE_MESSAGE_ADD = "before_message_add"
    AFTER_MESSAGE_ADD = "after_message_add"
    BEFORE_MEMORY_OPERATION = "before_memory_operation"
    AFTER_MEMORY_OPERATION = "after_memory_operation"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_PAUSE = "session_pause"


@dataclass
class MemoryContext:
    """Context information passed to memory hooks and strategies."""
    messages: List[ConversationMessage]
    total_tokens: int
    max_tokens: int
    session_id: str
    agent_name: str
    metadata: Dict[str, Any]


class BaseMemoryStrategy(ABC):
    """Abstract base class for memory management strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy with optional configuration."""
        self.config = config or {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name identifier."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of the strategy."""
        pass
    
    @abstractmethod
    def should_manage_memory(self, context: MemoryContext) -> bool:
        """
        Determine if memory management should be triggered.
        
        :param context: Current memory context
        :return: True if memory management should be performed
        """
        pass
    
    @abstractmethod
    def manage_memory(self, context: MemoryContext) -> List[ConversationMessage]:
        """
        Perform memory management on the message list.
        
        :param context: Current memory context
        :return: Managed list of messages
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate the strategy configuration.
        
        :return: True if configuration is valid
        """
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema for configuration validation.
        
        :return: JSON schema dict
        """
        return {}


class BaseMemoryManager(ABC):
    """Abstract base class for memory managers."""
    
    def __init__(self, max_tokens: int = 8000, storage_dir: Optional[Path] = None, 
                 strategy: Optional[BaseMemoryStrategy] = None):
        """
        Initialize memory manager.
        
        :param max_tokens: Maximum tokens to keep in memory
        :param storage_dir: Directory to store conversation history
        :param strategy: Memory management strategy
        """
        self.max_tokens = max_tokens
        self.storage_dir = storage_dir or (Path.home() / ".dsat" / "chat_history")
        self.strategy = strategy
        self._hooks: Dict[MemoryEvent, List[Callable]] = {event: [] for event in MemoryEvent}
        
        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def calculate_total_tokens(self, messages: List[ConversationMessage]) -> int:
        """Calculate total tokens for a list of messages."""
        pass
    
    @abstractmethod
    def add_message(self, messages: List[ConversationMessage], 
                   role: str, content: str, **kwargs) -> List[ConversationMessage]:
        """
        Add a message to the conversation with memory management.
        
        :param messages: Current message list
        :param role: Message role (user/assistant)
        :param content: Message content
        :param kwargs: Additional message metadata
        :return: Updated message list
        """
        pass
    
    @abstractmethod
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
        pass
    
    def register_hook(self, event: MemoryEvent, callback: Callable) -> None:
        """
        Register a hook for a specific memory event.
        
        :param event: Memory event to hook into
        :param callback: Callback function to execute
        """
        self._hooks[event].append(callback)
    
    def unregister_hook(self, event: MemoryEvent, callback: Callable) -> None:
        """
        Unregister a hook for a specific memory event.
        
        :param event: Memory event to unhook from
        :param callback: Callback function to remove
        """
        if callback in self._hooks[event]:
            self._hooks[event].remove(callback)
    
    def trigger_hooks(self, event: MemoryEvent, context: MemoryContext) -> MemoryContext:
        """
        Trigger all registered hooks for an event.
        
        :param event: Memory event that occurred
        :param context: Memory context to pass to hooks
        :return: Potentially modified context
        """
        for hook in self._hooks[event]:
            try:
                result = hook(context)
                # If hook returns a modified context, use it
                if isinstance(result, MemoryContext):
                    context = result
            except Exception as e:
                # Log hook errors but don't fail memory management
                print(f"Warning: Memory hook error for {event.value}: {e}")
        return context
    
    @abstractmethod
    def save_conversation(self, session_id: str, messages: List[ConversationMessage],
                         agent_config: Optional[Dict[str, Any]] = None) -> None:
        """Save conversation to persistent storage."""
        pass
    
    @abstractmethod
    def load_conversation(self, session_id: str) -> tuple[List[ConversationMessage], Optional[Dict[str, Any]]]:
        """Load conversation from persistent storage."""
        pass
    
    @abstractmethod
    def get_memory_stats(self, messages: List[ConversationMessage]) -> Dict[str, Any]:
        """Get memory usage statistics."""
        pass


class BaseSessionManager(ABC):
    """Abstract base class for session managers."""
    
    def __init__(self, memory_manager: BaseMemoryManager):
        """Initialize session manager with memory manager."""
        self.memory_manager = memory_manager
    
    @abstractmethod
    def create_session(self, agent_name: str, session_id: Optional[str] = None) -> str:
        """
        Create a new chat session.
        
        :param agent_name: Name of the agent
        :param session_id: Optional custom session ID
        :return: Session ID
        """
        pass
    
    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.
        
        :param session_id: Session identifier
        :return: Session data or None if not found
        """
        pass
    
    @abstractmethod
    def pause_session(self, session_id: str) -> None:
        """
        Pause a session (trigger persistence).
        
        :param session_id: Session identifier
        """
        pass
    
    @abstractmethod
    def resume_session(self, session_id: str) -> Optional[List[ConversationMessage]]:
        """
        Resume a session (load from storage).
        
        :param session_id: Session identifier
        :return: Conversation messages or None if not found
        """
        pass
    
    @abstractmethod
    def end_session(self, session_id: str) -> None:
        """
        End a session (final cleanup).
        
        :param session_id: Session identifier
        """
        pass
    
    @abstractmethod
    def list_sessions(self) -> List[str]:
        """List all available sessions."""
        pass


class MemoryPlugin(ABC):
    """Abstract base class for memory management plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name identifier."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable plugin description."""
        pass
    
    @abstractmethod
    def get_strategies(self) -> Dict[str, type[BaseMemoryStrategy]]:
        """
        Return memory strategies provided by this plugin.
        
        :return: Dict mapping strategy names to strategy classes
        """
        pass
    
    @abstractmethod
    def get_memory_manager_class(self) -> Optional[type[BaseMemoryManager]]:
        """
        Return custom memory manager class if provided.
        
        :return: Memory manager class or None
        """
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass