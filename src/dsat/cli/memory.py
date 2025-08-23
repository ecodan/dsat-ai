"""
Memory management utilities for chat CLI sessions.

Provides token counting, conversation persistence, and memory optimization
for chat conversations with LLM agents.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict


@dataclass
class ConversationMessage:
    """A single message in a conversation."""
    role: str  # "user" or "assistant" 
    content: str
    timestamp: str
    tokens: Optional[int] = None  # Token count for this message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMessage":
        """Create from dictionary."""
        return cls(**data)


class TokenCounter:
    """Simple token estimation utility."""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count for text using a simple heuristic.
        
        This is a rough approximation: ~4 characters per token on average.
        For production use, consider using tiktoken or similar libraries.
        
        :param text: Text to count tokens for
        :return: Estimated token count
        """
        if not text:
            return 0
        
        # Simple heuristic: average of 4 characters per token
        # Also count words and punctuation separately
        word_count = len(re.findall(r'\w+', text))
        char_count = len(text)
        
        # Use the higher of word count or char_count/4 for better estimation
        estimated = max(word_count, char_count // 4)
        return max(1, estimated)  # Minimum 1 token
    
    @classmethod
    def count_message_tokens(cls, message: ConversationMessage) -> int:
        """Count tokens in a conversation message."""
        if message.tokens is not None:
            return message.tokens
        
        # Add some overhead for role and formatting
        content_tokens = cls.estimate_tokens(message.content)
        role_tokens = cls.estimate_tokens(message.role)
        
        return content_tokens + role_tokens + 3  # +3 for formatting overhead


class MemoryManager:
    """Manages conversation memory for chat sessions."""
    
    def __init__(self, max_tokens: int = 8000, storage_dir: Optional[Path] = None):
        """
        Initialize memory manager.
        
        :param max_tokens: Maximum tokens to keep in memory
        :param storage_dir: Directory to store conversation history
        """
        self.max_tokens = max_tokens
        self.storage_dir = storage_dir or (Path.home() / ".dsat" / "chat_history")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_total_tokens(self, messages: List[ConversationMessage]) -> int:
        """Calculate total tokens for a list of messages."""
        return sum(TokenCounter.count_message_tokens(msg) for msg in messages)
    
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
    
    def prune_memory(self, messages: List[ConversationMessage], 
                    preserve_recent: int = 5) -> List[ConversationMessage]:
        """
        Prune memory by removing older messages while staying under token limit.
        
        Uses a sliding window approach that preserves:
        - Recent N messages
        - Important context markers (if any)
        
        :param messages: List of conversation messages
        :param preserve_recent: Number of recent messages to always preserve
        :return: Pruned list of messages
        """
        if not messages:
            return messages
        
        total_tokens = self.calculate_total_tokens(messages)
        
        if total_tokens <= self.max_tokens:
            return messages
        
        # Always preserve the most recent messages
        preserved_messages = messages[-preserve_recent:] if preserve_recent > 0 else []
        preserved_tokens = self.calculate_total_tokens(preserved_messages)
        
        if preserved_tokens >= self.max_tokens:
            # Even recent messages exceed limit, truncate them
            result = []
            tokens_used = 0
            
            for msg in reversed(preserved_messages):
                msg_tokens = TokenCounter.count_message_tokens(msg)
                if tokens_used + msg_tokens <= self.max_tokens:
                    result.insert(0, msg)
                    tokens_used += msg_tokens
                else:
                    break
            
            return result
        
        # Try to include older messages that fit
        result = preserved_messages.copy()
        tokens_used = preserved_tokens
        
        # Work backwards from the preserved messages
        candidate_messages = messages[:-preserve_recent] if preserve_recent > 0 else messages
        
        for msg in reversed(candidate_messages):
            msg_tokens = TokenCounter.count_message_tokens(msg)
            if tokens_used + msg_tokens <= self.max_tokens:
                result.insert(0, msg)
                tokens_used += msg_tokens
            else:
                break
        
        return result
    
    def get_session_id(self, agent_config_name: str) -> str:
        """
        Generate a session ID for storing conversation history.
        
        :param agent_config_name: Name of the agent configuration
        :return: Session ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"{agent_config_name}_{timestamp}"
    
    def save_conversation(self, session_id: str, messages: List[ConversationMessage],
                         agent_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Save conversation to persistent storage.
        
        :param session_id: Unique session identifier
        :param messages: List of conversation messages
        :param agent_config: Optional agent configuration metadata
        """
        conversation_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "agent_config": agent_config,
            "messages": [msg.to_dict() for msg in messages],
            "total_tokens": self.calculate_total_tokens(messages)
        }
        
        file_path = self.storage_dir / f"{session_id}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(conversation_data, f, indent=2)
        except Exception as e:
            # Don't crash if we can't save - just log the error
            print(f"Warning: Could not save conversation: {e}")
    
    def load_conversation(self, session_id: str) -> Tuple[List[ConversationMessage], Optional[Dict[str, Any]]]:
        """
        Load conversation from persistent storage.
        
        :param session_id: Session identifier to load
        :return: Tuple of (messages, agent_config)
        """
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
    
    def get_memory_stats(self, messages: List[ConversationMessage]) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        :param messages: List of conversation messages
        :return: Dictionary with memory statistics
        """
        total_tokens = self.calculate_total_tokens(messages)
        total_chars = sum(len(msg.content) for msg in messages)
        
        return {
            "total_messages": len(messages),
            "total_tokens": total_tokens,
            "total_characters": total_chars,
            "max_tokens": self.max_tokens,
            "memory_usage_percent": round((total_tokens / self.max_tokens) * 100, 1),
            "tokens_remaining": max(0, self.max_tokens - total_tokens)
        }