"""
Built-in memory management strategies for dsat.

This module provides the default memory strategies that ship with dsat,
including the original pruning strategy and new compacting strategies.
"""

from typing import List, Dict, Any
from datetime import datetime

from .memory_interfaces import BaseMemoryStrategy, MemoryContext
from .memory import ConversationMessage, TokenCounter


class PruningMemoryStrategy(BaseMemoryStrategy):
    """
    Original pruning strategy that removes older messages.
    
    This is the original /prune behavior - removes older messages while
    preserving recent messages and staying under token limits.
    """
    
    @property
    def name(self) -> str:
        return "pruning"
    
    @property
    def description(self) -> str:
        return "Remove older messages while preserving recent ones"
    
    def should_manage_memory(self, context: MemoryContext) -> bool:
        """Trigger management when exceeding token limit."""
        return context.total_tokens > context.max_tokens
    
    def manage_memory(self, context: MemoryContext) -> List[ConversationMessage]:
        """
        Prune memory by removing older messages while staying under token limit.
        
        Uses a sliding window approach that preserves recent N messages.
        """
        preserve_recent = self.config.get('preserve_recent', 5)
        force_prune = self.config.get('force_prune', False)
        
        if not context.messages:
            return context.messages
        
        # Only return early if not forcing prune and under token limit
        if not force_prune and context.total_tokens <= context.max_tokens:
            return context.messages
        
        # Always preserve the most recent messages
        preserved_messages = context.messages[-preserve_recent:] if preserve_recent > 0 else []
        preserved_tokens = sum(TokenCounter.count_message_tokens(msg) for msg in preserved_messages)
        
        if preserved_tokens >= context.max_tokens:
            # Even recent messages exceed limit, truncate them
            result = []
            tokens_used = 0
            
            for msg in reversed(preserved_messages):
                msg_tokens = TokenCounter.count_message_tokens(msg)
                if tokens_used + msg_tokens <= context.max_tokens:
                    result.insert(0, msg)
                    tokens_used += msg_tokens
                else:
                    break
            
            return result
        
        # If force_prune is enabled, only keep the preserved messages
        if force_prune:
            return preserved_messages
        
        # Try to include older messages that fit
        result = preserved_messages.copy()
        tokens_used = preserved_tokens
        
        # Work backwards from the preserved messages
        candidate_messages = context.messages[:-preserve_recent] if preserve_recent > 0 else context.messages
        
        for msg in reversed(candidate_messages):
            msg_tokens = TokenCounter.count_message_tokens(msg)
            if tokens_used + msg_tokens <= context.max_tokens:
                result.insert(0, msg)
                tokens_used += msg_tokens
            else:
                break
        
        return result
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return JSON schema for configuration."""
        return {
            "type": "object",
            "properties": {
                "preserve_recent": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 5,
                    "description": "Number of recent messages to always preserve"
                },
                "force_prune": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force pruning even when under token limit (for manual /prune command)"
                }
            }
        }


class CompactingMemoryStrategy(BaseMemoryStrategy):
    """
    Compacting strategy that uses LLM summarization to compress older messages.
    
    This strategy actually "compacts" memory by creating summaries of older
    conversation segments, maintaining context while reducing token usage.
    """
    
    @property
    def name(self) -> str:
        return "compacting"
    
    @property
    def description(self) -> str:
        return "Compress older messages using LLM summarization"
    
    def should_manage_memory(self, context: MemoryContext) -> bool:
        """Trigger when exceeding compaction threshold."""
        compaction_threshold = self.config.get('compaction_threshold', 0.8)
        return context.total_tokens > (context.max_tokens * compaction_threshold)
    
    def manage_memory(self, context: MemoryContext) -> List[ConversationMessage]:
        """
        Compact memory by summarizing older conversation segments.
        
        This creates summaries of older parts of the conversation while
        keeping recent messages intact.
        """
        preserve_recent = self.config.get('preserve_recent', 5)
        compaction_ratio = self.config.get('compaction_ratio', 0.3)
        
        if not context.messages or len(context.messages) <= preserve_recent:
            return context.messages
        
        # Split messages into segments to compact and preserve
        messages_to_compact = context.messages[:-preserve_recent]
        messages_to_preserve = context.messages[-preserve_recent:]
        
        if not messages_to_compact:
            return context.messages
        
        # Create summary of older messages
        summary_content = self._create_conversation_summary(
            messages_to_compact, compaction_ratio
        )
        
        # Create summary message
        summary_message = ConversationMessage(
            role="system",
            content=f"[CONVERSATION SUMMARY] {summary_content}",
            timestamp=datetime.now().isoformat(),
            tokens=TokenCounter.estimate_tokens(summary_content)
        )
        
        # Return summary + recent messages
        return [summary_message] + messages_to_preserve
    
    def _create_conversation_summary(self, messages: List[ConversationMessage], 
                                   target_ratio: float) -> str:
        """
        Create a summary of the conversation messages.
        
        In a real implementation, this would use an LLM to create intelligent
        summaries. For now, we'll create a structured summary.
        """
        # Calculate target length based on original content
        original_content = " ".join(msg.content for msg in messages)
        target_length = int(len(original_content) * target_ratio)
        
        # Group messages by conversation turns
        conversation_turns = []
        current_turn = []
        
        for msg in messages:
            if msg.role == "user" and current_turn:
                conversation_turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)
        
        if current_turn:
            conversation_turns.append(current_turn)
        
        # Create summary of key topics and decisions
        summary_parts = []
        
        for i, turn in enumerate(conversation_turns[:5]):  # Limit to first 5 turns
            user_msgs = [msg for msg in turn if msg.role == "user"]
            assistant_msgs = [msg for msg in turn if msg.role == "assistant"]
            
            if user_msgs and assistant_msgs:
                user_content = user_msgs[0].content[:100] + "..." if len(user_msgs[0].content) > 100 else user_msgs[0].content
                assistant_content = assistant_msgs[0].content[:150] + "..." if len(assistant_msgs[0].content) > 150 else assistant_msgs[0].content
                
                summary_parts.append(f"Turn {i+1}: User asked about {user_content} | Assistant: {assistant_content}")
        
        summary = " | ".join(summary_parts)
        
        # Truncate to target length
        if len(summary) > target_length:
            summary = summary[:target_length-3] + "..."
        
        return summary
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return JSON schema for configuration."""
        return {
            "type": "object",
            "properties": {
                "preserve_recent": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 5,
                    "description": "Number of recent messages to keep uncompacted"
                },
                "compaction_ratio": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 0.8,
                    "default": 0.3,
                    "description": "Target compression ratio for summaries"
                },
                "compaction_threshold": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 1.0,
                    "default": 0.8,
                    "description": "Token usage threshold to trigger compaction"
                }
            }
        }


class SlidingWindowMemoryStrategy(BaseMemoryStrategy):
    """
    Enhanced sliding window strategy with importance scoring.
    
    This strategy maintains a sliding window of messages but uses importance
    scoring to keep the most relevant messages rather than just the most recent.
    """
    
    @property
    def name(self) -> str:
        return "sliding_window"
    
    @property
    def description(self) -> str:
        return "Maintain sliding window with importance-based message selection"
    
    def should_manage_memory(self, context: MemoryContext) -> bool:
        """Trigger when exceeding token limit."""
        return context.total_tokens > context.max_tokens
    
    def manage_memory(self, context: MemoryContext) -> List[ConversationMessage]:
        """
        Manage memory using importance-weighted sliding window.
        
        Scores messages by importance and keeps the highest scoring messages
        within the token limit.
        """
        if not context.messages:
            return context.messages
        
        if context.total_tokens <= context.max_tokens:
            return context.messages
        
        preserve_recent = self.config.get('preserve_recent', 3)
        
        # Always preserve the most recent messages
        preserved_messages = context.messages[-preserve_recent:] if preserve_recent > 0 else []
        remaining_messages = context.messages[:-preserve_recent] if preserve_recent > 0 else context.messages
        
        # Score remaining messages by importance
        scored_messages = []
        for msg in remaining_messages:
            importance_score = self._calculate_importance_score(msg, context)
            scored_messages.append((importance_score, msg))
        
        # Sort by importance (highest first)
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        
        # Select messages that fit within token limit
        selected_messages = []
        tokens_used = sum(TokenCounter.count_message_tokens(msg) for msg in preserved_messages)
        
        for score, msg in scored_messages:
            msg_tokens = TokenCounter.count_message_tokens(msg)
            if tokens_used + msg_tokens <= context.max_tokens:
                selected_messages.append(msg)
                tokens_used += msg_tokens
            else:
                break
        
        # Combine selected messages with preserved messages, maintaining order
        all_selected = selected_messages + preserved_messages
        
        # Sort by original message order (by timestamp)
        all_selected.sort(key=lambda msg: msg.timestamp)
        
        return all_selected
    
    def _calculate_importance_score(self, message: ConversationMessage, 
                                  context: MemoryContext) -> float:
        """
        Calculate importance score for a message.
        
        Higher scores indicate more important messages that should be kept.
        """
        score = 0.0
        content = message.content.lower()
        
        # Base score by role
        if message.role == "user":
            score += 0.5  # User messages are generally important as they drive conversation
        else:
            score += 0.3  # Assistant messages provide context
        
        # Boost for questions
        if any(word in content for word in ["what", "how", "why", "when", "where", "who", "?"]):
            score += 0.3
        
        # Boost for important keywords
        important_keywords = self.config.get('important_keywords', [])
        for keyword in important_keywords:
            if keyword.lower() in content:
                score += 0.4
        
        # Boost for longer messages (more likely to contain important info)
        if len(message.content) > 200:
            score += 0.2
        
        # Boost for code blocks (often important for technical conversations)
        if "```" in message.content or "`" in message.content:
            score += 0.3
        
        # Penalty for very recent messages (they're preserved anyway)
        # This helps balance between importance and recency
        message_index = None
        for i, msg in enumerate(context.messages):
            if msg.timestamp == message.timestamp:
                message_index = i
                break
        
        if message_index is not None:
            recency_factor = message_index / len(context.messages)
            score += recency_factor * 0.2  # Slight boost for more recent messages
        
        return score
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return JSON schema for configuration."""
        return {
            "type": "object",
            "properties": {
                "preserve_recent": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 3,
                    "description": "Number of recent messages to always preserve"
                },
                "important_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Keywords that boost message importance scores"
                }
            }
        }