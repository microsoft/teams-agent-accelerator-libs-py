from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from memory_module.interfaces.types import Memory, Message, MessageInput, ShortTermMemoryRetrievalConfig


class BaseMemoryCore(ABC):
    """Base class for the memory core component."""

    @abstractmethod
    async def process_semantic_messages(self, messages: List[Message]) -> None:
        """Process multiple messages into semantic memories (general facts, preferences)."""
        pass

    @abstractmethod
    async def process_episodic_messages(self, messages: List[Message]) -> None:
        """Process multiple messages into episodic memories (specific events, experiences)."""
        pass

    @abstractmethod
    async def retrieve_memories(self, query: str, user_id: Optional[str], limit: Optional[int]) -> List[Memory]:
        """Retrieve memories based on a query."""
        pass

    @abstractmethod
    async def update_memory(self, memory_id: str, updated_memory: str) -> None:
        """Update memory with new fact."""
        pass

    @abstractmethod
    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
        """Get memories based on memory ids."""
        pass

    @abstractmethod
    async def get_user_memories(self, user_id: str) -> List[Memory]:
        """Get memories based on user id."""
        pass

    @abstractmethod
    async def remove_messages(self, message_ids: List[str]) -> None:
        """Remove messages and related memories."""
        pass

    @abstractmethod
    async def remove_memories(self, user_id: str) -> None:
        """Remove memories based on user id."""
        pass

    @abstractmethod
    async def get_messages(self, memory_ids: List[str]) -> Dict[str, List[Message]]:
        """Get messages based on memory ids."""
        pass

    @abstractmethod
    async def add_short_term_memory(self, message: MessageInput) -> Message:
        """Add a short-term memory entry."""
        pass

    @abstractmethod
    async def retrieve_chat_history(
        self, conversation_ref: str, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        pass
