from abc import ABC, abstractmethod
from typing import List, Optional

from memory_module.interfaces.types import Memory, Message, ShortTermMemoryRetrievalConfig


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
    async def retrieve(self, query: str, user_id: Optional[str], limit: Optional[int]) -> List[Memory]:
        """Retrieve memories based on a query."""
        pass

    @abstractmethod
    async def update(self, memory_id: str, updateMemory: str) -> None:
        """Update memory with new fact."""
        pass

    @abstractmethod
    async def remove_memories(self, user_id: str) -> None:
        """Remove memories based on user id."""
        pass

    @abstractmethod
    async def add_short_term_memory(self, message: Message) -> None:
        """Add a short-term memory entry."""
        pass

    @abstractmethod
    async def retrieve_short_term_memories(
        self, conversation_ref: str, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        pass
