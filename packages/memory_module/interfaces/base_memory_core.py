from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from memory_module.interfaces.types import (
    Memory,
    Message,
    MessageInput,
    RetrievalConfig,
    ShortTermMemoryRetrievalConfig,
)


class BaseMemoryCore(ABC):
    """Base class for the memory core component."""

    @abstractmethod
    async def process_semantic_messages(
        self,
        messages: List[Message],
        existing_memories: Optional[List[Memory]] = None,
    ) -> None:
        """Process multiple messages into semantic memories (general facts, preferences)."""
        pass

    # TODO: Let's remove it since we're not really using it. [confirmed] we can add it back later.
    @abstractmethod
    async def process_episodic_messages(self, messages: List[Message]) -> None:
        """Process multiple messages into episodic memories (specific events, experiences)."""
        pass

    # TODO: change to search_memories
    @abstractmethod
    async def retrieve_memories(
        self,
        user_id: Optional[str],
        config: RetrievalConfig,
    ) -> List[Memory]:
        """Retrieve memories based on a query."""
        pass

    @abstractmethod
    async def update_memory(self, memory_id: str, updated_memory: str) -> None:
        """Update memory with new fact."""
        pass

    # TODO: Should rename to get_memories_by_id
    # TODO: Merge `get_memories` and `get_memories_from_message` and `get_user_memories`. [confirmed]
    @abstractmethod
    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
        """Get memories based on memory ids."""
        pass

    # TODO: Should rename to get_memories_by_message_id
    @abstractmethod
    async def get_memories_from_message(self, message_id: str) -> List[Memory]:
        """Get memories based on message id."""
        pass

    # TODO: Should rename to get_memories_by_user_id
    @abstractmethod
    async def get_user_memories(self, user_id: str) -> List[Memory]:
        """Get memories based on user id."""
        pass

    @abstractmethod
    async def remove_messages(self, message_ids: List[str]) -> None:
        """Remove messages and related memories."""
        pass

    # TODO: Should rename to remove_memories_by_user_id
    @abstractmethod
    async def remove_memories(self, user_id: str) -> None:
        """Remove memories based on user id."""
        pass

    # TODO: SHould change this to get_messages , by message ids [confirmed]
    @abstractmethod
    async def get_messages(self, memory_ids: List[str]) -> Dict[str, List[Message]]:
        """Get messages based on memory ids."""
        pass

    # TODO: There's 3 words that represent the same thing.
    # 1. Short-term memory, 2. Messages, 3. Chat History.
    # I think we should stick with "Messages" everywhere except for the "retrieve_chat_history" method name. [confirmed]
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
