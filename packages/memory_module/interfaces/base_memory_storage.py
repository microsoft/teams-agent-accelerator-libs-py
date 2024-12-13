from abc import ABC, abstractmethod
from typing import List, Optional

from memory_module.interfaces.types import EmbedText, Memory, Message, ShortTermMemoryRetrievalConfig


class BaseMemoryStorage(ABC):
    """Base class for the storage component."""
    default_limit = 3
    @abstractmethod
    async def store_memory(
        self,
        memory: Memory,
        *,
        embedding_vector: List[float],
    ) -> int | None:
        """Store a memory in the storage system.

        Args:
            memory: The Memory object to store
            embedding_vector: Optional embedding vector for the memory
        """
        pass

    @abstractmethod
    async def update_memory(
        self,
        memory_id: str,
        updateMemory: str,
        *,
        embedding_vector:List[float]
    ) -> None:
        """replace an existing memory with new extracted fact and embedding"""
        pass
      
    @abstractmethod
    async def store_short_term_memory(self, message: Message) -> None:
        """Store a short-term memory entry.

        Args:
            message: The Message object representing the short-term memory to store.
        """
        pass

    @abstractmethod
    async def retrieve_memories(
        self, embedText: EmbedText, user_id: Optional[str], limit: Optional[int] = None
    ) -> List[Memory]:
        """Retrieve memories based on a query.

        Args:
            query: The search query string
            user_id: The ID of the user whose memories to retrieve
            limit: Optional maximum number of memories to return

        Returns:
            List of Memory objects matching the query and user_id
        """
        pass

    @abstractmethod
    async def clear_memories(self, user_id: str) -> None:
        """Clear all memories for a given conversation."""
        pass

    @abstractmethod
    async def get_all_memories(self, limit: Optional[int] = None) -> List[Memory]:
        """Retrieve all memories from storage.

        Args:
            limit: Optional maximum number of memories to return

        Returns:
            List of Memory objects ordered by creation date (newest first)
        """
        pass

    @abstractmethod
    async def retrieve_chat_history(
        self, conversation_ref: str, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        pass
