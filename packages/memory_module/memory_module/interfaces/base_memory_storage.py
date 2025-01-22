from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from memory_module.interfaces.types import (
    BaseMemoryInput,
    Memory,
    Message,
    MessageInput,
    TextEmbedding,
    Topic,
)


class BaseMemoryStorage(ABC):
    """Base class for the storage component."""

    default_limit = 10

    @abstractmethod
    async def store_memory(
        self,
        memory: BaseMemoryInput,
        *,
        embedding_vectors: List[TextEmbedding],
    ) -> str | None:
        """Store a memory in the storage system.

        Args:
            memory: The Memory object to store
            embedding_vectors: List of TextEmbedding objects containing both vectors and their source text
        """
        pass

    @abstractmethod
    async def update_memory(
        self,
        memory_id: str,
        updated_memory: str,
        *,
        embedding_vectors: List[TextEmbedding],
    ) -> None:
        """replace an existing memory with new extracted fact and embedding"""
        pass

    @abstractmethod
    async def upsert_message(self, message: MessageInput) -> Message:
        """Upsert a message into the storage system.

        Args:
            message: The Message object representing the short-term memory to store.
        """
        pass

    @abstractmethod
    async def search_memories(
        self,
        *,
        user_id: Optional[str],
        text_embedding: Optional[TextEmbedding] = None,
        topics: Optional[List[Topic]] = None,
        limit: Optional[int] = None,
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
    async def get_memories(
        self, *, memory_ids: Optional[List[str]] = None, user_id: Optional[str] = None
    ) -> List[Memory]:
        """Get memories based on memory ids or user id."""
        pass

    @abstractmethod
    async def get_messages(self, message_ids: List[str]) -> List[Message]:
        """Get messages based on message ids."""
        pass

    @abstractmethod
    async def remove_messages(self, message_ids: List[str]) -> None:
        pass

    @abstractmethod
    async def remove_memories(self, memory_ids: List[str]) -> None:
        pass

    @abstractmethod
    async def get_all_memories(
        self, limit: Optional[int] = None, message_ids: Optional[List[str]] = None
    ) -> List[Memory]:
        """Retrieve all memories from storage.

        Args:
            limit: Optional maximum number of memories to return
            message_ids: Optional list of message_id to filter memories

        Returns:
            List of Memory objects ordered by creation date (newest first)
        """
        pass

    @abstractmethod
    async def retrieve_conversation_history(
        self,
        conversation_ref: str,
        *,
        n_messages: Optional[int] = None,
        last_minutes: Optional[float] = None,
        before: Optional[datetime] = None,
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        pass
