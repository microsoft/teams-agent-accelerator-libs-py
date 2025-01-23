"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from teams_memory.interfaces.types import (
    BaseMemoryInput,
    Memory,
    Message,
    MessageInput,
    TextEmbedding,
    Topic,
)


class BaseMemoryStorage(ABC):
    """Base class for the storage component.

    This class defines the interface for persistent storage of memories and messages.
    It handles the low-level storage operations including storing, retrieving, updating,
    and searching memories and messages with their associated embeddings.
    """

    default_limit = 10

    @abstractmethod
    async def store_memory(
        self,
        memory: BaseMemoryInput,
        *,
        embedding_vectors: List[TextEmbedding],
    ) -> str | None:
        """Store a memory with its embedding vectors in the storage system.

        Args:
            memory: The Memory object to store, containing the memory content and metadata
            embedding_vectors: List of TextEmbedding objects containing both vectors
                            and their source text for semantic search

        Returns:
            str | None: The ID of the stored memory if successful, None otherwise
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
        """Update an existing memory with new content and embeddings.

        Args:
            memory_id: ID of the memory to update
            updated_memory: New content for the memory
            embedding_vectors: New embedding vectors for the updated content

        Raises:
            MemoryNotFoundError: If the specified memory_id doesn't exist
        """
        pass

    @abstractmethod
    async def get_memories(
        self, *, memory_ids: Optional[List[str]] = None, user_id: Optional[str] = None
    ) -> List[Memory]:
        """Retrieve memories by IDs or user.

        At least one parameter must be provided.

        Args:
            memory_ids: Optional list of specific memory IDs to retrieve
            user_id: Optional user ID to retrieve all memories for

        Returns:
            List[Memory]: List of memory objects matching the criteria

        Raises:
            ValueError: If neither memory_ids nor user_id is provided
        """
        pass

    @abstractmethod
    async def get_all_memories(
        self, limit: Optional[int] = None, message_ids: Optional[List[str]] = None
    ) -> List[Memory]:
        """Retrieve all memories from storage with optional filtering.

        Args:
            limit: Maximum number of memories to return. Defaults to default_limit if None
            message_ids: Optional list of message IDs to filter memories by source

        Returns:
            List[Memory]: List of memory objects ordered by creation date (newest first)
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
        """Search memories using semantic similarity and/or topics.

        Performs semantic search using embedding vectors and/or filters by topics.
        One of text_embedding or topics must be provided.

        Args:
            user_id: Filter memories by specific user ID. If None, search across all users
            text_embedding: Vector embedding for semantic similarity search
            topics: List of topics to filter memories by
            limit: Maximum number of memories to return. Defaults to default_limit if None

        Returns:
            List[Memory]: List of memories matching the criteria, ordered by relevance

        Raises:
            ValueError: If neither text_embedding nor topics is provided
        """
        pass

    @abstractmethod
    async def delete_memories(
        self, *, user_id: Optional[str] = None, memory_ids: Optional[List[str]] = None
    ) -> None:
        """Remove memories from storage.

        At least one parameter must be provided.

        Args:
            user_id: Optional user ID to remove all memories for
            memory_ids: Optional list of specific memory IDs to remove

        Raises:
            ValueError: If neither memory_ids nor user_id is provided
        """
        pass

    @abstractmethod
    async def upsert_message(self, message: MessageInput) -> Message:
        """Store or update a message in the storage system.

        Args:
            message: The Message object to store or update

        Returns:
            Message: The stored/updated message with assigned ID and metadata
        """
        pass

    @abstractmethod
    async def get_messages(self, message_ids: List[str]) -> List[Message]:
        """Retrieve messages by their IDs.

        Args:
            message_ids: List of message IDs to retrieve

        Returns:
            List[Message]: List of message objects matching the provided IDs

        Raises:
            MessageNotFoundError: If any of the specified message IDs don't exist
        """
        pass

    @abstractmethod
    async def delete_messages(self, message_ids: List[str]) -> None:
        """Remove messages from storage.

        Args:
            message_ids: List of message IDs to remove
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
        """Retrieve conversation history based on specified criteria.

        At least one filtering criteria must be provided.

        Args:
            conversation_ref: Unique identifier for the conversation
            n_messages: Number of most recent messages to retrieve
            last_minutes: Retrieve messages from the last N minutes
            before: Retrieve messages before this timestamp

        Returns:
            List[Message]: List of message objects from the conversation history,
                          ordered chronologically (oldest to newest)

        Raises:
            ValueError: If no filtering criteria is provided
        """
        pass
