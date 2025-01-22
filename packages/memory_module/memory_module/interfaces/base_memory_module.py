"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from memory_module.interfaces.types import (
    Memory,
    Message,
    MessageInput,
    Topic,
)


class _CommonBaseMemoryModule(ABC):
    """Common Internal Base class for the memory module interface."""

    # Memory Operations
    @abstractmethod
    async def get_memories(
        self, *, memory_ids: Optional[List[str]] = None, user_id: Optional[str] = None
    ) -> List[Memory]:
        """Get memories based on memory ids or user id. At least one parameter must be provided."""
        pass

    @abstractmethod
    async def remove_memories(
        self, *, user_id: Optional[str] = None, memory_ids: Optional[List[str]] = None
    ) -> None:
        """Remove memories and related messages"""
        pass

    # Message Operations
    @abstractmethod
    async def add_message(self, message: MessageInput) -> Message:
        """Add a message to be processed into memory."""
        pass

    @abstractmethod
    async def get_messages(self, message_ids: List[str]) -> List[Message]:
        """Get messages based on message ids."""
        pass

    @abstractmethod
    async def remove_messages(self, message_ids: List[str]) -> None:
        """Remove messages and related memories"""
        pass


class BaseMemoryModule(_CommonBaseMemoryModule, ABC):
    """Base class for the memory module interface."""

    # Search Operations
    @abstractmethod
    async def search_memories(
        self,
        *,
        user_id: Optional[str],
        query: Optional[str] = None,
        topic: Optional[Topic] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """Retrieve relevant memories based on search criteria.

        Args:
            user_id: Filter memories by specific user ID. If None, search across all users.
            query: Search string to match against memory content. Required if topic is None.
            topic: Filter memories by specific topic. Required if query is None.
            limit: Maximum number of memories to return. If None, returns all matching memories.

            One of query or topic must be provided. If both are provided, they are combined with an AND condition.

        Returns:
            List[Memory]: List of memory objects matching the search criteria, ordered by relevance.
        """
        pass

    # Conversation Operations
    @abstractmethod
    async def retrieve_conversation_history(
        self,
        conversation_ref: str,
        *,
        n_messages: Optional[int] = None,
        last_minutes: Optional[float] = None,
        before: Optional[datetime] = None,
    ) -> List[Message]:
        """Retrieve conversation history based on specified time or message count criteria.

        Args:
            conversation_ref: Unique identifier for the conversation.
            n_messages: Number of most recent messages to retrieve.
            last_minutes: Retrieve messages from the last N minutes.
            before: Retrieve messages before this timestamp.

            Atleast one of the criteria must be provided.

        Returns:
            List[Message]: List of message objects from the conversation history, ordered chronologically.
        """
        pass


class BaseScopedMemoryModule(_CommonBaseMemoryModule, ABC):
    """Base class for the memory module interface that is scoped to a conversation and a list of users"""

    # Properties
    @property
    @abstractmethod
    def conversation_ref(self) -> str: ...

    @property
    @abstractmethod
    def users_in_conversation_scope(self) -> List[str]: ...

    # Conversation Operations
    @abstractmethod
    async def retrieve_conversation_history(
        self,
        *,
        n_messages: Optional[int] = None,
        last_minutes: Optional[float] = None,
        before: Optional[datetime] = None,
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        pass

    # Search Operations
    @abstractmethod
    async def search_memories(
        self,
        *,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        topic: Optional[Topic] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """Retrieve relevant memories based on a query."""
        pass
