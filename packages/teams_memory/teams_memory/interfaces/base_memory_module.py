"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from teams_memory.interfaces.types import (
    Memory,
    Message,
    MessageInput,
)


class _CommonBaseMemoryModule(ABC):
    """Common Internal Base class for the memory module interface.

    This class provides the core functionality shared between different memory module implementations.
    It handles basic memory and message operations that are common across all memory module types.
    """

    @abstractmethod
    async def get_memories(
        self, *, memory_ids: Optional[List[str]] = None, user_id: Optional[str] = None
    ) -> List[Memory]:
        """Retrieve memories based on memory IDs or user ID.

        At least one parameter must be provided to filter the memories.

        Args:
            memory_ids (Optional[List[str]]): Optional list of specific memory IDs to retrieve.
            user_id (Optional[str]): Optional user ID to retrieve all memories for a specific user.

        Returns:
            List[Memory]: List of memory objects matching the specified criteria.

        Raises:
            ValueError: If neither memory_ids nor user_id is provided.
        """
        pass

    @abstractmethod
    async def remove_memories(
        self, *, user_id: Optional[str] = None, memory_ids: Optional[List[str]] = None
    ) -> None:
        """Remove memories from storage.

        Remove memories based on either user ID or specific memory IDs.
        At least one parameter must be provided.

        Args:
            user_id (Optional[str]): Optional user ID to remove all memories for a specific user.
            memory_ids (Optional[List[str]]): Optional list of specific memory IDs to remove.

        Raises:
            ValueError: If neither memory_ids nor user_id is provided.
        """
        pass

    @abstractmethod
    async def add_message(self, message: MessageInput) -> Message:
        """Add a new message to be processed into memory.

        This method stores a message and queues it for processing into long-term memories.

        Args:
            message (MessageInput): MessageInput object containing the message content and metadata.

        Returns:
            Message: The stored message object with assigned ID and metadata.
        """
        pass

    @abstractmethod
    async def get_messages(self, message_ids: List[str]) -> List[Message]:
        """Retrieve messages by their IDs.

        Args:
            message_ids (List[str]): List of message IDs to retrieve.

        Returns:
            List[Message]: List of message objects matching the provided IDs.
        """
        pass

    @abstractmethod
    async def remove_messages(self, message_ids: List[str]) -> None:
        """Remove messages and their related memories.

        This method removes both the messages and any memories that were derived from them.

        Args:
            message_ids (List[str]): List of message IDs to remove.
        """
        pass


class BaseMemoryModule(_CommonBaseMemoryModule, ABC):
    """Base class for the memory module interface.

    This class extends the common memory module functionality with additional features
    for searching memories and retrieving conversation history. It provides a complete
    interface for memory management in a conversational context.
    """

    @property
    @abstractmethod
    def is_listening(self) -> bool:
        """Get the listening state of the memory module.

        Returns:
            bool: True if the module is listening for messages, False otherwise.
        """
        pass

    @abstractmethod
    async def search_memories(
        self,
        *,
        user_id: Optional[str],
        query: Optional[str] = None,
        topic: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """Search for relevant memories based on various criteria.

        This method allows searching memories using text queries or topics, optionally
        filtered by user and limited in quantity. One of query or topic must be provided.

        Args:
            user_id (Optional[str]): Filter memories by specific user ID. If None, search across all users.
            query (Optional[str]): Search string to match against memory content. Required if topic is None.
            topic (Optional[str]): Filter memories by specific topic. Required if query is None.
            limit (Optional[int]): Maximum number of memories to return. If None, returns all matching memories.

        Returns:
            List[Memory]: List of memory objects matching the search criteria, ordered by relevance.

        Raises:
            ValueError: If neither query nor topic is provided.
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

        This method allows fetching conversation history using various time-based or
        quantity-based filters.

        Args:
            conversation_ref (str): Unique identifier for the conversation.
            n_messages (Optional[int]): Number of most recent messages to retrieve.
            last_minutes (Optional[float]): Retrieve messages from the last N minutes.
            before (Optional[datetime]): Retrieve messages before this timestamp.

        Returns:
            List[Message]: List of message objects from the conversation history,
                          ordered chronologically (oldest to newest).

        Raises:
            ValueError: If no filtering criteria (n_messages, last_minutes, or before) is provided.
        """
        pass

    async def listen(self) -> None:
        """Enable scheduling of memory extraction tasks from messages

        This method enables automatic scheduling of memory extraction tasks from messages.

        This should be called when the module is initialzed or when
        the server starts
        """
        pass

    async def process_messages(self, conversation_ref: str) -> None:
        """Process messages from the message buffer for a specific conversation

        This method can be called to force processing of messages from the message buffer
        for a specific conversation.

        If the module is listening, you don't need to call this method, but it can
        be used to force processing of messages for a specific conversation.

        If the module is not listening, this method can be used to process whatever
        messages are in the message buffer for the given conversation.
        """
        pass

    async def shutdown(self) -> None:
        """Shutdown the memory module

        This method should be called when the module is shutting down. This is only
        required if the module is listening.
        """
        pass


class BaseScopedMemoryModule(_CommonBaseMemoryModule, ABC):
    """Base class for the scoped memory module interface.

    This class provides memory module functionality that is scoped to a specific conversation
    and set of users. It's designed for managing memories within a limited context rather
    than across the entire system.
    """

    @property
    @abstractmethod
    def conversation_ref(self) -> str:
        """Get the conversation reference this module is scoped to.

        Returns:
            str: The unique identifier for the conversation.
        """
        ...

    @property
    @abstractmethod
    def users_in_conversation_scope(self) -> List[str]:
        """Get the list of users in the conversation scope.

        Returns:
            List[str]: List of user IDs that are part of this conversation scope.
        """
        ...

    @abstractmethod
    async def retrieve_conversation_history(
        self,
        *,
        n_messages: Optional[int] = None,
        last_minutes: Optional[float] = None,
        before: Optional[datetime] = None,
    ) -> List[Message]:
        """Retrieve conversation history for the scoped conversation.

        Similar to BaseMemoryModule.retrieve_conversation_history but automatically
        uses the scoped conversation_ref.

        Args:
            n_messages (Optional[int]): Number of most recent messages to retrieve.
            last_minutes (Optional[float]): Retrieve messages from the last N minutes.
            before (Optional[datetime]): Retrieve messages before this timestamp.

        Returns:
            List[Message]: List of message objects from the conversation history,
                          ordered chronologically.

        Raises:
            ValueError: If no filtering criteria is provided.
        """
        pass

    @abstractmethod
    async def search_memories(
        self,
        *,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        topic: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """Search memories within the scoped conversation context.

        Similar to BaseMemoryModule.search_memories but limited to memories
        from users in the conversation scope. One of query or topic must be provided.

        Args:
            user_id (Optional[str]): Filter memories by specific user ID (must be in conversation scope).
            query (Optional[str]): Search string to match against memory content.
            topic (Optional[str]): Filter memories by specific topic.
            limit (Optional[int]): Maximum number of memories to return.

        Returns:
            List[Memory]: List of memory objects matching the criteria.

        Raises:
            ValueError: If neither query nor topic is provided.
            InvalidUserError: If user_id is provided but not in conversation scope.
        """
        pass

    @abstractmethod
    async def process_messages(self) -> None:
        """Process messages for a specific conversation

        This method should be called to process messages for a specific conversation.

        This is only required if the module is not listening for automatic processing.
        """
        pass
