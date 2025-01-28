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


class BaseMemoryCore(ABC):
    """Base class for the memory core component.

    This class defines the core memory processing functionality, handling the transformation
    of messages into memories and managing the retrieval and manipulation of both
    messages and memories.
    """

    @abstractmethod
    async def process_semantic_messages(
        self,
        messages: List[Message],
        existing_memories: Optional[List[Memory]] = None,
    ) -> None:
        """Process messages into semantic memories.

        Analyzes messages to extract semantic information and creates long-term memories.
        If the memory already exists, we ignore it.

        Args:
            messages: List of messages to process into semantic memories.
            existing_memories: Optional list of existing memories to consider during processing.
                            Useful for context and avoiding duplicate memories.

        Raises:
            ProcessingError: If there's an error during semantic processing.
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
        """Search for memories using semantic criteria.

        Performs semantic search across memories using natural language queries
        or topic-based filtering. One of query or topic must be provided.

        Args:
            user_id (Optional[str]): Filter memories by specific user ID. If None, search across all users.
            query (Optional[str]): Natural language search query to match against memory content.
            topic (Optional[str]): Filter memories by specific topic category.
            limit (Optional[int]): Maximum number of memories to return.

        Returns:
            List[Memory]: List of memories matching the search criteria, ordered by relevance.

        Raises:
            ValueError: If neither query nor topic is provided.
        """
        pass

    @abstractmethod
    async def update_memory(self, memory_id: str, updated_memory: str) -> None:
        """Update an existing memory with new information.

        Modifies the content of an existing memory while preserving its relationships
        and metadata.

        Args:
            memory_id (str): ID of the memory to update.
            updated_memory (str): New content for the memory.

        Returns:
            None

        Raises:
            MemoryNotFoundError: If the specified memory_id doesn't exist.
        """
        pass

    @abstractmethod
    async def get_memories(
        self, *, memory_ids: Optional[List[str]] = None, user_id: Optional[str] = None
    ) -> List[Memory]:
        """Retrieve memories by IDs or user.

        Fetches memories based on specific IDs or all memories for a given user.
        At least one parameter must be provided.

        Args:
            memory_ids (Optional[List[str]]): Optional list of specific memory IDs to retrieve.
            user_id (Optional[str]): Optional user ID to retrieve all memories for.

        Returns:
            List[Memory]: List of memory objects matching the criteria.

        Raises:
            ValueError: If neither memory_ids nor user_id is provided.
        """
        pass

    @abstractmethod
    async def get_memories_from_message(self, message_id: str) -> List[Memory]:
        """Retrieve memories derived from a specific message.

        Gets all memories that were created from or are associated with a particular message.

        Args:
            message_id (str): ID of the message to get related memories for.

        Returns:
            List[Memory]: List of memories derived from or related to the message.
        """
        pass

    @abstractmethod
    async def remove_memories(
        self, *, user_id: Optional[str] = None, memory_ids: Optional[List[str]] = None
    ) -> None:
        """Remove memories from storage.

        Removes memories based on user ID or specific memory IDs.
        At least one parameter must be provided.

        Args:
            user_id (Optional[str]): Optional user ID to remove all memories for.
            memory_ids (Optional[List[str]]): Optional list of specific memory IDs to remove.

        Returns:
            None

        Raises:
            ValueError: If neither memory_ids nor user_id is provided.
        """
        pass

    @abstractmethod
    async def add_message(self, message: MessageInput) -> Message:
        """Add a new message for processing.

        Stores a message and prepares it for semantic processing into memories.

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

        Raises:
            MessageNotFoundError: If any of the specified message IDs don't exist.
        """
        pass

    @abstractmethod
    async def remove_messages(self, message_ids: List[str]) -> None:
        """Remove messages and their derived memories.

        Deletes messages and any memories that were created from them.

        Args:
            message_ids (List[str]): List of message IDs to remove.

        Returns:
            None
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

        Fetches conversation messages using time-based or quantity-based filters.
        At least one criteria must be provided.

        Args:
            conversation_ref (str): Unique identifier for the conversation.
            n_messages (Optional[int]): Number of most recent messages to retrieve.
            last_minutes (Optional[float]): Retrieve messages from the last N minutes.
            before (Optional[datetime]): Retrieve messages before this timestamp.

        Returns:
            List[Message]: List of message objects from the conversation history,
                          ordered chronologically (oldest to newest).

        Raises:
            ValueError: If no filtering criteria is provided.
        """
        pass
