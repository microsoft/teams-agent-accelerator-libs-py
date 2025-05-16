from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from teams_memory.interfaces.types import Message, MessageInput


class BaseMessageStorage(ABC):
    """Base class for the storage component for messages only."""

    @abstractmethod
    async def upsert_message(self, message: MessageInput) -> Message:
        """Store or update a message in the storage system.

        Args:
            message (MessageInput): The Message object to store or update

        Returns:
            Message: The stored/updated message with assigned ID and metadata
        """
        pass

    @abstractmethod
    async def get_messages(self, message_ids: List[str]) -> List[Message]:
        """Retrieve messages by their IDs.

        Args:
            message_ids (List[str]): List of message IDs to retrieve

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
            message_ids (List[str]): List of message IDs to remove
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
            conversation_ref (str): Unique identifier for the conversation
            n_messages (Optional[int]): Number of most recent messages to retrieve
            last_minutes (Optional[float]): Retrieve messages from the last N minutes
            before (Optional[datetime]): Retrieve messages before this timestamp

        Returns:
            List[Message]: List of message objects from the conversation history,
                          ordered chronologically (oldest to newest)

        Raises:
            ValueError: If no filtering criteria is provided
        """
        pass
