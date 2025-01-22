"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from memory_module.interfaces.types import Message


class BaseMessageBufferStorage(ABC):
    """Base class for storing buffered messages."""

    @abstractmethod
    async def store_buffered_message(self, message: Message) -> None:
        """Store a message in the buffer.

        Args:
            message: The Message object to store
        """
        pass

    @abstractmethod
    async def get_buffered_messages(self, conversation_ref: str) -> List[Message]:
        """Retrieve all buffered messages for a conversation.

        Args:
            conversation_ref: The conversation reference to retrieve messages for

        Returns:
            List of Message objects for the conversation
        """
        pass

    @abstractmethod
    async def get_conversations_from_buffered_messages(
        self, message_ids: List[str]
    ) -> Dict[str, List[str]]:
        """Get conversation - messages maps"""
        pass

    @abstractmethod
    async def clear_buffered_messages(
        self, conversation_ref: str, before: Optional[datetime.datetime] = None
    ) -> None:
        """Remove all buffered messages for a conversation. If the `before` parameter is provided,
        only messages created on or before that time will be removed.

        Args:
            conversation_ref: The conversation reference to clear messages for
            before: Optional cutoff time to clear messages before
        """
        pass

    @abstractmethod
    async def remove_buffered_messages_by_id(self, message_ids: List[str]) -> None:
        """Remove list of messages in buffered storage

        Args:
            message_ids: List of messages to be removed
        """

    @abstractmethod
    async def count_buffered_messages(
        self, conversation_refs: List[str]
    ) -> Dict[str, int]:
        """Count the number of buffered messages for selected conversations.

        Args:
            conversation_ref: The conversation reference to count messages for

        Returns:
            Number of buffered messages for the conversation
        """
        pass
