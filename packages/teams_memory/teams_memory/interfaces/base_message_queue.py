"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod
from typing import List

from teams_memory.interfaces.types import Message


class BaseMessageQueue(ABC):
    """Base class for the message queue component.

    This class defines the interface for managing a queue of messages that need to be
    processed by the memory system. It provides methods for adding new messages and
    removing processed messages from the queue.
    """

    @abstractmethod
    async def enqueue(self, message: Message) -> None:
        """Add a new message to the processing queue.

        This method adds a message to the queue for eventual processing into memories.
        Messages are typically processed in the order they are received.

        Args:
            message (Message): The Message object to be queued for processing. This should contain
                    all necessary information like content, metadata, and conversation context.

        Raises:
            QueueFullError: If the queue has reached its maximum capacity.
            InvalidMessageError: If the message format is invalid or missing required fields.
        """
        pass

    @abstractmethod
    async def dequeue(self, message_ids: List[str]) -> None:
        """Remove processed messages from the queue.

        This method removes messages that have been successfully processed into memories
        from the queue. This helps maintain queue size and prevent reprocessing.

        Args:
            message_ids (List[str]): List of message IDs to remove from the queue. These should be
                        messages that have been successfully processed into memories.

        Raises:
            MessageNotFoundError: If any of the specified message IDs are not found in the queue.
            InvalidMessageIDError: If any of the provided message IDs are invalid.
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the message queue with pre-existing messages"""
        pass

    @abstractmethod
    async def process_messages(self, conversation_ref: str) -> None:
        """Process messages for a specific conversation"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the message queue and release resources."""
        pass
