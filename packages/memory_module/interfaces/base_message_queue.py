from abc import ABC, abstractmethod
from typing import List

from memory_module.interfaces.types import Message


class BaseMessageQueue(ABC):
    """Base class for the message queue component."""

    @abstractmethod
    async def enqueue(self, message: Message) -> None:
        """Add a message to the queue for a given conversation."""
        pass

    @abstractmethod
    async def dequeue(self, message_ids: List[str]) -> None:
        """Remove a list of messages from the queue"""
        pass
