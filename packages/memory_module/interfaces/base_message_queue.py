from abc import ABC, abstractmethod

from memory_module.interfaces.types import Message


class BaseMessageQueue(ABC):
    """Base class for the message queue component."""

    @abstractmethod
    async def enqueue(self, conversation_ref: str, message: Message) -> None:
        """Add a message to the queue for a given conversation."""
        pass

    @abstractmethod
    async def dequeue(self, conversation_ref: str) -> Message:
        """Remove and return the next message from the queue."""
        pass

    @abstractmethod
    async def get_queue_length(self, conversation_ref: str) -> int:
        """Get the current length of the queue for a conversation."""
        pass