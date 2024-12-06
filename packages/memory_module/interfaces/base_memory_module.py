from abc import ABC, abstractmethod
from typing import List

from memory_module.interfaces.types import Memory, Message


class BaseMemoryModule(ABC):
    """Base class for the memory module interface."""

    @abstractmethod
    async def add_message(self, message: Message) -> None:
        """Add a message to be processed into memory."""
        pass

    @abstractmethod
    async def retrieve_memories(self, query: str) -> List[Memory]:
        """Retrieve relevant memories based on a query."""
        pass
