from abc import ABC, abstractmethod
from typing import List, Optional

from memory_module.interfaces.types import Memory, Message, ShortTermMemoryRetrievalConfig


class BaseMemoryModule(ABC):
    """Base class for the memory module interface."""

    @abstractmethod
    async def add_message(self, message: Message) -> None:
        """Add a message to be processed into memory."""
        pass

    @abstractmethod
    async def retrieve_memories(self, query: str, user_id: Optional[str], limit: Optional[int]) -> List[Memory]:
        """Retrieve relevant memories based on a query."""
        pass

    @abstractmethod
    async def retrieve_chat_history(
        self, conversation_ref: str, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        pass
