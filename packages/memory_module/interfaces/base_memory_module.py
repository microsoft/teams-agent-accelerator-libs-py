from abc import ABC, abstractmethod
from typing import Dict, List, Optional

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

    @abstractmethod
    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
        """Get memories based on memory ids."""
        pass

    @abstractmethod
    async def get_messages(self, memory_ids: List[str]) -> Dict[str, List[Message]]:
        """Get messages based on memory ids."""
        pass
