from abc import ABC, abstractmethod
from typing import List, Optional

from memory_module.interfaces.types import Memory, Message


class BaseMemoryCore(ABC):
    """Base class for the memory core component."""

    @abstractmethod
    async def process_messages(self, messages: List[Message]) -> None:
        """Process multiple messages into memories."""
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        user_id: Optional[str],
    ) -> List[Memory]:
        """Retrieve memories based on a query."""
        pass