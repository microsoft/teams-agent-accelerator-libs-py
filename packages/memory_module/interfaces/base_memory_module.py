from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from memory_module.interfaces.types import (
    Memory,
    Message,
    MessageInput,
    RetrievalConfig,
    ShortTermMemoryRetrievalConfig,
)


class BaseMemoryModule(ABC):
    """Base class for the memory module interface."""

    @abstractmethod
    async def add_message(self, message: MessageInput) -> Message:
        """Add a message to be processed into memory."""
        pass

    # TODO: Ambiguity between retrieve_memories and get_memories. Should update to "search_memories" or similar.
    # TODO: "search_mmemories" make the change
    @abstractmethod
    async def retrieve_memories(
        self,
        user_id: Optional[str],
        config: RetrievalConfig,
    ) -> List[Memory]:
        """Retrieve relevant memories based on a query."""
        pass

    # TODO: Get rid of the ShortTermMemoryRetrievalConfig pattern in the entire codebase
    # and change it to named parameters.
    # Except the MemoryModuleConfig.
    @abstractmethod
    async def retrieve_chat_history(
        self, conversation_ref: str, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        pass

    # TODO: It might be a good ideas to consolidate "get_memories" and "get_user_memories" into one method
    # That takes an optional user_id parameter. This way we can avoid having two methods that do the same thing.
    # TODO: Better name is "get_memories_by_id"
    # TODO: Consolidate both methods into one.
    @abstractmethod
    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
        """Get memories based on memory ids."""
        pass

    # # TODO: Better name is "get_memories_by_user_id"
    # @abstractmethod
    # async def get_memories(self, user_id: str) -> List[Memory]:
    #     """Get memories based on user id."""
    #     pass

    # TODO: Might seem the same as "retrieve_chat_history". Better name is "get_messages_by_memory_ids".
    # TODO: Change this to get_messages_by_id. I.e. not getting it by memory_ids.
    @abstractmethod
    async def get_messages(self, memory_ids: List[str]) -> Dict[str, List[Message]]:
        """Get messages based on memory ids."""
        pass

    @abstractmethod
    async def remove_messages(self, message_ids: List[str]) -> None:
        """Remove messages and related memories"""
        pass

    # TODO: Add remove_memories method to the interface.
    @abstractmethod
    async def remove_memories(self, message):
        pass
