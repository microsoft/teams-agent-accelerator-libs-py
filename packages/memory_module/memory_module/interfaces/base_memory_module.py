"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from memory_module.interfaces.types import (
    Memory,
    Message,
    MessageInput,
    RetrievalConfig,
    ShortTermMemoryRetrievalConfig,
)


class _CommonBaseMemoryModule(ABC):
    """Common Internal Base class for the memory module interface."""

    @abstractmethod
    async def add_message(self, message: MessageInput) -> Message:
        """Add a message to be processed into memory."""
        pass

    @abstractmethod
    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
        """Get memories based on memory ids."""
        pass

    @abstractmethod
    async def get_user_memories(self, user_id: str) -> List[Memory]:
        """Get memories based on user id."""
        pass

    @abstractmethod
    async def get_messages(self, memory_ids: List[str]) -> Dict[str, List[Message]]:
        """Get messages based on memory ids."""
        pass

    @abstractmethod
    async def remove_messages(self, message_ids: List[str]) -> None:
        """Remove messages and related memories"""
        pass


class BaseMemoryModule(_CommonBaseMemoryModule, ABC):
    """Base class for the memory module interface."""

    @abstractmethod
    async def retrieve_memories(
        self,
        user_id: Optional[str],
        config: RetrievalConfig,
    ) -> List[Memory]:
        """Retrieve relevant memories based on a query."""
        pass

    @abstractmethod
    async def retrieve_chat_history(
        self, conversation_ref: str, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        pass


class BaseScopedMemoryModule(_CommonBaseMemoryModule, ABC):
    """Base class for the memory module interface that is scoped to a conversation and a list of users"""

    @property
    @abstractmethod
    def conversation_ref(self): ...

    @property
    @abstractmethod
    def users_in_conversation_scope(self): ...

    @abstractmethod
    async def retrieve_chat_history(
        self, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        pass

    @abstractmethod
    async def retrieve_memories(
        self,
        *,
        user_id: Optional[str] = None,
        config: RetrievalConfig,
    ) -> List[Memory]:
        """Retrieve relevant memories based on a query."""
        pass
