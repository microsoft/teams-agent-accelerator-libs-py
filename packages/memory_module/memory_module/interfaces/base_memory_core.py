"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from memory_module.interfaces.types import (
    Memory,
    Message,
    MessageInput,
    Topic,
)


class BaseMemoryCore(ABC):
    """Base class for the memory core component."""

    @abstractmethod
    async def process_semantic_messages(
        self,
        messages: List[Message],
        existing_memories: Optional[List[Memory]] = None,
    ) -> None:
        """Process multiple messages into semantic memories (general facts, preferences)."""
        pass

    @abstractmethod
    async def search_memories(
        self,
        *,
        user_id: Optional[str],
        query: Optional[str] = None,
        topic: Optional[Topic] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """Retrieve memories based on a query."""
        pass

    @abstractmethod
    async def update_memory(self, memory_id: str, updated_memory: str) -> None:
        """Update memory with new fact."""
        pass

    @abstractmethod
    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
        """Get memories based on memory ids."""
        pass

    @abstractmethod
    async def get_memories_from_message(self, message_id: str) -> List[Memory]:
        """Get memories based on message id."""
        pass

    @abstractmethod
    async def get_user_memories(self, user_id: str) -> List[Memory]:
        """Get memories based on user id."""
        pass

    @abstractmethod
    async def remove_messages(self, message_ids: List[str]) -> None:
        """Remove messages and related memories."""
        pass

    @abstractmethod
    async def remove_memories(self, user_id: str) -> None:
        """Remove memories based on user id."""
        pass

    @abstractmethod
    async def get_messages(self, memory_ids: List[str]) -> Dict[str, List[Message]]:
        """Get messages based on memory ids."""
        pass

    @abstractmethod
    async def add_short_term_memory(self, message: MessageInput) -> Message:
        """Add a short-term memory entry."""
        pass

    @abstractmethod
    async def retrieve_conversation_history(
        self,
        conversation_ref: str,
        *,
        n_messages: Optional[int] = None,
        last_minutes: Optional[float] = None,
        before: Optional[datetime] = None,
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        pass
