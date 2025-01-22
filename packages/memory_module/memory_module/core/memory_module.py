"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import logging
from datetime import datetime
from typing import List, Optional

from memory_module.config import MemoryModuleConfig
from memory_module.core.memory_core import MemoryCore
from memory_module.core.message_queue import MessageQueue
from memory_module.interfaces.base_memory_core import BaseMemoryCore
from memory_module.interfaces.base_memory_module import (
    BaseMemoryModule,
    BaseScopedMemoryModule,
)
from memory_module.interfaces.base_message_queue import BaseMessageQueue
from memory_module.interfaces.types import (
    Memory,
    Message,
    MessageInput,
    Topic,
)
from memory_module.services.llm_service import LLMService
from memory_module.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class MemoryModule(BaseMemoryModule):
    """Implementation of the memory module interface."""

    def __init__(
        self,
        config: MemoryModuleConfig,
        llm_service: Optional[LLMService] = None,
        memory_core: Optional[BaseMemoryCore] = None,
        message_queue: Optional[BaseMessageQueue] = None,
    ):
        """Initialize the memory module.

        Args:
            config: Memory module configuration
            llm_service: Optional LLM service instance
            memory_core: Optional BaseMemoryCore instance
            message_queue: Optional BaseMessageQueue instance
        """
        self.config = config

        self.llm_service = llm_service or LLMService(config=config.llm)
        self.memory_core: BaseMemoryCore = memory_core or MemoryCore(
            config=config, llm_service=self.llm_service
        )
        self.message_queue: BaseMessageQueue = message_queue or MessageQueue(
            config=config, memory_core=self.memory_core
        )

        if config.enable_logging:
            configure_logging()

        logger.debug(f"MemoryModule initialized with config: {config}")

    async def add_message(self, message: MessageInput) -> Message:
        """Add a message to be processed into memory."""
        logger.debug(
            f"add message to memory module. {message.type}: `{message.content}`"
        )
        message_res = await self.memory_core.add_message(message)
        await self.message_queue.enqueue(message_res)
        return message_res

    async def search_memories(
        self,
        user_id: Optional[str],
        query: Optional[str] = None,
        topic: Optional[Topic] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """Retrieve relevant memories based on a query."""
        logger.debug(
            "retrieve memories from config: user_id=%s, query=%s, topic=%s, limit=%s",
            user_id,
            query,
            topic,
            limit,
        )

        if query is None and topic is None:
            raise ValueError("Either query or topic must be provided")

        memories = await self.memory_core.search_memories(
            user_id=user_id, query=query, topic=topic, limit=limit
        )
        logger.debug(f"retrieved memories: {memories}")
        return memories

    async def get_memories(
        self, *, memory_ids: Optional[List[str]] = None, user_id: Optional[str] = None
    ) -> List[Memory]:
        """Get memories based on memory ids or user id."""
        if memory_ids is None and user_id is None:
            raise ValueError("Either memory_ids or user_id must be provided")
        return await self.memory_core.get_memories(
            memory_ids=memory_ids, user_id=user_id
        )

    async def get_messages(self, message_ids: List[str]) -> List[Message]:
        return await self.memory_core.get_messages(message_ids)

    async def remove_messages(self, message_ids: List[str]) -> None:
        """
        Message will be in three statuses:
        1. Queued but not processed. Handle by message_queue.dequeue
        2. In processing. Possibly handle by message_core.remove_messages is process is done.
        Otherwise we can be notified with warning log.
        3. Processed and memory is created. Handle by message_core.remove_messages
        """
        await self.message_queue.dequeue(message_ids)
        if message_ids:
            await self.memory_core.remove_messages(message_ids)

    async def update_memory(self, memory_id: str, updated_memory: str) -> None:
        """Update memory with new fact"""
        return await self.memory_core.update_memory(memory_id, updated_memory)

    async def remove_memories(
        self, *, user_id: Optional[str] = None, memory_ids: Optional[List[str]] = None
    ) -> None:
        """Remove memories based on user id."""
        logger.debug(f"removing all memories associated with user ({user_id})")
        return await self.memory_core.remove_memories(
            user_id=user_id, memory_ids=memory_ids
        )

    async def retrieve_conversation_history(
        self,
        conversation_ref: str,
        *,
        n_messages: Optional[int] = None,
        last_minutes: Optional[float] = None,
        before: Optional[datetime] = None,
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""

        if n_messages is None and last_minutes is None:
            raise ValueError("Either n_messages or last_minutes must be provided")

        return await self.memory_core.retrieve_conversation_history(
            conversation_ref,
            n_messages=n_messages,
            last_minutes=last_minutes,
            before=before,
        )


class ScopedMemoryModule(BaseScopedMemoryModule):
    def __init__(
        self,
        memory_module: BaseMemoryModule,
        users_in_conversation_scope: List[str],
        conversation_ref: str,
    ):
        self.memory_module = memory_module
        self._users_in_conversation_scope = users_in_conversation_scope
        self._conversation_ref = conversation_ref

    @property
    def users_in_conversation_scope(self):
        return self._users_in_conversation_scope

    @property
    def conversation_ref(self):
        return self._conversation_ref

    def _validate_user(self, user_id: Optional[str]) -> str:
        """
        Validate user_id. If user_id is not provided, we need to ensure that there
        is only one user in the conversation scope.

        Otherwise, we require that the user_id is provided in the arguments.
        """

        if user_id and user_id not in self.users_in_conversation_scope:
            raise ValueError(f"User {user_id} is not in the conversation scope")
        if not user_id:
            if len(self.users_in_conversation_scope) > 1:
                raise ValueError(
                    "No user id provided and there are multiple users in the conversation scope"
                )
            return self.users_in_conversation_scope[0]
        return user_id

    async def search_memories(
        self,
        *,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        topic: Optional[Topic] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        validated_user_id = self._validate_user(user_id)
        return await self.memory_module.search_memories(
            user_id=validated_user_id, query=query, topic=topic, limit=limit
        )

    async def retrieve_conversation_history(
        self,
        *,
        n_messages: Optional[int] = None,
        last_minutes: Optional[float] = None,
        before: Optional[datetime] = None,
    ) -> List[Message]:
        return await self.memory_module.retrieve_conversation_history(
            self.conversation_ref,
            n_messages=n_messages,
            last_minutes=last_minutes,
            before=before,
        )

    async def get_memories(
        self,
        *,
        memory_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ):
        validated_user_id = self._validate_user(user_id) if user_id else None
        return await self.memory_module.get_memories(
            memory_ids=memory_ids, user_id=validated_user_id
        )

    # Implement abstract methods by forwarding to memory_module
    async def add_message(self, message):
        return await self.memory_module.add_message(message)

    async def get_messages(self, *args, **kwargs):
        return await self.memory_module.get_messages(*args, **kwargs)

    async def remove_messages(self, *args, **kwargs):
        return await self.memory_module.remove_messages(*args, **kwargs)

    async def remove_memories(
        self, *, user_id: Optional[str] = None, memory_ids: Optional[List[str]] = None
    ):
        # If user_id is not provided, we still need to ensure that in a scoped setting,
        # we are only removing memories that belong to a user in this conversation scope.
        # So we are validating the user_id here.
        validated_user_id = self._validate_user(user_id)
        return await self.memory_module.remove_memories(
            user_id=validated_user_id, memory_ids=memory_ids
        )
