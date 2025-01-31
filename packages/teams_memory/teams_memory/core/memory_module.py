"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

from teams_memory.config import MemoryModuleConfig
from teams_memory.core.memory_core import MemoryCore
from teams_memory.core.message_queue import MessageQueue
from teams_memory.interfaces.base_memory_core import BaseMemoryCore
from teams_memory.interfaces.base_memory_module import (
    BaseMemoryModule,
    BaseScopedMemoryModule,
)
from teams_memory.interfaces.base_message_queue import BaseMessageQueue
from teams_memory.interfaces.errors import InvalidUserError
from teams_memory.interfaces.types import (
    Memory,
    MemoryWithAttributions,
    Message,
    MessageInput,
)
from teams_memory.services.llm_service import LLMService
from teams_memory.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class MemoryModule(BaseMemoryModule):
    """Implementation of the memory module interface."""

    _is_listening: bool = False

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

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    async def listen(self) -> None:
        """Enable scheduling of memory extraction tasks from messages"""
        if self._is_listening:
            logger.warning("MemoryModule is already listening")
            return

        await self.message_queue.initialize()
        self._is_listening = True

    async def process_messages(self, conversation_ref: str) -> None:
        return await self.message_queue.process_messages(conversation_ref)

    async def add_message(self, message: MessageInput) -> Message:
        """Add a message to be processed into memory."""
        logger.debug(
            f"add message to memory module. {message.type}: `{message.content}`"
        )
        message_res = await self.memory_core.add_message(message)
        await self.message_queue.enqueue(message_res)
        return message_res

    def _validate_topic(self, topic: Optional[str]) -> bool:
        """Validate topic. If topic is None, return None. Otherwise, return topic."""
        if topic is None:
            return True
        return any(topic in t.name for t in self.config.topics)

    async def answer_question(
        self,
        user_id: Optional[str],
        question: str,
        query: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> Optional[Tuple[str, List[Memory]]]:
        """Answer a question based on the existing memories."""
        return await self.memory_core.answer_question(
            user_id=user_id, question=question, query=query, topic=topic
        )

    async def search_memories(
        self,
        user_id: Optional[str],
        query: Optional[str] = None,
        topic: Optional[str] = None,
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

        if not self._validate_topic(topic):
            raise ValueError(f"Topic {topic} is not in the config")

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

    async def get_memories_with_attributions(
        self, memory_ids: List[str]
    ) -> List[MemoryWithAttributions]:
        """A utility method that get memories and their attributed messages.
        This is useful in scenarioes where citations are needed for a memory.


        Args:
            memory_ids: List of memory IDs to fetch

        Returns:
            List of MemoryWithAttributions objects containing memories and their messages
        """
        if not memory_ids:

            return []

        memories = await self.get_memories(memory_ids=memory_ids)
        if not memories:
            return []

        # Collect all unique message IDs across all memories
        all_message_ids = set()
        for memory in memories:
            if memory.message_attributions:
                all_message_ids.update(memory.message_attributions)

        # Fetch all messages in one call
        if not all_message_ids:
            return []

        messages = await self.get_messages(list(all_message_ids))
        messages_by_id = {msg.id: msg for msg in messages}

        # Build result by matching messages to memories
        result = []
        for memory in memories:
            if not memory.message_attributions:
                continue

            memory_messages = [
                messages_by_id[msg_id]
                for msg_id in memory.message_attributions
                if msg_id in messages_by_id
            ]

            if memory_messages:
                result.append(
                    MemoryWithAttributions(memory=memory, messages=memory_messages)
                )

        return result

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

    async def shutdown(self) -> None:
        await self.message_queue.shutdown()


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
            raise InvalidUserError(f"User {user_id} is not in the conversation scope")
        if not user_id:
            if len(self.users_in_conversation_scope) > 1:
                raise InvalidUserError(
                    "No user id provided and there are multiple users in the conversation scope"
                )
            return self.users_in_conversation_scope[0]
        return user_id

    async def answer_question(
        self,
        user_id: Optional[str],
        question: str,
        query: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> Optional[Tuple[str, List[Memory]]]:
        validated_user_id = self._validate_user(user_id)
        return await self.memory_module.answer_question(
            user_id=validated_user_id, question=question, query=query, topic=topic
        )

    async def search_memories(
        self,
        *,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        topic: Optional[str] = None,
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
    async def get_memories_with_attributions(
        self, memory_ids: List[str]
    ) -> List[MemoryWithAttributions]:
        return await self.memory_module.get_memories_with_attributions(memory_ids)

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

    async def process_messages(self) -> None:
        return await self.memory_module.process_messages(self.conversation_ref)
