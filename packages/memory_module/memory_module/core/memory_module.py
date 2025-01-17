import logging
from typing import Dict, List, Optional

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
    RetrievalConfig,
    ShortTermMemoryRetrievalConfig,
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
        self.memory_core: BaseMemoryCore = memory_core or MemoryCore(config=config, llm_service=self.llm_service)
        self.message_queue: BaseMessageQueue = message_queue or MessageQueue(
            config=config, memory_core=self.memory_core
        )

        if config.enable_logging:
            configure_logging()

        logger.debug(f"MemoryModule initialized with config: {config}")

    async def add_message(self, message: MessageInput) -> Message:
        """Add a message to be processed into memory."""
        logger.debug(f"add message to memory module. {message.type}: `{message.content}`")
        message_res = await self.memory_core.add_short_term_memory(message)
        await self.message_queue.enqueue(message_res)
        return message_res

    async def retrieve_memories(
        self,
        user_id: Optional[str],
        config: RetrievalConfig,
    ) -> List[Memory]:
        """Retrieve relevant memories based on a query."""
        logger.debug("retrieve memories from config: %s", config)
        memories = await self.memory_core.retrieve_memories(user_id=user_id, config=config)
        logger.debug(f"retrieved memories: {memories}")
        return memories

    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
        return await self.memory_core.get_memories(memory_ids)

    async def get_user_memories(self, user_id: str) -> List[Memory]:
        return await self.memory_core.get_user_memories(user_id)

    async def get_messages(self, memory_ids: List[str]) -> Dict[str, List[Message]]:
        return await self.memory_core.get_messages(memory_ids)

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

    async def remove_memories(self, user_id: str) -> None:
        """Remove memories based on user id."""
        logger.debug(f"removing all memories associated with user ({user_id})")
        return await self.memory_core.remove_memories(user_id)

    async def retrieve_chat_history(
        self, conversation_ref: str, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        return await self.memory_core.retrieve_chat_history(conversation_ref, config)


class ScopedMemoryModule(BaseScopedMemoryModule):
    def __init__(self, memory_module: BaseMemoryModule, users_in_conversation_scope: List[str], conversation_ref: str):
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
        if user_id and user_id not in self.users_in_conversation_scope:
            raise ValueError(f"User {user_id} is not in the conversation scope")
        if not user_id:
            if len(self.users_in_conversation_scope) > 1:
                raise ValueError("No user id provided and there are multiple users in the conversation scope")
            return self.users_in_conversation_scope[0]
        return user_id

    async def retrieve_memories(self, config: RetrievalConfig, user_id: Optional[str] = None) -> List[Memory]:
        validated_user_id = self._validate_user(user_id)
        return await self.memory_module.retrieve_memories(validated_user_id, config)

    async def retrieve_chat_history(self, config: ShortTermMemoryRetrievalConfig) -> List[Message]:
        return await self.memory_module.retrieve_chat_history(self.conversation_ref, config)

    # Implement abstract methods by forwarding to memory_module
    async def add_message(self, message):
        return await self.memory_module.add_message(message)

    async def get_memories(self, *args, **kwargs):
        return await self.memory_module.get_memories(*args, **kwargs)

    async def get_messages(self, *args, **kwargs):
        return await self.memory_module.get_messages(*args, **kwargs)

    async def get_user_memories(self, *args, **kwargs):
        return await self.memory_module.get_user_memories(*args, **kwargs)

    async def remove_messages(self, *args, **kwargs):
        return await self.memory_module.remove_messages(*args, **kwargs)
