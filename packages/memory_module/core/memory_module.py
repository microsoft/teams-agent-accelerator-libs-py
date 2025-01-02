import logging
from typing import Dict, List, Optional

from memory_module.config import MemoryModuleConfig
from memory_module.core.memory_core import MemoryCore
from memory_module.core.message_queue import MessageQueue
from memory_module.interfaces.base_memory_core import BaseMemoryCore
from memory_module.interfaces.base_memory_module import BaseMemoryModule
from memory_module.interfaces.base_message_queue import BaseMessageQueue
from memory_module.interfaces.types import Memory, Message, MessageInput, ShortTermMemoryRetrievalConfig
from memory_module.services.llm_service import LLMService
from memory_module.utils.logging import set_verbose_logging

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
            set_verbose_logging()

    async def add_message(self, message: MessageInput) -> Message:
        """Add a message to be processed into memory."""
        logger.debug(f"add message to memory module. {message.type}: `{message.content}`")
        message_res = await self.memory_core.add_short_term_memory(message)
        await self.message_queue.enqueue(message_res)
        return message_res

    async def retrieve_memories(self, query: str, user_id: Optional[str], limit: Optional[int]) -> List[Memory]:
        """Retrieve relevant memories based on a query."""
        logger.debug(f"retrieve memories from (query: {query}, user_id: {user_id}, limit: {limit})")
        memories = await self.memory_core.retrieve_memories(query, user_id, limit)
        logger.debug(f"retrieved memories: {memories}")
        return memories

    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
        return await self.memory_core.get_memories(memory_ids)

    async def get_user_memories(self, user_id: str) -> List[Memory]:
        return await self.memory_core.get_user_memories(user_id)

    async def get_messages(self, memory_ids: List[str]) -> Dict[str, List[Message]]:
        return await self.memory_core.get_messages(memory_ids)

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
