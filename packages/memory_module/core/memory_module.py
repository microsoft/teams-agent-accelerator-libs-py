from typing import List, Optional

from memory_module.config import MemoryModuleConfig
from memory_module.core.memory_core import MemoryCore
from memory_module.core.message_queue import MessageQueue
from memory_module.interfaces.base_memory_core import BaseMemoryCore
from memory_module.interfaces.base_memory_module import BaseMemoryModule
from memory_module.interfaces.base_message_queue import BaseMessageQueue
from memory_module.interfaces.types import Memory, Message, ShortTermMemoryRetrievalConfig
from memory_module.services.llm_service import LLMService


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
        self.memory_core = memory_core or MemoryCore(config=config, llm_service=self.llm_service)
        self.message_queue = message_queue or MessageQueue(config=config, memory_core=self.memory_core)

    async def add_message(self, message: Message) -> None:
        """Add a message to be processed into memory."""
        await self.memory_core.add_short_term_memory(message)
        await self.message_queue.enqueue(message)

    async def retrieve_memories(self, query: str, user_id: Optional[str], limit: Optional[int]) -> List[Memory]:
        """Retrieve relevant memories based on a query."""
        return await self.memory_core.retrieve(query, user_id, limit)

    async def update_memory(self, memory_id: str, updated_memory: str) -> None:
        """Update memory with new fact"""
        return await self.memory_core.update(memory_id, updated_memory)

    async def remove_memories(self, user_id: str) -> None:
        """Remove memories based on user id."""
        return await self.memory_core.remove_memories(user_id)

    async def retrieve_chat_history(
        self, conversation_ref: str, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        return await self.memory_core.retrieve_chat_history(conversation_ref, config)
