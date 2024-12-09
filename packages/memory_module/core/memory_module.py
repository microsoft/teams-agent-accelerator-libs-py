from typing import List, Optional

from memory_module.config import MemoryModuleConfig
from memory_module.core.memory_core import MemoryCore
from memory_module.core.message_queue import MessageQueue
from memory_module.interfaces.base_memory_core import BaseMemoryCore
from memory_module.interfaces.base_memory_module import BaseMemoryModule
from memory_module.interfaces.base_message_queue import BaseMessageQueue
from memory_module.interfaces.types import Memory, Message
from memory_module.services.llm_service import LLMService


class MemoryModule(BaseMemoryModule):
    """Implementation of the memory module interface."""

    def __init__(
        self,
        config: MemoryModuleConfig,
        llm_service: LLMService,
        memory_core: Optional[BaseMemoryCore] = None,
        message_queue: Optional[BaseMessageQueue] = None,
    ):
        """Initialize the memory module.

        Args:
            config: Memory module configuration
            llm_service: LLM service instance
            memory_core: Optional BaseMemoryCore instance
            message_queue: Optional BaseMessageQueue instance
        """
        self.config = config
        self.llm_service = llm_service

        if memory_core is None:
            self.memory_core = MemoryCore(config=config, llm_service=llm_service)
        else:
            self.memory_core = memory_core

        if message_queue is None:
            self.message_queue = MessageQueue(config=config, memory_core=self.memory_core)
        else:
            self.message_queue = message_queue

    async def add_message(self, message: Message) -> None:
        """Add a message to be processed into memory."""
        await self.message_queue.enqueue(message)

    async def retrieve_memories(self, query: str, user_id: Optional[str]) -> List[Memory]:
        """Retrieve relevant memories based on a query."""
        return await self.memory_core.retrieve(query, user_id)
