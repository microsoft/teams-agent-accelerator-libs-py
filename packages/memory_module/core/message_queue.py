from typing import List, Optional

from memory_module.config import MemoryModuleConfig
from memory_module.core.message_buffer import MessageBuffer
from memory_module.interfaces.base_memory_core import BaseMemoryCore
from memory_module.interfaces.base_message_buffer_storage import BaseMessageBufferStorage
from memory_module.interfaces.base_message_queue import BaseMessageQueue
from memory_module.interfaces.types import Message


class MessageQueue(BaseMessageQueue):
    """Implementation of the message queue component."""

    def __init__(
        self,
        config: MemoryModuleConfig,
        memory_core: BaseMemoryCore,
        message_buffer_storage: Optional[BaseMessageBufferStorage] = None,
    ):
        """Initialize the message queue with a memory core and optional message buffer.

        Args:
            config: Memory module configuration
            memory_core: Core memory processing component
            message_buffer_storage: Optional custom message buffer storage implementation
        """
        self.memory_core = memory_core
        self.message_buffer = MessageBuffer(
            config=config,
            process_callback=self._process_for_episodic_messages,
            storage=message_buffer_storage,
        )

    async def enqueue(self, message: Message) -> None:
        """Add a message to the queue for processing.

        Messages are buffered by conversation_ref. When enough messages accumulate,
        they are processed as an episodic memory. Individual messages are still
        processed immediately as semantic memories.
        """
        # Process message immediately for semantic memory
        await self._process_for_semantic_messages([message])

        # Buffer message for episodic memory processing
        await self.message_buffer.add_message(message)

    async def _process_for_semantic_messages(self, messages: List[Message]) -> None:
        """Process a list of messages using the memory core.

        Args:
            messages: List of messages to process
        """
        await self.memory_core.process_semantic_messages(messages)

    async def _process_for_episodic_messages(self, messages: List[Message]) -> None:
        """Process a list of messages as episodic memory.

        Args:
            messages: List of messages to process
        """
        await self.memory_core.process_episodic_messages(messages)
