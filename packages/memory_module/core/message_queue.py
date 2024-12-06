from typing import List, Optional

from memory_module.core.message_buffer import MessageBuffer
from memory_module.interfaces.base_memory_core import BaseMemoryCore
from memory_module.interfaces.base_message_buffer_storage import BaseMessageBufferStorage
from memory_module.interfaces.base_message_queue import BaseMessageQueue
from memory_module.interfaces.types import Message


class MessageQueue(BaseMessageQueue):
    """Implementation of the message queue component."""

    def __init__(
        self,
        memory_core: BaseMemoryCore,
        buffer_size: int = 5,
        message_queue_storage: Optional[BaseMessageBufferStorage] = None,
    ):
        """Initialize the message queue with a memory core and optional message buffer.

        Args:
            memory_core: Core memory processing component
            buffer_size: Number of messages to buffer (used if message_buffer not provided)
            message_buffer: Optional custom message buffer implementation
        """
        self.memory_core = memory_core
        self.message_buffer = MessageBuffer(
            buffer_size=buffer_size,
            process_callback=self._process_for_episodic_messages,
            storage=message_queue_storage,
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
