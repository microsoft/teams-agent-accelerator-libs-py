import datetime
from typing import List, Optional, Tuple

from memory_module.config import MemoryModuleConfig
from memory_module.core.message_buffer import MessageBuffer
from memory_module.interfaces.base_memory_core import BaseMemoryCore
from memory_module.interfaces.base_message_buffer_storage import (
    BaseMessageBufferStorage,
)
from memory_module.interfaces.base_message_queue import BaseMessageQueue
from memory_module.interfaces.types import (
    Memory,
    Message,
    ShortTermMemoryRetrievalConfig,
)


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
        self.config = config
        self.message_buffer = MessageBuffer(
            config=config,
            process_callback=self._process_for_semantic_messages,
            storage=message_buffer_storage,
        )

    async def enqueue(self, message: Message) -> None:
        """Add a message to the queue for processing.

        Messages are buffered by conversation_ref. When enough messages accumulate,
        they are processed as to extract semantic memories.
        """
        await self.message_buffer.add_message(message)

    async def dequeue(self, message_ids: List[str]) -> None:
        """Remove list of messages from queue"""
        await self.message_buffer.remove_messages(message_ids)

    async def _process_for_semantic_messages(self, messages: List[Message]) -> None:
        """Process a list of messages using the memory core.

        Args:
            messages: List of messages to process
        """
        count = len(messages)
        memories: List[Memory] = []
        if count < self.config.buffer_size:
            # If there are not enough messages in the buffer, pull the difference from the chat history
            oldest = min(messages, key=lambda x: x.created_at)
            diff = self.config.buffer_size - count
            stored_messages, memories = await self._get_recent_messages_and_memories(
                oldest.conversation_ref, before=oldest.created_at, n_messages=diff
            )
            messages = stored_messages + messages

        await self.memory_core.process_semantic_messages(
            messages=messages, existing_memories=memories
        )

    async def _get_recent_messages_and_memories(
        self, conversation_ref: str, before: datetime.datetime, n_messages: int
    ) -> Tuple[List[Message], List[Memory]]:
        """Pull messages from short term memory for a conversation."""
        # The messages in the buffer are a subset of the messages in the chat history
        config = ShortTermMemoryRetrievalConfig(n_messages=n_messages, before=before)
        messages = await self.memory_core.retrieve_chat_history(
            conversation_ref, config=config
        )

        all_memories = []
        for message in messages:
            # Get memories for each message
            memories = await self.memory_core.get_memories_from_message(
                message_id=message.id
            )
            # Aggregate all memories
            all_memories.extend(memories)

        return messages, all_memories

    async def _process_for_episodic_messages(self, messages: List[Message]) -> None:
        """Process a list of messages as episodic memory.

        Args:
            messages: List of messages to process
        """
        await self.memory_core.process_episodic_messages(messages)
