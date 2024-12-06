from typing import Awaitable, Callable, List, Optional

from memory_module.interfaces.base_message_buffer_storage import (
    BaseMessageBufferStorage,
)
from memory_module.interfaces.types import Message
from memory_module.storage.sqlite_message_buffer_storage import (
    SQLiteMessageBufferStorage,
)


class MessageBuffer:
    """Buffers messages by conversation_ref until reaching a threshold for processing."""

    def __init__(
        self,
        buffer_size: int,
        process_callback: Callable[[List[Message]], Awaitable[None]],
        storage: Optional[BaseMessageBufferStorage] = None,
    ):
        """Initialize the message buffer.

        Args:
            buffer_size: Number of messages to collect before triggering processing
            process_callback: Async function to call when buffer threshold is reached
            storage: Optional storage implementation for message persistence
        """
        self.buffer_size = buffer_size
        self._process_callback = process_callback
        self.storage = storage if storage is not None else SQLiteMessageBufferStorage()

    async def add_message(self, message: Message) -> None:
        """Add a message to the buffer and process if threshold reached.

        Args:
            message: The message to add to the buffer
        """
        # Store the message
        await self.storage.store_buffered_message(message)

        # Check if we've reached the buffer size
        count = await self.storage.count_buffered_messages(message.conversation_ref)
        if count >= self.buffer_size:
            # Get all messages for this conversation
            messages = await self.storage.get_buffered_messages(message.conversation_ref)
            # Clear the buffer for this conversation
            await self.storage.clear_buffered_messages(message.conversation_ref)
            # Process the messages (now awaiting the callback)
            await self._process_callback(messages)
