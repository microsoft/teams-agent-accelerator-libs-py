from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, List, Optional, Set

from memory_module.config import MemoryModuleConfig
from memory_module.interfaces.base_message_buffer_storage import BaseMessageBufferStorage
from memory_module.interfaces.base_scheduled_events_service import BaseScheduledEventsService
from memory_module.interfaces.types import Message
from memory_module.services.scheduled_events_service import ScheduledEventsService
from memory_module.storage.in_memory_storage import InMemoryStorage
from memory_module.storage.sqlite_message_buffer_storage import SQLiteMessageBufferStorage


class MessageBuffer:
    """Buffers messages by conversation_ref until reaching a threshold for processing."""

    def __init__(
        self,
        config: MemoryModuleConfig,
        process_callback: Callable[[List[Message]], Awaitable[None]],
        storage: Optional[BaseMessageBufferStorage] = None,
        scheduler: Optional[BaseScheduledEventsService] = None,
    ):
        """Initialize the message buffer."""
        self.buffer_size = config.buffer_size
        self.timeout_seconds = config.timeout_seconds
        self._process_callback = process_callback
        self.storage = storage or (
            SQLiteMessageBufferStorage(db_path=config.db_path) if config.db_path is not None else InMemoryStorage()
        )
        self.scheduler = scheduler or ScheduledEventsService(config=config)
        self.scheduler.callback = self._handle_timeout

        # Track conversations being processed
        self._processing: Set[str] = set()

    async def _process_conversation_messages(self, conversation_ref: str) -> None:
        """Process all messages for a conversation and clear its buffer.

        Args:
            conversation_ref: The conversation reference to process
        """
        # Skip if already being processed
        if self._is_processing(conversation_ref):
            return

        try:
            self._processing.add(conversation_ref)
            messages = await self.storage.get_buffered_messages(conversation_ref)
            if messages:  # Only process if there are messages
                latest = messages[-1]
                await self._process_callback(messages)
                await self.storage.clear_buffered_messages(conversation_ref, before=latest.created_at)
        finally:
            # Always remove from processing set
            self._processing.remove(conversation_ref)

    def _is_processing(self, conversation_ref: str) -> bool:
        """Check if a conversation is currently being processed."""
        return conversation_ref in self._processing

    async def _handle_timeout(self, id: str, object: Any, time: datetime) -> None:
        """Handle a conversation timeout by processing its messages."""
        await self._process_conversation_messages(id)

    async def add_message(self, message: Message) -> None:
        """Add a message to the buffer and process if threshold reached."""
        # Store the message
        await self.storage.store_buffered_message(message)

        # TODO: Possible race condition here where the count includes messages currently being processed
        # but not yet removed from the buffer. This could cause the timer to not be triggered, but seems like
        # a rare edge case.
        # Check if this is the first message in the conversation
        count = await self.storage.count_buffered_messages(message.conversation_ref)
        if count == 1:
            # Start timeout for this conversation
            timeout_time = datetime.now() + timedelta(seconds=self.timeout_seconds)
            await self.scheduler.add_event(
                id=message.conversation_ref,
                object=None,
                time=timeout_time,
            )

        # Check if we've reached the buffer size
        if count >= self.buffer_size:
            await self._process_conversation_messages(message.conversation_ref)
            await self.scheduler.cancel_event(message.conversation_ref)

    async def flush(self) -> None:
        """Flush all messages from the buffer."""
        if isinstance(self.scheduler, ScheduledEventsService):
            await self.scheduler.flush()
