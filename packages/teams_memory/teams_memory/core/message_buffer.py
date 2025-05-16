"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, List, Optional, Set

from teams_memory.config import (
    InMemoryStorageConfig,
    MemoryModuleConfig,
    SQLiteStorageConfig,
)
from teams_memory.interfaces.base_message_buffer_storage import (
    BaseMessageBufferStorage,
)
from teams_memory.interfaces.base_scheduled_events_service import (
    BaseScheduledEventsService,
)
from teams_memory.interfaces.types import Message
from teams_memory.services.scheduled_events_service import ScheduledEventsService
from teams_memory.storage.in_memory_storage import InMemoryStorage
from teams_memory.storage.sqlite_message_buffer_storage import (
    SQLiteMessageBufferStorage,
)

logger = logging.getLogger(__name__)


class MessageBuffer:
    """Buffers messages by conversation_ref until reaching a threshold for processing."""

    _enable_automatic_processing: bool = False

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
        self.storage = storage or self._build_storage(config)
        self.scheduler = scheduler or ScheduledEventsService(config=config)
        self.scheduler.callback = self._handle_timeout

        # Track conversations being processed
        self._processing: Set[str] = set()

    def _build_storage(self, config: MemoryModuleConfig) -> BaseMessageBufferStorage:
        storage_config = config.get_storage_config("message_buffer")
        if isinstance(storage_config, InMemoryStorageConfig):
            return InMemoryStorage()
        if isinstance(storage_config, SQLiteStorageConfig):
            return SQLiteMessageBufferStorage(storage_config)

        raise ValueError(f"Invalid storage type: {storage_config}")

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
                await self.storage.clear_buffered_messages(
                    conversation_ref, before=latest.created_at
                )
        finally:
            # Always remove from processing set
            self._processing.remove(conversation_ref)

    def _is_processing(self, conversation_ref: str) -> bool:
        """Check if a conversation is currently being processed."""
        return conversation_ref in self._processing

    async def _handle_timeout(self, id: str, object: Any, time: datetime) -> None:
        """Handle a conversation timeout by processing its messages."""
        await self._process_conversation_messages(id)

    async def initialize(self) -> None:
        """Initialize the message buffer with pre-existing messages"""
        # get all the conversations that have messages in the buffer
        buffered_messages_by_conversation = (
            await self.storage.get_earliest_buffered_message()
        )
        for (
            conversation,
            earliest_buffered_message,
        ) in buffered_messages_by_conversation.items():
            expected_timeout_time = earliest_buffered_message.created_at + timedelta(
                seconds=self.timeout_seconds
            )
            current_time = datetime.now(expected_timeout_time.tzinfo)
            time_left_to_expected_time = expected_timeout_time - current_time
            time_left_to_expected_time = max(time_left_to_expected_time, timedelta(0))
            updated_timeout_time = datetime.now() + time_left_to_expected_time
            logger.debug(
                "Initialized buffer for %s with timeout %s",
                conversation,
                updated_timeout_time,
            )
            await self.scheduler.add_event(
                id=conversation,
                object=None,
                time=updated_timeout_time,
            )

        self._enable_automatic_processing = True

    async def process_messages(self, conversation_ref: str) -> None:
        await self._process_conversation_messages(conversation_ref)
        await self.scheduler.cancel_event(conversation_ref)

    async def add_message(self, message: Message) -> None:
        """Add a message to the buffer and process if threshold reached."""
        # Store the message
        await self.storage.store_buffered_message(message)

        if not self._enable_automatic_processing:
            logger.debug(
                "Automatic processing is not enabled, skipping message buffer processing"
            )
            return

        # TODO: Possible race condition here where the count includes messages currently being processed
        # but not yet removed from the buffer. This could cause the timer to not be triggered, but seems like
        # a rare edge case.
        # Check if this is the first message in the conversation
        count = (
            await self.storage.count_buffered_messages([message.conversation_ref])
        )[message.conversation_ref]
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
            await self.process_messages(message.conversation_ref)

    async def remove_messages(self, message_ids: List[str]) -> None:
        """Remove list of messages from buffer if not in processing

        Return:
            remaining message ids that is in progress or already processed
        """
        removed_message_ids = []
        ref_dict = await self.storage.get_conversations_from_buffered_messages(
            message_ids
        )
        if not ref_dict:
            logger.info("no messages in buffer that need to be removed")
            return

        count_list = await self.storage.count_buffered_messages(list(ref_dict.keys()))
        for key, value in ref_dict.items():
            # if the conversation is in processing, leave it to be removed later
            if self._is_processing(key):
                logger.warning(
                    "messages {} cannot be removed since the conversation {} is in processing".format(
                        ",".join(ref_dict[key]), key
                    )
                )
            # if the conversation is not started
            else:
                # clean up scheduler if all messages are removed for the conversation
                if count_list[key] == len(value):
                    await self.scheduler.cancel_event(key)
                    logger.info(
                        "remove conversation {} from buffer since all related messages will be removed".format(
                            key
                        )
                    )
                removed_message_ids += value

        await self.storage.remove_buffered_messages_by_id(removed_message_ids)
        logger.info(
            "messages {} are removed from buffer".format(",".join(removed_message_ids))
        )
        for item in removed_message_ids:
            message_ids.remove(item)

    async def shutdown(self) -> None:
        """Shutdown the message buffer and release resources."""
        if isinstance(self.scheduler, ScheduledEventsService):
            await self.scheduler.cleanup()
