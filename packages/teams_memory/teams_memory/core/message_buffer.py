"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, List, Optional, Set

from teams_memory.config import MemoryModuleConfig
from teams_memory.interfaces.base_memory_storage import (
    BaseMemoryStorage,
)
from teams_memory.interfaces.base_scheduled_events_service import (
    BaseScheduledEventsService,
)
from teams_memory.interfaces.types import Message
from teams_memory.services.scheduled_events_service import ScheduledEventsService

logger = logging.getLogger(__name__)


@dataclass
class MessageBufferScheduledEventObject:
    conversation_ref: str
    first_message_timestamp: datetime


class MessageBuffer:
    """Buffers messages by conversation_ref until reaching a threshold for processing."""

    _enable_automatic_processing: bool = False

    def __init__(
        self,
        config: MemoryModuleConfig,
        process_callback: Callable[[List[Message]], Awaitable[None]],
        storage: BaseMemoryStorage,
        scheduler: Optional[BaseScheduledEventsService] = None,
    ):
        """Initialize the message buffer."""
        self.buffer_size = config.buffer_size
        self.timeout_seconds = config.timeout_seconds
        self._process_callback = process_callback
        self.storage = storage
        self.scheduler = scheduler or ScheduledEventsService(config=config)
        self.scheduler.callback = self._handle_timeout

        # Track conversations being processed
        self._processing: Set[str] = set()

    async def _process_conversation_messages(
        self, conversation_ref: str, originally_scheduled_at: datetime
    ) -> None:
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
        await self._process_conversation_messages(id, time)

    async def initialize(self) -> None:
        """Initialize the message buffer with pre-existing messages"""
        # get all the conversations that have messages in the buffer
        await self.scheduler.initialize()

        self._enable_automatic_processing = True

    async def process_messages(self, conversation_ref: str):
        await self._process_conversation_messages(conversation_ref)
        await self.scheduler.cancel_event(conversation_ref)

    async def add_message(self, message: Message) -> None:
        """Add a message to the buffer and process if threshold reached."""
        if not self._enable_automatic_processing:
            logger.debug(
                "Automatic processing is not enabled, skipping message buffer processing"
            )
            return

        # TODO: Possible race condition here where the count includes messages currently being processed
        # but not yet removed from the buffer. This could cause the timer to not be triggered, but seems like
        # a rare edge case.
        # Check if this is the first message in the conversation
        first_pending_event = next(
            (
                event
                for event in self.scheduler.pending_events
                if event.id == message.conversation_ref
            ),
            None,
        )

        if first_pending_event:
            messages = await self.storage.retrieve_conversation_history(
                conversation_ref=message.conversation_ref,
                after=first_pending_event.created_at,
            )
            count = len(messages)
        else:
            count = 0
        # Check if we've reached the buffer size
        if count >= self.buffer_size:
            await self.process_messages(message.conversation_ref)
        elif count == 0:
            await self.scheduler.add_event(
                id=message.conversation_ref,
                object=MessageBufferScheduledEventObject(
                    conversation_ref=message.conversation_ref,
                    first_message_timestamp=message.created_at,
                ),
                time=datetime.now() + timedelta(seconds=self.timeout_seconds),
            )

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
