"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional

from teams_memory.config import MemoryModuleConfig
from teams_memory.interfaces.base_scheduled_events_service import (
    BaseScheduledEventsService,
    Event,
)
from teams_memory.interfaces.base_scheduled_events_storage import (
    BaseScheduledEventsStorage,
)
from teams_memory.storage.in_memory_storage import InMemoryStorage
from teams_memory.storage.sqlite_scheduled_events_storage import (
    SQLiteScheduledEventsStorage,
)

logger = logging.getLogger(__name__)


class TaskInfo(NamedTuple):
    event: Event
    task: asyncio.Task


class ScheduledEventsService(BaseScheduledEventsService):
    """Default implementation of scheduled events service using asyncio."""

    def __init__(
        self,
        config: MemoryModuleConfig,
        storage: Optional[BaseScheduledEventsStorage] = None,
    ):
        """Initialize the scheduled events service.

        Args:
            config: Memory module configuration
            storage: Optional storage implementation for event persistence
        """
        self._callback_func: Optional[
            Callable[[str, Any, datetime], Awaitable[None]]
        ] = None
        self._tasks: Dict[str, TaskInfo] = {}
        self.storage = storage or self._build_storage(config)

    def _build_storage(self, config: MemoryModuleConfig) -> BaseScheduledEventsStorage:
        if not config.storage or config.storage.storage_type == "in-memory":
            return InMemoryStorage()
        if config.storage.storage_type == "sqlite":
            return SQLiteScheduledEventsStorage(config.storage)

        raise ValueError(f"Invalid storage type: {config.storage.storage_type}")

    async def initialize(self) -> None:
        """Asynchronously initialize tasks from storage."""

        events = await self.storage.get_all_events()
        if events:
            logger.info("Found %d events in storage", len(events))
            for event in events:
                await self._create_task(event)

    @property
    def callback(self) -> Optional[Callable[[str, Any, datetime], Awaitable[None]]]:
        return self._callback_func

    @callback.setter
    def callback(self, value: Callable[[str, Any, datetime], Awaitable[None]]) -> None:
        self._callback_func = value

    @property
    def pending_events(self) -> List[Event]:
        """Get list of pending events from storage."""
        return [
            task_info.event
            for task_info in self._tasks.values()
            if not task_info.task.done()
        ]

    async def add_event(self, id: str, object: Any, time: datetime | timedelta) -> None:
        """Schedule a new event to be executed at the specified time."""
        # Cancel existing task if there is one
        if id in self._tasks and not self._tasks[id].task.done():
            self._tasks[id].task.cancel()

        # Store in persistent storage
        event = await self.storage.upsert_event(id, object, time)

        await self._create_task(event)

    async def _create_task(self, event: Event) -> None:
        """Create a new task for the event."""
        # Calculate delay
        now = datetime.now()
        delay = (event.time - now).total_seconds()
        if delay < 0:
            delay = 0

        if event.id in self._tasks:
            existing_task = self._tasks[event.id].event
            if abs((existing_task.time - event.time).total_seconds()) > 0:
                logger.info("Cancelling existing task for event %s", event.id)
                self._tasks[event.id].task.cancel()
            else:
                logger.debug(
                    "Task already exists for event %s. Skipping rescheduling", event.id
                )
                return

        # Create and store new task
        self._tasks[event.id] = TaskInfo(
            event=event,
            task=asyncio.create_task(self._schedule_event(event, delay), name=event.id),
        )

    async def _schedule_event(self, event: Event, delay: float) -> None:
        """Internal method to handle the scheduling and execution of an event."""
        try:
            await asyncio.sleep(delay)

            # Remove from storage
            await self.storage.delete_event(event.id)

            # Execute callback if set
            if self._callback_func:
                await self._callback_func(event.id, event.object, event.time)

        except asyncio.CancelledError:
            # Clean up if task was cancelled
            await self.storage.delete_event(event.id)

        except Exception as e:
            logger.error("Error scheduling event: %s", str(e))

    async def cancel_event(self, id: str) -> None:
        """Cancel a scheduled event.

        Args:
            id: Unique identifier of the event to cancel
        """
        # Cancel and remove the task if it exists
        if id in self._tasks and not self._tasks[id].task.done():
            self._tasks[id].task.cancel()
            self._tasks.pop(id, None)

        # Remove from storage
        await self.storage.delete_event(id)

    async def cleanup(self) -> None:
        """Clean up pending events when shutting down."""
        pending_tasks = [
            task_info.task
            for task_info in self._tasks.values()
            if not task_info.task.done()
        ]
        if pending_tasks:
            for task in pending_tasks:
                task.cancel()
            try:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
                self._tasks.clear()
            except Exception as e:
                logger.error("Error cleaning up scheduled events: %s", str(e))
