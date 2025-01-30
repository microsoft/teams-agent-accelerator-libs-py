"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, List, Optional

from teams_memory.interfaces.base_scheduled_events_service import Event


class BaseScheduledEventsStorage(ABC):
    """Abstract base class for persisting scheduled events."""

    @abstractmethod
    async def upsert_event(
        self, id: str, object: Any, time: datetime | timedelta
    ) -> Event:
        """Store a scheduled event.

        If the event already exists, it will only be updated if the time has changed.

        Args:
            id: The ID of the event
            object: The object to store
            time: The time to schedule the event
        """
        pass

    @abstractmethod
    async def get_event(self, event_id: str) -> Optional[Event]:
        """Retrieve a specific event by ID.

        Args:
            event_id: ID of the event to retrieve

        Returns:
            The event if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_event(self, event_id: str) -> None:
        """Delete an event from storage.

        Args:
            event_id: ID of the event to delete
        """
        pass

    @abstractmethod
    async def get_all_events(self) -> List[Event]:
        """Retrieve all stored events.

        Returns:
            List of all stored events
        """
        pass

    @abstractmethod
    async def clear_all_events(self) -> None:
        """Remove all stored events."""
        pass
