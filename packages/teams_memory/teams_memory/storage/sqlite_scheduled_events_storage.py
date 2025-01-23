"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import List, Optional

from teams_memory.config import StorageConfig
from teams_memory.interfaces.base_scheduled_events_service import Event
from teams_memory.interfaces.base_scheduled_events_storage import (
    BaseScheduledEventsStorage,
)
from teams_memory.storage.sqlite_storage import SQLiteStorage

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


class SQLiteScheduledEventsStorage(BaseScheduledEventsStorage):
    """SQLite implementation of scheduled events storage."""

    def __init__(self, config: StorageConfig):
        """Initialize SQLite scheduled events storage.

        Args:
            db_path: Optional path to the SQLite database file
        """
        self.storage = SQLiteStorage(config.db_path or DEFAULT_DB_PATH)

    async def store_event(self, event: Event) -> None:
        """Store a scheduled event."""
        query = """
            INSERT OR REPLACE INTO scheduled_events
            (id, object, scheduled_time)
            VALUES (?, ?, ?)
        """
        await self.storage.execute(
            query,
            (
                event.id,
                json.dumps(event.object),  # Serialize to JSON
                event.time.astimezone(datetime.timezone.utc),
            ),
        )

    async def get_event(self, event_id: str) -> Optional[Event]:
        """Retrieve a specific event by ID."""
        query = """
            SELECT
                id,
                object,
                scheduled_time as time
            FROM scheduled_events
            WHERE id = ?
        """
        result = await self.storage.fetch_one(query, (event_id,))
        if result:
            result["object"] = json.loads(result["object"])  # Deserialize from JSON
            result["time"] = datetime.datetime.fromisoformat(result["time"])
            return Event(**result)
        return None

    async def get_all_events(self) -> List[Event]:
        """Retrieve all stored events."""
        query = """
            SELECT
                id,
                object,
                scheduled_time as time
            FROM scheduled_events
            ORDER BY scheduled_time ASC
        """
        results = await self.storage.fetch_all(query)
        return [
            Event(
                id=row["id"],
                object=json.loads(row["object"]),  # Deserialize from JSON
                time=datetime.datetime.fromisoformat(row["time"]),
            )
            for row in results
        ]

    async def delete_event(self, event_id: str) -> None:
        """Delete an event from storage."""
        query = "DELETE FROM scheduled_events WHERE id = ?"
        await self.storage.execute(query, (event_id,))

    async def clear_all_events(self) -> None:
        """Remove all stored events."""
        await self.storage.execute("DELETE FROM scheduled_events")
