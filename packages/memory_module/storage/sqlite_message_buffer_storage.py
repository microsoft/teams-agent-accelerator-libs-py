import datetime
import logging
from pathlib import Path
from typing import List, Optional

from memory_module.interfaces.base_message_buffer_storage import (
    BaseMessageBufferStorage,
)
from memory_module.interfaces.types import Message
from memory_module.storage.sqlite_storage import SQLiteStorage
from memory_module.storage.utils import build_message_from_dict

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


class SQLiteMessageBufferStorage(BaseMessageBufferStorage):
    """SQLite implementation of message buffer storage."""

    def __init__(self, db_path: Optional[str | Path] = None):
        """Initialize SQLite message buffer storage.

        Args:
            db_path: Optional path to the SQLite database file
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.storage = SQLiteStorage(self.db_path)

    async def store_buffered_message(self, message: Message) -> None:
        """Store a message in the buffer."""
        query = """
            INSERT INTO buffered_messages
            (message_id, conversation_ref, created_at)
            VALUES (?, ?, ?)
        """
        await self.storage.execute(
            query,
            (
                message.id,
                message.conversation_ref,
                message.created_at.astimezone(datetime.timezone.utc),
            ),
        )

    async def get_buffered_messages(self, conversation_ref: str) -> List[Message]:
        """Retrieve all buffered messages for a conversation."""
        query = """
            SELECT
                m.*
            FROM buffered_messages b
            JOIN messages m ON b.message_id = m.id
            WHERE b.conversation_ref = ?
            ORDER BY b.created_at ASC
        """
        results = await self.storage.fetch_all(query, (conversation_ref,))
        return [build_message_from_dict(row) for row in results]

    async def clear_buffered_messages(self, conversation_ref: str) -> None:
        """Remove all buffered messages for a conversation."""
        query = "DELETE FROM buffered_messages WHERE conversation_ref = ?"
        await self.storage.execute(query, (conversation_ref,))

    async def remove_buffered_messages_by_id(self, message_ids: List[str]) -> None:
        """Remove list of messages in buffered storage"""
        query = """
            DELETE
            FROM buffered_messages
            WHERE message_id IN ({})
        """.format(",".join(["?"] * len(message_ids)))
        await self.storage.execute(query, tuple(message_ids))

    async def count_buffered_messages(self, conversation_ref: str) -> int:
        """Count the number of buffered messages for a conversation."""
        query = """
            SELECT COUNT(*) as count
            FROM buffered_messages
            WHERE conversation_ref = ?
        """
        result = await self.storage.fetch_one(query, (conversation_ref,))
        return result["count"] if result else 0
