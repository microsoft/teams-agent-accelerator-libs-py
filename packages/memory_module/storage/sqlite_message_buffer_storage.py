import logging
from pathlib import Path
from typing import List, Optional

from memory_module.interfaces.base_message_buffer_storage import (
    BaseMessageBufferStorage,
)
from memory_module.interfaces.types import Message
from memory_module.storage.sqlite_storage import SQLiteStorage

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


class SQLiteMessageBufferStorage(BaseMessageBufferStorage):
    """SQLite implementation of message buffer storage."""

    @staticmethod
    def ensure_db_folder(db_path: Path) -> None:
        """Create the database folder if it doesn't exist."""
        db_path.parent.mkdir(parents=True, exist_ok=True)

    def __init__(self, db_path: Optional[str | Path] = None):
        """Initialize SQLite message buffer storage.

        Args:
            db_path: Optional path to the SQLite database file
        """
        if not db_path:
            logger.info(f"No database path provided, using default: {DEFAULT_DB_PATH}")
            self.ensure_db_folder(DEFAULT_DB_PATH)
        self.db_path = db_path or DEFAULT_DB_PATH
        self.storage = SQLiteStorage(self.db_path)

    async def store_buffered_message(self, message: Message) -> None:
        """Store a message in the buffer."""
        query = """
            INSERT INTO buffered_messages 
            (message_id, content, author_id, conversation_ref, created_at)
            VALUES (?, ?, ?, ?, ?)
        """
        await self.storage.execute(
            query,
            (
                message.id,
                message.content,
                message.author_id,
                message.conversation_ref,
                message.created_at,
            ),
        )

    async def get_buffered_messages(self, conversation_ref: str) -> List[Message]:
        """Retrieve all buffered messages for a conversation."""
        query = """
            SELECT 
                message_id as id,
                content,
                author_id,
                conversation_ref,
                created_at
            FROM buffered_messages 
            WHERE conversation_ref = ?
            ORDER BY created_at ASC
        """
        results = await self.storage.fetch_all(query, (conversation_ref,))
        return [Message(**row) for row in results]

    async def clear_buffered_messages(self, conversation_ref: str) -> None:
        """Remove all buffered messages for a conversation."""
        query = "DELETE FROM buffered_messages WHERE conversation_ref = ?"
        await self.storage.execute(query, (conversation_ref,))

    async def count_buffered_messages(self, conversation_ref: str) -> int:
        """Count the number of buffered messages for a conversation."""
        query = """
            SELECT COUNT(*) as count 
            FROM buffered_messages 
            WHERE conversation_ref = ?
        """
        result = await self.storage.fetch_one(query, (conversation_ref,))
        return result["count"] if result else 0
