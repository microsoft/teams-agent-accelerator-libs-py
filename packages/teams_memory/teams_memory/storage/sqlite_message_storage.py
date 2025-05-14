import datetime
import uuid
from pathlib import Path
from typing import Any, List, Optional

from teams_memory.config import StorageConfig
from teams_memory.interfaces.base_message_storage import BaseMessageStorage
from teams_memory.interfaces.types import InternalMessageInput, Message, MessageInput
from teams_memory.storage.sqlite_storage import SQLiteStorage
from teams_memory.storage.utils import build_message_from_dict

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


class SQLiteMessageStorage(BaseMessageStorage):
    """SQLite implementation of message storage."""

    def __init__(self, config: StorageConfig):
        self.storage = SQLiteStorage(config.db_path or DEFAULT_DB_PATH)

    async def upsert_message(self, message: MessageInput) -> Message:
        if isinstance(message, InternalMessageInput):
            id = str(uuid.uuid4())
        else:
            id = message.id

        if message.created_at:
            created_at = message.created_at
        else:
            created_at = datetime.datetime.now()

        created_at = created_at.astimezone(datetime.timezone.utc)

        if isinstance(message, InternalMessageInput):
            deep_link = None
        else:
            deep_link = message.deep_link
        await self.storage.execute(
            """INSERT OR REPLACE INTO messages (
                id,
                content,
                author_id,
                conversation_ref,
                created_at,
                type,
                deep_link
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                id,
                message.content,
                message.author_id,
                message.conversation_ref,
                created_at,
                message.type,
                deep_link,
            ),
        )

        row = await self.storage.fetch_one("SELECT * FROM messages WHERE id = ?", (id,))
        if not row:
            raise ValueError(f"Message with id {id} not found in storage")
        return build_message_from_dict(row)

    async def get_messages(self, message_ids: List[str]) -> List[Message]:
        if not message_ids:
            return []

        query = f"""
            SELECT *
            FROM messages
            WHERE id IN ({",".join(["?"] * len(message_ids))})
        """

        rows = await self.storage.fetch_all(query, tuple(message_ids))
        return [build_message_from_dict(row) for row in rows]

    async def delete_messages(self, message_ids: List[str]) -> None:
        async with self.storage.transaction() as cursor:
            await cursor.execute(
                f"DELETE FROM messages WHERE id in ({','.join(['?'] * len(message_ids))})",
                tuple(message_ids),
            )

    async def retrieve_conversation_history(
        self,
        conversation_ref: str,
        *,
        n_messages: Optional[int] = None,
        last_minutes: Optional[float] = None,
        before: Optional[datetime.datetime] = None,
    ) -> List[Message]:
        query = "SELECT * FROM messages WHERE conversation_ref = ?"
        params: tuple[Any, ...] = (conversation_ref,)

        if last_minutes is not None:
            cutoff_time = datetime.datetime.now(
                datetime.timezone.utc
            ) - datetime.timedelta(minutes=last_minutes)
            query += " AND created_at >= ?"
            params += (cutoff_time,)

        if before is not None:
            query += " AND created_at < ?"
            params += (before.astimezone(datetime.timezone.utc),)

        query += " ORDER BY created_at DESC"
        if n_messages is not None:
            query += " LIMIT ?"
            params += (str(n_messages),)

        rows = await self.storage.fetch_all(query, params)
        return [build_message_from_dict(row) for row in rows][::-1]
