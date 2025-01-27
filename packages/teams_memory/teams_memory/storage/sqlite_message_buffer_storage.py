"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from teams_memory.config import StorageConfig
from teams_memory.interfaces.base_message_buffer_storage import (
    BaseMessageBufferStorage,
    BufferedMessage,
)
from teams_memory.interfaces.types import Message
from teams_memory.storage.sqlite_storage import SQLiteStorage
from teams_memory.storage.utils import build_message_from_dict

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


class SQLiteMessageBufferStorage(BaseMessageBufferStorage):
    """SQLite implementation of message buffer storage."""

    def __init__(self, config: StorageConfig):
        """Initialize SQLite message buffer storage.

        Args:
            db_path: Optional path to the SQLite database file
        """
        self.storage = SQLiteStorage(config.db_path or DEFAULT_DB_PATH)

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

    async def get_conversations_from_buffered_messages(
        self, message_ids: List[str]
    ) -> Dict[str, List[str]]:
        """Get conversation - buffered messages map based on message ids"""
        query = """
            SELECT
                message_id, conversation_ref
            FROM buffered_messages
            WHERE message_id IN ({})
        """.format(
            ",".join(["?"] * len(message_ids))
        )
        results = await self.storage.fetch_all(query, tuple(message_ids))

        ref_dict: Dict[str, List[str]] = {}
        for result in results:
            ref = result["conversation_ref"]
            if ref not in ref_dict:
                ref_dict[ref] = []

            ref_dict[ref].append(result["message_id"])

        return ref_dict

    async def clear_buffered_messages(
        self, conversation_ref: str, before: Optional[datetime.datetime] = None
    ) -> None:
        """Remove all buffered messages for a conversation. If the before parameter is provided,
        only messages created on or before that time will be removed."""
        query = """
            DELETE FROM buffered_messages
            WHERE conversation_ref = ?
        """
        params: tuple[Any, ...] = (conversation_ref,)
        if before:
            query += " AND created_at <= ?"
            params += (before.astimezone(datetime.timezone.utc),)
        await self.storage.execute(query, params)

    async def remove_buffered_messages_by_id(self, message_ids: List[str]) -> None:
        """Remove list of messages in buffered storage"""
        query = """
            DELETE
            FROM buffered_messages
            WHERE message_id IN ({})
        """.format(
            ",".join(["?"] * len(message_ids))
        )
        await self.storage.execute(query, tuple(message_ids))

    async def count_buffered_messages(
        self, conversation_refs: List[str]
    ) -> Dict[str, int]:
        """Count the number of buffered messages for a conversation."""
        query = """
            SELECT conversation_ref, COUNT(*) as count
            FROM buffered_messages
            WHERE conversation_ref IN ({})
            GROUP BY conversation_ref
        """.format(
            ",".join(["?"] * len(conversation_refs))
        )
        results = await self.storage.fetch_all(query, tuple(conversation_refs))

        count_dict: Dict[str, int] = {}
        for result in results:
            count_dict[result["conversation_ref"]] = result["count"]

        return count_dict

    async def get_earliest_buffered_message(
        self, conversation_refs: Optional[List[str]] = None
    ) -> Dict[str, BufferedMessage]:
        """Get the earliest buffered message for a conversation or all conversations if None provided"""
        params: tuple = ()
        query = """
            SELECT
                message_id, conversation_ref, created_at
            FROM buffered_messages
        """

        if conversation_refs:
            query += " WHERE conversation_ref IN ({})".format(
                ",".join(["?"] * len(conversation_refs))
            )
            params += tuple(conversation_refs)

        query += " GROUP BY conversation_ref ORDER BY created_at ASC"
        results = await self.storage.fetch_all(query, params)

        return {
            result["conversation_ref"]: BufferedMessage(**result) for result in results
        }
