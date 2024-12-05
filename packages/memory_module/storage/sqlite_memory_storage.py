import logging
from pathlib import Path
from typing import List, Optional

import aiosqlite

from memory_module.interfaces.base_memory_storage import BaseMemoryStorage
from memory_module.interfaces.types import Memory
from memory_module.storage.sqlite_storage import SQLiteStorage

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


class SQLiteMemoryStorage(BaseMemoryStorage):
    """SQLite implementation of memory storage."""

    @staticmethod
    def ensure_db_folder(db_path: Path) -> None:
        """Create the database folder if it doesn't exist."""
        db_path.parent.mkdir(parents=True, exist_ok=True)

    def __init__(self, db_path: Optional[str | Path] = None):
        if not db_path:
            logger.info(f"No database path provided, using default: {DEFAULT_DB_PATH}")
            self.ensure_db_folder(DEFAULT_DB_PATH)
        self.db_path = db_path or DEFAULT_DB_PATH
        self.storage = SQLiteStorage(self.db_path)

    async def store_memory(self, memory: Memory) -> None:
        """Store a memory and its message attributions."""
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cursor:
                await conn.execute("BEGIN TRANSACTION")
                try:
                    # Store the memory
                    await cursor.execute(
                        "INSERT INTO memories (content, created_at, user_id) VALUES (?, ?, ?)",
                        (memory.content, memory.created_at, memory.user_id),
                    )

                    memory_id = cursor.lastrowid

                    # Store message attributions
                    if memory.message_attributions:
                        await cursor.executemany(
                            "INSERT INTO memory_attributions (memory_id, message_id) VALUES (?, ?)",
                            [
                                (memory_id, msg_id)
                                for msg_id in memory.message_attributions
                            ],
                        )

                    await conn.commit()
                    return memory_id
                except Exception:
                    await conn.rollback()
                    raise

    async def retrieve_memories(
        self, query: str, user_id: str, limit: Optional[int] = None
    ) -> List[Memory]:
        """Retrieve memories based on a query."""
        sql_query = """
            SELECT * FROM memories
            WHERE (user_id = ? OR user_id IS NULL)
            AND content LIKE ?
            ORDER BY created_at DESC
        """
        if limit:
            sql_query += f" LIMIT {limit}"

        results = await self.storage.fetch_all(sql_query, (user_id, f"%{query}%"))

        return [Memory(**row) for row in results]

    async def clear_memories(self, user_id: str) -> None:
        """Clear all memories for a given user."""
        await self.storage.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))

    async def get_memory(self, memory_id: int) -> Optional[Memory]:
        """Retrieve a memory with its message attributions."""
        query = """
            SELECT
                m.id,
                m.content,
                m.created_at,
                m.user_id,
                ma.message_id
            FROM memories m
            LEFT JOIN memory_attributions ma ON m.id = ma.memory_id
            WHERE m.id = ?
        """

        rows = await self.storage.fetch_all(query, (memory_id,))

        if not rows:
            return None

        # First row contains the memory data
        memory_data = {
            "id": rows[0]["id"],
            "content": rows[0]["content"],
            "created_at": rows[0]["created_at"],
            "user_id": rows[0]["user_id"],
            "message_attributions": [
                row["message_id"] for row in rows if row["message_id"]
            ],
        }

        return Memory(**memory_data)

    async def get_all_memories(self, limit: Optional[int] = None) -> List[Memory]:
        """Retrieve all memories with their message attributions."""
        query = """
            SELECT
                m.id,
                m.content,
                m.created_at,
                m.user_id,
                ma.message_id
            FROM memories m
            LEFT JOIN memory_attributions ma ON m.id = ma.memory_id
            ORDER BY m.created_at DESC
        """

        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        else:
            params = ()

        rows = await self.storage.fetch_all(query, params)

        # Group rows by memory_id
        memories_dict = {}
        for row in rows:
            memory_id = row["id"]
            if memory_id not in memories_dict:
                memories_dict[memory_id] = {
                    "id": memory_id,
                    "content": row["content"],
                    "created_at": row["created_at"],
                    "user_id": row["user_id"],
                    "message_attributions": [],
                }

            if row["message_id"]:
                memories_dict[memory_id]["message_attributions"].append(
                    row["message_id"]
                )

        return [Memory(**memory_data) for memory_data in memories_dict.values()]