import logging
from pathlib import Path
from typing import List, Optional

import sqlite_vec
from memory_module.interfaces.base_memory_storage import BaseMemoryStorage
from memory_module.interfaces.types import EmbedText, Memory, Message, ShortTermMemoryRetrievalConfig
from memory_module.storage.sqlite_storage import SQLiteStorage

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


class SQLiteMemoryStorage(BaseMemoryStorage):
    """SQLite implementation of memory storage."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.storage = SQLiteStorage(self.db_path)

    async def store_memory(self, memory: Memory, *, embedding_vectors: List[List[float]]) -> int | None:
        """Store a memory and its message attributions."""
        serialized_embeddings = [
            sqlite_vec.serialize_float32(embedding_vector) for embedding_vector in embedding_vectors
        ]

        async with self.storage.transaction() as cursor:
            # Store the memory
            await cursor.execute(
                """INSERT INTO memories
                    (content, created_at, user_id, memory_type)
                    VALUES (?, ?, ?, ?)""",
                (
                    memory.content,
                    memory.created_at,
                    memory.user_id,
                    memory.memory_type.value,
                ),
            )

            memory_id = cursor.lastrowid

            # Store message attributions
            if memory.message_attributions:
                await cursor.executemany(
                    "INSERT INTO memory_attributions (memory_id, message_id) VALUES (?, ?)",
                    [(memory_id, msg_id) for msg_id in memory.message_attributions],
                )

            # Store embedding in embeddings table
            await cursor.executemany(
                "INSERT INTO embeddings (memory_id, embedding) VALUES (?, ?)",
                [(memory_id, serialized_embedding) for serialized_embedding in serialized_embeddings],
            )

            await cursor.execute(
                """
                INSERT INTO vec_items (memory_embedding_id, embedding)
                SELECT id, embedding
                FROM embeddings
                WHERE memory_id = ?
                """,
                (memory_id,),
            )
        return memory_id

    async def insert_memory_to_existing_record(
        self, memory_id: str, memory: Memory, *, embedding_vector: List[float]
    ) -> None:
        """Once an async memory extraction process is done, update it with extracted fact and insert embedding"""
        serialized_embedding = sqlite_vec.serialize_float32(embedding_vector)

        async with self.storage.transaction() as cursor:
            # Update the memory content
            await cursor.execute(
                """UPDATE memories
                    SET content = ?
                    WHERE id = ?""",
                (memory.content, memory_id),
            )

            # Store embedding in embeddings table
            await cursor.execute(
                "INSERT INTO embeddings (memory_id, embedding) VALUES (?, ?)",
                (memory_id, serialized_embedding),
            )
            embedding_id = cursor.lastrowid

            # Store in vec_items table
            await cursor.execute(
                "INSERT INTO vec_items (memory_embedding_id, embedding) VALUES (?, ?)",
                (embedding_id, serialized_embedding),
            )

    async def update_memory(self, memory_id: str, updateMemory: str, *, embedding_vector: List[float]) -> None:
        """replace an existing memory with new extracted fact and embedding"""
        serialized_embedding = sqlite_vec.serialize_float32(embedding_vector)

        async with self.storage.transaction() as cursor:
            # Update the memory content
            await cursor.execute("UPDATE memories SET content = ? WHERE id = ?", (updateMemory, memory_id))

            # Update embedding in embeddings table
            await cursor.execute(
                "UPDATE embeddings SET embedding = ? WHERE memory_id = ?",
                (
                    serialized_embedding,
                    memory_id,
                ),
            )
            await cursor.execute("SELECT id FROM embeddings WHERE memory_id = ?", (memory_id,))

            embedding_id = await cursor.fetchone()

            # Update in vec_items table
            await cursor.execute(
                "UPDATE vec_items SET embedding = ? WHERE memory_embedding_id = ?",
                (
                    serialized_embedding,
                    embedding_id[0],
                ),
            )

    async def retrieve_memories(
        self, embedText: EmbedText, user_id: Optional[str], limit: Optional[int] = None
    ) -> List[Memory]:
        """Retrieve memories based on a query."""
        query = """
            WITH ranked_memories AS (
                SELECT
                    e.memory_id,
                    distance
                FROM vec_items
                JOIN embeddings e ON vec_items.memory_embedding_id = e.id
                WHERE vec_items.embedding MATCH ? AND K = ? AND distance < ?
                ORDER BY distance ASC
            )
            SELECT
                m.id,
                m.content,
                m.created_at,
                m.user_id,
                m.memory_type,
                ma.message_id
            FROM ranked_memories rm
            JOIN memories m ON m.id = rm.memory_id
            LEFT JOIN memory_attributions ma ON m.id = ma.memory_id
            ORDER BY rm.distance ASC
        """

        rows = await self.storage.fetch_all(
            query,
            (
                sqlite_vec.serialize_float32(embedText.embedding_vector),
                limit or 3,
                1.0,
            ),
        )

        # Group rows by memory_id to handle message attributions
        memories_dict = {}
        for row in rows:
            memory_id = row["id"]
            if memory_id not in memories_dict:
                memories_dict[memory_id] = {
                    "id": memory_id,
                    "content": row["content"],
                    "created_at": row["created_at"],
                    "user_id": row["user_id"],
                    "memory_type": row["memory_type"],
                    "message_attributions": [],
                }

            if row["message_id"]:
                memories_dict[memory_id]["message_attributions"].append(row["message_id"])

        return [Memory(**memory_data) for memory_data in memories_dict.values()]

    async def clear_memories(self, user_id: str) -> None:
        """Clear all memories for a given user."""
        query = """
            SELECT
                m.id,
                e.id AS embed_id
            FROM memories m
            LEFT JOIN embeddings e
            WHERE m.user_id = ? AND m.id = e.memory_id
        """
        id_rows = await self.storage.fetch_all(query, (user_id,))
        memory_id_list = [row["id"] for row in id_rows]
        embed_id_list = [row["embed_id"] for row in id_rows]

        # Remove memory
        async with self.storage.transaction() as cursor:
            await cursor.execute(
                f"DELETE FROM vec_items WHERE memory_embedding_id in ({",".join(["?"]*len(embed_id_list))})",
                tuple(embed_id_list),
            )

            await cursor.execute(
                f"DELETE FROM embeddings WHERE memory_id in ({",".join(["?"]*len(memory_id_list))})",
                tuple(memory_id_list),
            )

            await cursor.execute(
                f"DELETE FROM memory_attributions WHERE memory_id in ({",".join(["?"]*len(memory_id_list))})",
                tuple(memory_id_list),
            )

            await cursor.execute(
                f"DELETE FROM memories WHERE id in ({",".join(["?"]*len(memory_id_list))})", tuple(memory_id_list)
            )

    async def get_memory(self, memory_id: int) -> Optional[Memory]:
        """Retrieve a memory with its message attributions."""
        query = """
            SELECT
                m.id,
                m.content,
                m.created_at,
                m.user_id,
                m.memory_type,
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
            "memory_type": rows[0]["memory_type"],
            "message_attributions": [row["message_id"] for row in rows if row["message_id"]],
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
                m.memory_type,
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
                    "memory_type": row["memory_type"],
                    "message_attributions": [],
                }

            if row["message_id"]:
                memories_dict[memory_id]["message_attributions"].append(row["message_id"])

        return [Memory(**memory_data) for memory_data in memories_dict.values()]

    async def store_short_term_memory(self, message: Message) -> None:
        """Store a short-term memory entry."""
        async with self.storage.transaction() as cursor:
            await cursor.execute(
                """INSERT INTO messages (id, content, author_id, conversation_ref, created_at, is_assistant_message)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    message.id,
                    message.content,
                    message.author_id,
                    message.conversation_ref,
                    message.created_at,
                    message.is_assistant_message,
                ),
            )

    async def retrieve_chat_history(
        self, conversation_ref: str, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        query = "SELECT * FROM messages WHERE conversation_ref = ?"
        params = [conversation_ref]

        if "n_messages" in config and config["n_messages"] is not None:
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(config["n_messages"])

        if "last_minutes" in config and config["last_minutes"] is not None:
            query += " AND created_at >= datetime('now', ?) ORDER BY created_at DESC"
            params.append(f"-{config['last_minutes']} minutes")

        rows = await self.storage.fetch_all(query, params)

        return [Message(**row) for row in rows][::-1]
