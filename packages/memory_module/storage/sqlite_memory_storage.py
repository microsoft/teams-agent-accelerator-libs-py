import datetime
import heapq
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import sqlite_vec
from memory_module.interfaces.base_memory_storage import BaseMemoryStorage
from memory_module.interfaces.types import (
    BaseMemoryInput,
    EmbedText,
    InternalMessageInput,
    Memory,
    Message,
    MessageInput,
    ShortTermMemoryRetrievalConfig,
)
from memory_module.storage.sqlite_storage import SQLiteStorage
from memory_module.storage.utils import build_message_from_dict

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


class SQLiteMemoryStorage(BaseMemoryStorage):
    """SQLite implementation of memory storage."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.storage = SQLiteStorage(self.db_path)

    async def store_memory(self, memory: BaseMemoryInput, *, embedding_vectors: List[List[float]]) -> str:
        """Store a memory and its message attributions."""
        serialized_embeddings = [
            sqlite_vec.serialize_float32(embedding_vector) for embedding_vector in embedding_vectors
        ]

        memory_id = str(uuid.uuid4())

        async with self.storage.transaction() as cursor:
            # Store the memory
            await cursor.execute(
                """INSERT INTO memories
                    (id, content, created_at, updated_at, user_id, memory_type)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    memory_id,
                    memory.content,
                    memory.created_at.isoformat(),
                    memory.updated_at,
                    memory.user_id,
                    memory.memory_type.value,
                ),
            )

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

    async def update_memory(self, updated_memory: Memory, *, embedding_vectors: List[List[float]]) -> None:
        """replace an existing memory with new extracted fact, attributions and embedding"""
        serialized_embeddings = [
            sqlite_vec.serialize_float32(embedding_vector) for embedding_vector in embedding_vectors
        ]

        async with self.storage.transaction() as cursor:
            # Update the memory content
            await cursor.execute(
                "UPDATE memories SET content = ?, updated_at = ?  WHERE id = ?",
                (updated_memory.content, updated_memory.updated_at, updated_memory.id)
            )

            # Remove all existing embeddings from `vec_items` table
            await cursor.execute("""
                DELETE FROM vec_items
                WHERE EXISTS (
                    SELECT id, memory_id
                    FROM embeddings
                    WHERE vec_items.memory_embedding_id = id AND memory_id = ?
                )""",
                (updated_memory.id,)
            )

            # Remove all existing embeddings for `embeddings` table
            await cursor.execute("DELETE FROM embeddings WHERE memory_id = ?", (updated_memory.id,))

            # Add new embeddings in `embeddings` table
            await cursor.executemany(
                "INSERT INTO embeddings (memory_id, embedding) VALUES (?, ?)",
                [(updated_memory.id, serialized_embedding) for serialized_embedding in serialized_embeddings],
            )

            # Add new embeddings in `vec_items` table
            await cursor.execute(
                """
                INSERT INTO vec_items (memory_embedding_id, embedding)
                SELECT id, embedding
                FROM embeddings
                WHERE memory_id = ?
                """,
                (updated_memory.id,),
            )

            # Update memory_attributions
            await cursor.execute(
                "DELETE FROM memory_attributions WHERE memory_id = ?",
                (updated_memory.id,)
            )
            if updated_memory.message_attributions:
                await cursor.executemany(
                    "INSERT INTO memory_attributions (memory_id, message_id) VALUES (?, ?)",
                    [(updated_memory.id, msg_id) for msg_id in updated_memory.message_attributions],
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
                ma.message_id,
                rm.distance
            FROM ranked_memories rm
            JOIN memories m ON m.id = rm.memory_id
            LEFT JOIN memory_attributions ma ON m.id = ma.memory_id
            WHERE m.user_id = ?
            ORDER BY rm.distance ASC
        """

        rows = await self.storage.fetch_all(
            query,
            (
                sqlite_vec.serialize_float32(embedText.embedding_vector),
                limit or self.default_limit,
                1.0,
                user_id
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
                    "distance": row["distance"],
                }

            if row["message_id"]:
                memories_dict[memory_id]["message_attributions"].append(row["message_id"])

        return [Memory(**memory_data) for memory_data in memories_dict.values()]

    async def get_most_similar_memory_with_embeddings(
        self, embeddings: List[List[float]], user_id: Optional[str]
    ) -> Memory:
        candidates = []
        for embedding in embeddings:

            embedText = EmbedText(embedding_vector=embedding, text="")
            similar_memory = await self.retrieve_memories(embedText, user_id, 1)
            if len(similar_memory) >= 1:
                candidates.append(similar_memory[0])
        heapq.heapify(candidates)
        return heapq.heappop(candidates)

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
                m.updated_at,
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
            "updated_at": rows[0]["updated_at"],
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

    async def store_short_term_memory(self, message: MessageInput) -> Message:
        """Store a short-term memory entry."""
        if isinstance(message, InternalMessageInput):
            id = str(uuid.uuid4())
        else:
            id = message.id

        created_at = message.created_at or datetime.datetime.now()

        if isinstance(message, InternalMessageInput):
            deep_link = None
        else:
            deep_link = message.deep_link
        await self.storage.execute(
            """INSERT INTO messages (
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
                created_at.isoformat(),
                message.type,
                deep_link,
            ),
        )

        row = await self.storage.fetch_one("SELECT * FROM messages WHERE id = ?", (id,))
        if not row:
            raise ValueError(f"Message with id {id} not found in storage")
        return build_message_from_dict(row)

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

        return [build_message_from_dict(row) for row in rows][::-1]

    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
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
            WHERE m.id IN ({})
        """.format(",".join(["?"] * len(memory_ids)))

        rows = await self.storage.fetch_all(query, tuple(memory_ids))

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

    async def get_messages(self, memory_ids: List[str]) -> Dict[str, List[Message]]:
        """Get messages based on memory ids."""
        query = """
            SELECT ma.memory_id, m.*
            FROM memory_attributions ma
            JOIN messages m ON ma.message_id = m.id
            WHERE ma.memory_id IN ({})
        """.format(",".join(["?"] * len(memory_ids)))

        rows = await self.storage.fetch_all(query, tuple(memory_ids))

        messages_dict = {}
        for row in rows:
            memory_id = row["memory_id"]
            if memory_id not in messages_dict:
                messages_dict[memory_id] = []

            messages_dict[memory_id].append(build_message_from_dict(row))

        return messages_dict
