import datetime
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import sqlite_vec
from memory_module.interfaces.base_memory_storage import BaseMemoryStorage
from memory_module.interfaces.types import (
    BaseMemoryInput,
    InternalMessageInput,
    Memory,
    Message,
    MessageInput,
    ShortTermMemoryRetrievalConfig,
    TextEmbedding,
    Topic,
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

    async def store_memory(
        self, memory: BaseMemoryInput, *, embedding_vectors: List[TextEmbedding]
    ) -> str:
        """Store a memory and its message attributions."""
        serialized_embeddings = [
            sqlite_vec.serialize_float32(embedding.embedding_vector)
            for embedding in embedding_vectors
        ]

        memory_id = str(uuid.uuid4())

        async with self.storage.transaction() as cursor:
            # Store the memory
            # Convert topics list to comma-separated string if it's a list
            topics_str = (
                ",".join(memory.topics)
                if isinstance(memory.topics, list)
                else memory.topics
            )

            await cursor.execute(
                """INSERT INTO memories
                    (id, content, created_at, user_id, memory_type, topics)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    memory_id,
                    memory.content,
                    memory.created_at.astimezone(datetime.timezone.utc),
                    memory.user_id,
                    memory.memory_type.value,
                    topics_str,
                ),
            )

            # Store message attributions
            if memory.message_attributions:
                await cursor.executemany(
                    "INSERT INTO memory_attributions (memory_id, message_id) VALUES (?, ?)",
                    [(memory_id, msg_id) for msg_id in memory.message_attributions],
                )

            # Store embedding in embeddings table with text
            await cursor.executemany(
                "INSERT INTO embeddings (memory_id, embedding, text) VALUES (?, ?, ?)",
                [
                    (memory_id, serialized_embedding, embedding.text)
                    for serialized_embedding, embedding in zip(
                        serialized_embeddings, embedding_vectors, strict=False
                    )
                ],
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

    async def update_memory(
        self,
        memory_id: str,
        updated_memory: str,
        *,
        embedding_vectors: List[TextEmbedding],
    ) -> None:
        """replace an existing memory with new extracted fact and embedding"""
        serialized_embeddings = [
            sqlite_vec.serialize_float32(embedding.embedding_vector)
            for embedding in embedding_vectors
        ]

        async with self.storage.transaction() as cursor:
            # Update the memory content
            await cursor.execute(
                "UPDATE memories SET content = ? WHERE id = ?",
                (updated_memory, memory_id),
            )

            # remove all the embeddings for this memory
            await cursor.execute(
                "DELETE FROM embeddings WHERE memory_id = ?", (memory_id,)
            )

            # Update embedding in embeddings table with text
            await cursor.executemany(
                "INSERT INTO embeddings (memory_id, embedding, text) VALUES (?, ?, ?)",
                [
                    (memory_id, serialized_embedding, embedding.text)
                    for serialized_embedding, embedding in zip(
                        serialized_embeddings, embedding_vectors, strict=False
                    )
                ],
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

    async def retrieve_memories(
        self,
        *,
        user_id: Optional[str],
        text_embedding: Optional[TextEmbedding] = None,
        topics: Optional[List[Topic]] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        base_query = """
            SELECT
                m.*,
                GROUP_CONCAT(ma.message_id) as message_attributions
                {distance_select}
            FROM memories m
            LEFT JOIN memory_attributions ma ON m.id = ma.memory_id
            {embedding_join}
            WHERE 1=1
            {topic_filter}
            {user_filter}
            GROUP BY m.id
            {order_by}
            {limit_clause}
        """

        params = []

        # Handle embedding search first since its params come first in the query
        embedding_join = ""
        distance_select = ""
        order_by = "ORDER BY m.created_at DESC"

        if text_embedding:
            embedding_join = """
                JOIN (
                    SELECT
                        e.memory_id,
                        e.text,
                        MIN(distance) as distance
                    FROM vec_items
                    JOIN embeddings e ON vec_items.memory_embedding_id = e.id
                    WHERE vec_items.embedding MATCH ? AND K = ? AND distance < ?
                    GROUP BY e.memory_id
                ) rm ON m.id = rm.memory_id
            """
            distance_select = ", rm.distance as _distance, rm.text as _embedding_text"
            order_by = "ORDER BY rm.distance ASC"
            params.extend(
                [
                    sqlite_vec.serialize_float32(text_embedding.embedding_vector),
                    limit or self.default_limit,
                    1.0,
                ]
            )

        # Handle topic and user filters after embedding params
        topic_filter = ""
        if topics:
            # Create a single AND condition with multiple LIKE clauses
            topic_filter = (
                " AND (" + " OR ".join(["m.topics LIKE ?"] * len(topics)) + ")"
            )
            params.extend(f"%{t.name}%" for t in topics)

        user_filter = ""
        if user_id:
            user_filter = "AND m.user_id = ?"
            params.append(user_id)

        # Handle limit last
        limit_clause = ""
        if limit and not text_embedding:  # Only add LIMIT if not using vector search
            limit_clause = "LIMIT ?"
            params.append(limit or self.default_limit)

        query = base_query.format(
            distance_select=distance_select,
            embedding_join=embedding_join,
            topic_filter=topic_filter,
            user_filter=user_filter,
            order_by=order_by,
            limit_clause=limit_clause,
        )

        rows = await self.storage.fetch_all(query, tuple(params))
        return [
            self._build_memory(row, set((row["message_attributions"] or "").split(",")))
            for row in rows
        ]

    async def clear_memories(self, user_id: str) -> None:
        """Clear all memories for a given user."""
        query = """
            SELECT
                m.id
            FROM memories m
            WHERE m.user_id = ?
        """
        rows = await self.storage.fetch_all(query, (user_id,))
        memory_ids = [row["id"] for row in rows]
        # Remove memory
        await self.remove_memories(memory_ids)

    async def get_memory(self, memory_id: int) -> Optional[Memory]:
        query = """
            SELECT
                m.*,
                GROUP_CONCAT(ma.message_id) as message_attributions
            FROM memories m
            LEFT JOIN memory_attributions ma ON m.id = ma.memory_id
            WHERE m.id = ?
            GROUP BY m.id
        """

        row = await self.storage.fetch_one(query, (memory_id,))
        if not row:
            return None

        return self._build_memory(
            row, set((row["message_attributions"] or "").split(","))
        )

    async def get_all_memories(
        self, limit: Optional[int] = None, message_ids: Optional[List[str]] = None
    ) -> List[Memory]:
        """Retrieve all memories with their message attributions."""
        query = """
            SELECT
                m.*,
                GROUP_CONCAT(ma.message_id) as message_attributions
            FROM memories m
            LEFT JOIN memory_attributions ma ON m.id = ma.memory_id
        """
        params: tuple = tuple()
        if message_ids is not None:
            query += " WHERE ma.message_id IN ({})".format(
                ",".join(["?"] * len(message_ids))
            )
            params += tuple(
                message_ids,
            )

        query += """
        GROUP BY m.id
        ORDER BY m.created_at DESC
        """

        if limit is not None:
            query += " LIMIT ?"
            params += (limit,)

        rows = await self.storage.fetch_all(query, params)
        return [
            self._build_memory(row, set((row["message_attributions"] or "").split(",")))
            for row in rows
        ]

    async def store_short_term_memory(self, message: MessageInput) -> Message:
        """Store a short-term memory entry."""
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

    async def retrieve_chat_history(
        self, conversation_ref: str, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        query = "SELECT * FROM messages WHERE conversation_ref = ?"
        params: tuple = (conversation_ref,)

        if config.last_minutes is not None:
            cutoff_time = datetime.datetime.now(
                datetime.timezone.utc
            ) - datetime.timedelta(minutes=config.last_minutes)
            query += " AND created_at >= ?"
            params += (cutoff_time,)

        if config.before is not None:
            query += " AND created_at < ?"
            params += (config.before.astimezone(datetime.timezone.utc),)

        query += " ORDER BY created_at DESC"
        if config.n_messages is not None:
            query += " LIMIT ?"
            params += (str(config.n_messages),)

        rows = await self.storage.fetch_all(query, params)
        return [build_message_from_dict(row) for row in rows][::-1]

    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
        if not memory_ids:
            return []

        query = f"""
            SELECT
                m.*,
                GROUP_CONCAT(ma.message_id) as message_attributions
            FROM memories m
            LEFT JOIN memory_attributions ma ON m.id = ma.memory_id
            WHERE m.id IN ({",".join(["?"] * len(memory_ids))})
        """

        rows = await self.storage.fetch_all(query, tuple(memory_ids))

        return [
            self._build_memory(row, set((row["message_attributions"] or "").split(",")))
            for row in rows
        ]

    async def get_user_memories(self, user_id: str) -> List[Memory]:
        query = """
            SELECT
                m.*,
                GROUP_CONCAT(ma.message_id) as message_attributions
            FROM memories m
            LEFT JOIN memory_attributions ma ON m.id = ma.memory_id
            WHERE m.user_id = ?
            GROUP BY m.id
        """

        rows = await self.storage.fetch_all(query, (user_id,))
        return [
            self._build_memory(row, set((row["message_attributions"] or "").split(",")))
            for row in rows
        ]

    async def get_messages(self, memory_ids: List[str]) -> Dict[str, List[Message]]:
        if not memory_ids:
            return {}

        query = f"""
            SELECT
                ma.memory_id as _memory_id,
                m.*
            FROM memory_attributions ma
            JOIN messages m ON ma.message_id = m.id
            WHERE ma.memory_id IN ({",".join(["?"] * len(memory_ids))})
        """

        rows = await self.storage.fetch_all(query, tuple(memory_ids))

        messages_dict: Dict[str, List[Message]] = {}
        for row in rows:
            memory_id = row["_memory_id"]
            if memory_id not in messages_dict:
                messages_dict[memory_id] = []
            messages_dict[memory_id].append(
                build_message_from_dict(
                    {k: v for k, v in row.items() if not k.startswith("_")}
                )
            )

        return messages_dict

    def _build_memory(
        self, memory_values: dict, message_attributions: set[str]
    ) -> Memory:
        memory_keys = [
            "id",
            "content",
            "created_at",
            "user_id",
            "memory_type",
            "topics",
        ]
        # Convert topics string back to list if it exists
        if memory_values.get("topics"):
            memory_values["topics"] = memory_values["topics"].split(",")
        return Memory(
            **{k: v for k, v in memory_values.items() if k in memory_keys},
            message_attributions=message_attributions,
        )

    async def remove_messages(self, message_ids: List[str]) -> None:
        async with self.storage.transaction() as cursor:
            await cursor.execute(
                f"DELETE FROM messages WHERE id in ({','.join(['?'] * len(message_ids))})",
                tuple(message_ids),
            )

    async def remove_memories(self, memory_ids: List[str]) -> None:
        async with self.storage.transaction() as cursor:
            await cursor.execute(
                """DELETE FROM vec_items WHERE memory_embedding_id in (
                    SELECT id FROM embeddings WHERE memory_id in ({})
                )""".format(
                    ",".join(["?"] * len(memory_ids))
                ),
                tuple(memory_ids),
            )

            await cursor.execute(
                "DELETE FROM embeddings WHERE memory_id in ({})".format(
                    ",".join(["?"] * len(memory_ids))
                ),
                tuple(memory_ids),
            )

            await cursor.execute(
                "DELETE FROM memory_attributions WHERE memory_id in ({})".format(
                    ",".join(["?"] * len(memory_ids))
                ),
                tuple(memory_ids),
            )

            await cursor.execute(
                "DELETE FROM memories WHERE id in ({})".format(
                    ",".join(["?"] * len(memory_ids))
                ),
                tuple(memory_ids),
            )
