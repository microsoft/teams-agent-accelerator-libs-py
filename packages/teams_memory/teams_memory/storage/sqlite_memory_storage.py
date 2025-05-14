"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import datetime
import logging
import uuid
from pathlib import Path
from typing import Any, List, Optional

import sqlite_vec
from teams_memory.config import StorageConfig
from teams_memory.interfaces.base_memory_storage import BaseMemoryStorage
from teams_memory.interfaces.types import (
    BaseMemoryInput,
    Memory,
    TextEmbedding,
)
from teams_memory.storage.sqlite_storage import SQLiteStorage

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


class SQLiteMemoryStorage(BaseMemoryStorage):
    """SQLite implementation of memory storage."""

    def __init__(self, config: StorageConfig):
        self.storage = SQLiteStorage(config.db_path or DEFAULT_DB_PATH)

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

    async def search_memories(
        self,
        *,
        user_id: Optional[str],
        text_embedding: Optional[TextEmbedding] = None,
        topics: Optional[List[str]] = None,
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
            params.extend(f"%{t}%" for t in topics)

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

    async def delete_memories(
        self, *, user_id: Optional[str] = None, memory_ids: Optional[List[str]] = None
    ) -> None:
        """Delete memories based on user_id and/or memory_ids."""
        if user_id is None and memory_ids is None:
            raise ValueError("Either user_id or memory_ids must be provided")

        conditions = []
        params = []

        if memory_ids:
            conditions.append(f"m.id IN ({','.join(['?'] * len(memory_ids))})")
            params.extend(memory_ids)

        if user_id:
            conditions.append("m.user_id = ?")
            params.append(user_id)

        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT m.id
            FROM memories m
            WHERE {where_clause}
        """

        rows = await self.storage.fetch_all(query, tuple(params))
        memories_to_delete = [row["id"] for row in rows]

        if memories_to_delete:
            await self._delete_memories(memories_to_delete)

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

    async def get_attributed_memories(self, message_ids: List[str]) -> List[Memory]:
        """Retrieve all memories with their message attributions."""
        if not message_ids:
            return []

        query = f"""
            SELECT
                m.*,
                GROUP_CONCAT(ma.message_id) as message_attributions
            FROM memories m
            LEFT JOIN memory_attributions ma ON m.id = ma.memory_id
            WHERE ma.message_id IN ({",".join(["?"] * len(message_ids))})
            GROUP BY m.id
            ORDER BY m.created_at DESC
        """

        rows = await self.storage.fetch_all(query, tuple(message_ids))
        return [
            self._build_memory(row, set((row["message_attributions"] or "").split(",")))
            for row in rows
        ]

    async def _delete_memories(self, memory_ids: List[str]) -> None:
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

    def _build_memory(
        self, memory_values: dict[str, Any], message_attributions: set[str]
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

    async def get_memories(
        self,
        *,
        memory_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> List[Memory]:
        """Get memories based on memory ids or user id."""
        if memory_ids is None and user_id is None:
            raise ValueError("Either memory_ids or user_id must be provided")

        conditions = []
        params = []

        if memory_ids:
            conditions.append(f"m.id IN ({','.join(['?'] * len(memory_ids))})")
            params.extend(memory_ids)

        if user_id:
            conditions.append("m.user_id = ?")
            params.append(user_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT
                m.*,
                GROUP_CONCAT(ma.message_id) as message_attributions
            FROM memories m
            LEFT JOIN memory_attributions ma ON m.id = ma.memory_id
            WHERE {where_clause}
            GROUP BY m.id
        """

        rows = await self.storage.fetch_all(query, tuple(params))
        return [
            self._build_memory(row, set((row["message_attributions"] or "").split(",")))
            for row in rows
        ]
