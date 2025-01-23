"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import datetime
import uuid
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, TypedDict

import numpy as np
from teams_memory.interfaces.base_memory_storage import BaseMemoryStorage
from teams_memory.interfaces.base_message_buffer_storage import (
    BaseMessageBufferStorage,
)
from teams_memory.interfaces.base_scheduled_events_service import Event
from teams_memory.interfaces.base_scheduled_events_storage import (
    BaseScheduledEventsStorage,
)
from teams_memory.interfaces.types import (
    AssistantMessage,
    AssistantMessageInput,
    BaseMemoryInput,
    InternalMessage,
    InternalMessageInput,
    Memory,
    Message,
    MessageInput,
    TextEmbedding,
    Topic,
    UserMessage,
    UserMessageInput,
)


class InMemoryInternalStore(TypedDict):
    memories: Dict[str, Memory]
    embeddings: Dict[str, List[TextEmbedding]]
    buffered_messages: Dict[str, List[Message]]
    scheduled_events: Dict[str, Event]
    messages: Dict[str, List[Message]]


class _MemorySimilarity(NamedTuple):
    memory: Memory
    similarity: float


class InMemoryStorage(
    BaseMemoryStorage, BaseMessageBufferStorage, BaseScheduledEventsStorage
):
    def __init__(self):
        self.storage: InMemoryInternalStore = {
            "embeddings": {},
            "buffered_messages": defaultdict(list),
            "scheduled_events": {},
            "memories": {},
            "messages": defaultdict(list),
        }

    async def store_memory(
        self,
        memory: BaseMemoryInput,
        *,
        embedding_vectors: List[TextEmbedding],
    ) -> str | None:
        memory_id = str(len(self.storage["memories"]) + 1)
        memory_obj = Memory(**memory.model_dump(), id=memory_id)
        self.storage["memories"][memory_id] = memory_obj
        self.storage["embeddings"][memory_id] = embedding_vectors
        return memory_id

    async def update_memory(
        self,
        memory_id: str,
        updated_memory: str,
        *,
        embedding_vectors: List[TextEmbedding],
    ) -> None:
        if memory_id in self.storage["memories"]:
            self.storage["memories"][memory_id].content = updated_memory
            self.storage["embeddings"][memory_id] = embedding_vectors

    async def upsert_message(self, message: MessageInput) -> Message:
        if isinstance(message, InternalMessageInput):
            id = str(uuid.uuid4())
        else:
            id = message.id

        created_at = message.created_at or datetime.datetime.now()

        if isinstance(message, InternalMessageInput):
            deep_link = None
        else:
            deep_link = message.deep_link

        message_obj: Message
        if isinstance(message, UserMessageInput):
            message_obj = UserMessage(
                id=id,
                content=message.content,
                created_at=created_at,
                conversation_ref=message.conversation_ref,
                deep_link=deep_link,
                author_id=message.author_id,
            )
        elif isinstance(message, AssistantMessageInput):
            message_obj = AssistantMessage(
                id=id,
                content=message.content,
                created_at=created_at,
                conversation_ref=message.conversation_ref,
                deep_link=deep_link,
                author_id=message.author_id,
            )
        else:
            message_obj = InternalMessage(
                id=id,
                content=message.content,
                created_at=created_at,
                conversation_ref=message.conversation_ref,
                author_id=message.author_id,
            )

        # Upsert the message into the storage system
        # if message already exists, remove it
        if not self.storage["messages"][message.conversation_ref]:
            self.storage["messages"][message.conversation_ref] = []

        self.storage["messages"][message.conversation_ref] = sorted(
            list(
                filter(
                    lambda message: message.id != message_obj.id,
                    self.storage["messages"][message.conversation_ref],
                )
            )
            + [message_obj],
            key=lambda x: x.created_at,
        )

        return message_obj

    async def search_memories(
        self,
        *,
        user_id: Optional[str],
        text_embedding: Optional[TextEmbedding] = None,
        topics: Optional[List[Topic]] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        limit = limit or self.default_limit
        memories = []

        # Filter memories by user_id and topics first
        filtered_memories = list(self.storage["memories"].values())
        if user_id:
            filtered_memories = [m for m in filtered_memories if m.user_id == user_id]
        if topics:
            filtered_memories = [
                m
                for m in filtered_memories
                if m.topics and any(topic.name in m.topics for topic in topics)
            ]

        # If we have text_embedding, calculate similarities and sort
        if text_embedding:
            sorted_memories: list[_MemorySimilarity] = []

            for memory in filtered_memories:
                embeddings = self.storage["embeddings"].get(memory.id, [])
                if not embeddings:
                    continue

                # Find the embedding with lowest distance
                best_distance = float("inf")
                for embedding in embeddings:
                    distance = self._cosine_distance(
                        text_embedding.embedding_vector, embedding.embedding_vector
                    )
                    best_distance = min(best_distance, distance)

                # Filter based on distance threshold
                if best_distance > 1.0:  # adjust threshold as needed
                    continue

                sorted_memories.append(_MemorySimilarity(memory, best_distance))

            # Sort by distance (ascending)
            sorted_memories.sort(key=lambda x: x.similarity)
            memories = [
                Memory(**item.memory.__dict__) for item in sorted_memories[:limit]
            ]
        else:
            # If no embedding, sort by created_at
            memories = sorted(
                filtered_memories, key=lambda x: x.created_at, reverse=True
            )[:limit]

        return memories

    async def get_memories(
        self, *, memory_ids: Optional[List[str]] = None, user_id: Optional[str] = None
    ) -> List[Memory]:
        """Get memories based on memory ids or user id."""
        if memory_ids is None and user_id is None:
            raise ValueError("Either memory_ids or user_id must be provided")

        memories: List[Memory] = []
        if memory_ids:
            memories.extend(
                self.storage["memories"][memory_id].copy()
                for memory_id in memory_ids
                if memory_id in self.storage["memories"]
            )
        if user_id:
            memories.extend(
                memory.copy()
                for memory in self.storage["memories"].values()
                if memory.user_id == user_id
            )
        return memories

    async def get_messages(self, message_ids: List[str]) -> List[Message]:
        messages = []
        for message_id in message_ids:
            # Search through all conversations for the message
            for conv_messages in self.storage["messages"].values():
                for msg in conv_messages:
                    if msg.id == message_id:
                        messages.append(msg)
                        break
        return messages

    async def delete_messages(self, message_ids: List[str]) -> None:
        for message_id in message_ids:
            self.storage["messages"].pop(message_id, None)

    def _cosine_distance(
        self, memory_vector: List[float], query_vector: List[float]
    ) -> float:
        memory_array = np.array(memory_vector)
        query_array = np.array(query_vector)

        # Compute cosine similarity: dot(a, b) / (norm(a) * norm(b))
        dot_product = np.dot(memory_array, query_array)
        norm_a = np.linalg.norm(memory_array)
        norm_b = np.linalg.norm(query_array)

        # Convert similarity [-1, 1] to distance [0, 2]
        # 1 (most similar) -> 0 (closest distance)
        # -1 (least similar) -> 2 (furthest distance)
        similarity = dot_product / (norm_a * norm_b)
        distance = 1 - similarity  # Convert to distance [0, 2]

        return distance

    async def delete_memories(
        self, *, user_id: Optional[str] = None, memory_ids: Optional[List[str]] = None
    ) -> None:
        if user_id is None and memory_ids is None:
            raise ValueError("Either user_id or memory_ids must be provided")

        memories_to_delete: List[str] = []
        if memory_ids:
            # If we have memory_ids, filter by user_id if provided
            memories_to_delete.extend(
                memory_id
                for memory_id in memory_ids
                if memory_id in self.storage["memories"]
                and (
                    user_id is None
                    or self.storage["memories"][memory_id].user_id == user_id
                )
            )
        elif user_id:
            # If we only have user_id, get all memories for that user
            memories_to_delete.extend(
                memory_id
                for memory_id, memory in self.storage["memories"].items()
                if memory.user_id == user_id
            )

        # Remove matching memories and their embeddings
        for memory_id in memories_to_delete:
            self.storage["embeddings"].pop(memory_id, None)
            self.storage["memories"].pop(memory_id, None)

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        return self.storage["memories"].get(memory_id)

    async def get_all_memories(
        self, limit: Optional[int] = None, message_ids: Optional[List[str]] = None
    ) -> List[Memory]:
        memories = [value for key, value in self.storage["memories"].items()]

        if limit is not None:
            memories = memories[:limit]

        if message_ids is not None:
            memories = [
                memory
                for memory in memories
                if memory.message_attributions is not None
                and len(
                    np.intersect1d(
                        np.array(message_ids),
                        np.array(list(memory.message_attributions)),
                    ).tolist()
                )
                > 0
            ]

        return memories

    async def store_buffered_message(self, message: Message) -> None:
        """Store a message in the buffer."""
        self.storage["buffered_messages"][message.conversation_ref].append(message)

    async def get_buffered_messages(self, conversation_ref: str) -> List[Message]:
        """Retrieve all buffered messages for a conversation."""
        return self.storage["buffered_messages"][conversation_ref]

    async def get_conversations_from_buffered_messages(
        self, message_ids: List[str]
    ) -> Dict[str, List[str]]:
        ref_dict: Dict[str, List[str]] = {}
        for key, value in self.storage["buffered_messages"].items():
            stored_message_ids = [item.id for item in value]
            common_message_ids: List[str] = np.intersect1d(np.array(message_ids), np.array(stored_message_ids)).tolist()  # type: ignore
            if len(common_message_ids) > 0:
                ref_dict[key] = common_message_ids

        return ref_dict

    async def clear_buffered_messages(
        self, conversation_ref: str, before: Optional[datetime.datetime] = None
    ) -> None:
        """Remove all buffered messages for a conversation. If the before parameter is provided,
        only messages created on or before that time will be removed."""
        messages = self.storage["buffered_messages"][conversation_ref]
        if before:
            self.storage["buffered_messages"][conversation_ref] = [
                msg for msg in messages if msg.created_at > before
            ]
        else:
            self.storage["buffered_messages"][conversation_ref] = []

    async def remove_buffered_messages_by_id(self, message_ids: List[str]) -> None:
        """Remove list of messages in buffered storage"""
        for key, value in self.storage["buffered_messages"].items():
            self.storage["buffered_messages"][key] = [
                item for item in value if item.id not in message_ids
            ]

    async def count_buffered_messages(
        self, conversation_refs: List[str]
    ) -> Dict[str, int]:
        """Count the number of buffered messages for a conversation."""
        count_dict: Dict[str, int] = {}
        for ref in conversation_refs:
            count_dict[ref] = len(self.storage["buffered_messages"][ref])
        return count_dict

    async def store_event(self, event: Event) -> None:
        """Store a scheduled event."""
        self.storage["scheduled_events"][event.id] = event

    async def get_event(self, event_id: str) -> Optional[Event]:
        """Retrieve a specific event by ID."""
        return self.storage["scheduled_events"].get(event_id)

    async def delete_event(self, event_id: str) -> None:
        """Delete an event from storage."""
        self.storage["scheduled_events"].pop(event_id, None)

    async def get_all_events(self) -> List[Event]:
        """Retrieve all stored events."""
        return list(self.storage["scheduled_events"].values())

    async def clear_all_events(self) -> None:
        """Remove all stored events."""
        self.storage["scheduled_events"].clear()

    async def cancel_event(self, id: str) -> None:
        """Cancel a scheduled event.

        Args:
            id: Unique identifier of the event to cancel
        """
        await self.delete_event(id)

    async def retrieve_conversation_history(
        self,
        conversation_ref: str,
        *,
        n_messages: Optional[int] = None,
        last_minutes: Optional[float] = None,
        before: Optional[datetime.datetime] = None,
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        messages = []

        # Get messages for the conversation
        conversation_messages = self.storage["messages"].get(conversation_ref, [])

        if n_messages is not None:
            messages = conversation_messages[-n_messages:]
        elif last_minutes is not None:
            current_time = datetime.datetime.now()
            messages = [
                msg
                for msg in conversation_messages
                if (current_time - msg.created_at).total_seconds() / 60 <= last_minutes
            ]

        # Sort messages in descending order based on created_at
        messages.sort(key=lambda msg: msg.created_at, reverse=True)

        # Filter messages based on before
        if before is not None:
            messages = [msg for msg in messages if msg.created_at < before]

        return messages
