import datetime
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from memory_module.interfaces.base_memory_storage import BaseMemoryStorage
from memory_module.interfaces.base_message_buffer_storage import (
    BaseMessageBufferStorage,
)
from memory_module.interfaces.base_scheduled_events_service import Event
from memory_module.interfaces.base_scheduled_events_storage import BaseScheduledEventsStorage
from memory_module.interfaces.types import BaseMemoryInput, EmbedText, Memory, Message, ShortTermMemoryRetrievalConfig


class InMemoryStorage(BaseMemoryStorage, BaseMessageBufferStorage, BaseScheduledEventsStorage):
    def __init__(self):
        self.storage: Dict = {
            "embeddings": {},
            "buffered_messages": defaultdict(list),
            "scheduled_events": {},
            "memories": {},
        }

    async def store_memory(
        self,
        memory: BaseMemoryInput,
        *,
        embedding_vectors: List[List[float]],
    ) -> int | None:
        memory_id = str(len(self.storage["memories"]) + 1)
        self.storage["memories"][memory_id] = memory
        self.storage["embeddings"][memory_id] = embedding_vectors
        return int(memory_id)

    async def update_memory(self, memory_id: str, updated_memory: str, *, embedding_vectors: List[List[float]]) -> None:
        if memory_id in self.storage["memories"]:
            self.storage["memories"][memory_id].content = updated_memory
            self.storage["embeddings"][memory_id] = embedding_vectors

    async def store_short_term_memory(self, message: Message) -> None:
        await self.store_buffered_message(message)

    async def retrieve_memories(
        self, embedText: EmbedText, user_id: Optional[str], limit: Optional[int] = None
    ) -> List[Memory]:
        limit = limit or self.default_limit
        sorted_memories = []

        for memory_id, embeddings in self.storage["embeddings"].items():
            memory = self.storage["memories"][memory_id]
            if user_id and memory.user_id != user_id:
                continue

            # Find the embedding with highest similarity (lowest distance)
            best_similarity = float("-inf")
            for embedding in embeddings:
                similarity = self._cosine_similarity(embedText.embedding_vector, embedding)
                best_similarity = max(best_similarity, similarity)

            sorted_memories.append(
                {
                    "id": memory_id,
                    "memory": memory,
                    "distance": best_similarity,
                }
            )

        sorted_memories.sort(key=lambda x: x["distance"], reverse=True)
        return [Memory(id=item["id"], **item["memory"].__dict__) for item in sorted_memories[:limit]]

    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
        return [
            self.storage["memories"][memory_id].copy()
            for memory_id in memory_ids
            if memory_id in self.storage["memories"]
        ]

    async def get_messages(self, memory_ids: List[int]) -> Dict[int, List[Message]]:
        messages_dict: Dict[int, List[Message]] = {}
        for memory_id in memory_ids:
            str_id = str(memory_id)
            if str_id in self.storage["memories"]:
                memory = self.storage["memories"][str_id]
                if hasattr(memory, "message_attributions"):
                    messages = []
                    for msg_id in memory.message_attributions:
                        # Search through buffered messages to find matching message
                        for conv_messages in self.storage["buffered_messages"].values():
                            for msg in conv_messages:
                                if msg.id == msg_id:
                                    messages.append(msg)
                    messages_dict[memory_id] = messages
        return messages_dict

    def _cosine_similarity(self, memory_vector: List[float], query_vector: List[float]) -> float:
        return np.dot(np.array(query_vector), np.array(memory_vector))

    async def clear_memories(self, user_id: str) -> None:
        memory_ids_for_user = [
            memory_id for memory_id, memory in self.storage["memories"].items() if memory.user_id == user_id
        ]
        # remove all memories for user
        for memory_id in memory_ids_for_user:
            self.storage["embeddings"].pop(str(memory_id), None)
            self.storage["memories"].pop(str(memory_id), None)

    async def get_memory(self, memory_id: int) -> Optional[Memory]:
        return self.storage["memories"].get(str(memory_id))

    async def get_all_memories(self, limit: Optional[int] = None) -> List[Memory]:
        return [value for key, value in self.storage["memories"].items()][:limit]

    async def store_buffered_message(self, message: Message) -> None:
        """Store a message in the buffer."""
        self.storage["buffered_messages"][message.conversation_ref].append(message)

    async def get_buffered_messages(self, conversation_ref: str) -> List[Message]:
        """Retrieve all buffered messages for a conversation."""
        return self.storage["buffered_messages"][conversation_ref]

    async def clear_buffered_messages(self, conversation_ref: str) -> None:
        """Remove all buffered messages for a conversation."""
        self.storage["buffered_messages"][conversation_ref] = []

    async def count_buffered_messages(self, conversation_ref: str) -> int:
        """Count the number of buffered messages for a conversation."""
        return len(self.storage["buffered_messages"][conversation_ref])

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

    async def retrieve_chat_history(
        self, conversation_ref: str, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        messages = []

        # Get messages for the conversation
        conversation_messages = self.storage["buffered_messages"].get(conversation_ref, [])

        if config.n_messages is not None:
            messages = conversation_messages[-config.n_messages :]
        elif config.last_minutes is not None:
            current_time = datetime.datetime.now()
            messages = [
                msg
                for msg in conversation_messages
                if (current_time - msg.created_at).total_seconds() / 60 <= config.last_minutes
            ]

        # Sort messages in descending order based on created_at
        messages.sort(key=lambda msg: msg.created_at, reverse=True)

        return messages
