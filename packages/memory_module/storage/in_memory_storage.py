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
from memory_module.interfaces.types import EmbedText, Memory, Message, ShortTermMemoryRetrievalConfig


class InMemoryStorage(BaseMemoryStorage, BaseMessageBufferStorage, BaseScheduledEventsStorage):
    def __init__(self):
        self.storage: Dict = {
            "embeddings": {},
            "messages": {},
            "buffered_messages": defaultdict(list),  # type: Dict[str, List[Message]]
            "scheduled_events": {},
        }

    async def store_memory(
        self,
        memory: Memory,
        *,
        embedding_vector: List[List[float]],
    ) -> None:
        self.storage[memory.id] = memory
        self.storage["embeddings"][memory.id] = embedding_vector

    async def update_memory(self, memory: Memory, *, embedding_vectors:List[List[float]]) -> None:
        if memory.id in self.storage:
            self.storage[memory.id].content = memory
            self.storage["embeddings"][memory.id] = embedding_vectors

    async def store_short_term_message(self, message: Message) -> None:
        if message.conversation_ref not in self.storage["messages"]:
            self.storage["messages"][message.conversation_ref] = []
        self.storage["messages"][message.conversation_ref].append(message)

    async def retrieve_memories(
        self, embedText: EmbedText, user_id: Optional[str], limit: Optional[int] = None
    ) -> List[Memory]:
        limit = limit or self.default_limit
        sorted_memories = [
            {
                "id": value.id,
                "distance": self._cosine_similarity(embedText.embedding_vector, self.storage["embeddings"][value.id]),
            }
            for key, value in self.storage.items()
        ]
        sorted_memories = sorted(sorted_memories, key=lambda x: x["distance"], reverse=True)[:limit]
        return [Memory(**{**self.storage[item["id"]], **item}) for item in sorted_memories]

    def _cosine_similarity(self, memory_vector: List[float], query_vector: List[float]) -> float:
        return np.dot(np.array(query_vector), np.array(memory_vector))

    async def clear_memories(self, user_id: str) -> None:
        for _key, value in self.storage.items():
            if value.user_id == user_id:
                self.storage.pop(value.id)

    async def get_memory(self, memory_id: int) -> Optional[Memory]:
        return self.storage[memory_id]

    async def get_all_memories(self, limit: Optional[int] = None) -> List[Memory]:
        return [value for key, value in self.storage.items()][:limit]

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
            current_time = datetime.now()
            messages = [
                msg
                for msg in conversation_messages
                if (current_time - msg.created_at).total_seconds() / 60 <= config.last_minutes
            ]

        # Sort messages in descending order based on created_at
        messages.sort(key=lambda msg: msg.created_at, reverse=True)

        return messages
