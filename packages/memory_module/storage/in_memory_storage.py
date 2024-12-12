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
from memory_module.interfaces.types import Memory, Message, ShortTermMemoryRetrievalConfig
from memory_module.services.llm_service import LLMService


class InMemoryStorage(BaseMemoryStorage, BaseMessageBufferStorage, BaseScheduledEventsStorage):
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.storage: Dict = {
            "embeddings": {},
            "buffered_messages": defaultdict(list),  # type: Dict[str, List[Message]]
            "scheduled_events": {},
        }
        self.llm_service = llm_service or LLMService()

    async def store_memory(
        self,
        memory: Memory,
        *,
        embedding_vector: List[float],
    ) -> None:
        self.storage[memory.id] = memory
        self.storage["embeddings"][memory.id] = embedding_vector

    async def retrieve_memories(self, query: str, user_id: str, limit: Optional[int] = None) -> List[Memory]:
        embedding_list = await self.llm_service.embedding([query])
        query_vector = embedding_list.data[0]
        sorted_collection = sorted(
            self.storage.items(), key=lambda memory: self._cosine_similarity(memory, query_vector)
        )[:3]
        return sorted_collection

    def _cosine_similarity(self, memory: Memory, query_vector: List[float]) -> float:
        return np.dot(np.array(query_vector), np.array(self.storage["embeddings"][memory.id]))

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

    async def retrieve_short_term_memories(
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
