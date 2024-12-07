from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from memory_module.interfaces.base_memory_storage import BaseMemoryStorage
from memory_module.interfaces.base_message_buffer_storage import (
    BaseMessageBufferStorage,
)
from memory_module.interfaces.base_scheduled_events_service import Event
from memory_module.interfaces.base_scheduled_events_storage import BaseScheduledEventsStorage
from memory_module.interfaces.types import Memory, Message
from memory_module.services.llm_service import LLMService


class InMemoryStorage(BaseMemoryStorage, BaseMessageBufferStorage, BaseScheduledEventsStorage):
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.storage: Dict[str, Dict] = {
            "embeddings": {},
            "buffered_messages": defaultdict(list),  # type: Dict[str, List[Message]]
            "scheduled_events": {},
        }
        self.llm_service = llm_service or LLMService()

    async def store_memory(
        self,
        memory: Memory,
        *,
        embedding_vector: List[float] = None,
    ) -> None:
        self.storage[memory.id] = memory
        self.storage["embeddings"][memory.id] = embedding_vector

    async def retrieve_memories(self, query: str, user_id: str) -> List[Memory]:
        embedding_list = self.llm_service.get_embeddings([query])
        query_vector = np.array(embedding_list[0])
        sorted_collection = sorted(
            self.storage["embeddings"].items(),
            key=lambda record: self._cosine_similarity(record, query_vector),
        )[:3]
        return sorted_collection

    def _cosine_similarity(self, record: Memory, query_vector: list[float]):
        return np.dot(query_vector, np.array(record.embedding_vector))

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
