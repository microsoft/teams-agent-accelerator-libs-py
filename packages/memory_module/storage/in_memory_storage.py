from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from memory_module.interfaces.base_memory_storage import BaseMemoryStorage
from memory_module.interfaces.base_message_buffer_storage import (
    BaseMessageBufferStorage,
)
from memory_module.interfaces.types import Memory, Message
from memory_module.services.llm_service import LLMService


class InMemoryStorage(BaseMemoryStorage, BaseMessageBufferStorage):
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.storage: Dict[str, Dict] = {
            "embeddings": {},
            "buffered_messages": defaultdict(list),  # type: Dict[str, List[Message]]
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
