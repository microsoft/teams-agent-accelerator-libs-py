from typing import List, Optional

import numpy as np

from memory_module.interfaces.base_memory_storage import BaseMemoryStorage
from memory_module.interfaces.types import Memory
from memory_module.services.llm_service import LLMService


class InMemoryStorage(BaseMemoryStorage):
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.storage = {}
        self.storage["embeddings"] = {}
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