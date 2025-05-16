try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.aio._search_client_async import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )
    from azure.search.documents.models import VectorizedQuery
except ModuleNotFoundError as e:
    raise ImportError(
        "The 'azure-search-documents' package is required for this feature. "
        "Install it with: pip install '.[azure-search]'"
    ) from e

import logging
import uuid
from typing import List, Optional

from teams_memory.config import AzureAISearchStorageConfig
from teams_memory.interfaces.base_memory_storage import BaseMemoryStorage
from teams_memory.interfaces.types import BaseMemoryInput, Memory, TextEmbedding

logger = logging.getLogger(__name__)


class AzureAISearchMemoryStorage(BaseMemoryStorage):
    def __init__(self, config: AzureAISearchStorageConfig):
        self.config = config
        self.create_memories_index()
        self.client = SearchClient(
            endpoint=self.config.endpoint,
            index_name=self.config.index_name,
            credential=AzureKeyCredential(self.config.api_key),
        )

    def create_memories_index(self) -> None:
        """Create the Azure AI Search index for memories if it does not exist."""
        credential = AzureKeyCredential(self.config.api_key)
        client = SearchIndexClient(self.config.endpoint, credential)

        if self.index_exists(client, self.config.index_name):
            logger.debug("Index '%s' already exists.", self.config.index_name)
            return

        # Define the vector search configuration
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-1",
                    kind="hnsw",
                    metric="cosine",
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile-1", algorithm_configuration_name="hnsw-1"
                )
            ],
        )

        # Define the index schema
        fields = [
            SimpleField(
                name="id", type=SearchFieldDataType.String, key=True, filterable=True
            ),
            SearchableField(
                name="content", type=SearchFieldDataType.String, retrievable=True
            ),
            SimpleField(
                name="created_at",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="user_id", type=SearchFieldDataType.String, filterable=True
            ),
            SimpleField(
                name="memory_type", type=SearchFieldDataType.String, filterable=True
            ),
            SearchableField(
                name="topics",
                type=SearchFieldDataType.String,
                retrievable=True,
                filterable=True,
                facetable=True,
                collection=True,
            ),
            SearchableField(
                name="message_attributions",
                type=SearchFieldDataType.String,
                retrievable=True,
                filterable=True,
                collection=True,
            ),
            # Vector field (use SearchField, not SimpleField)
            SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.config.embedding_dimensions,
                vector_search_profile_name="vector-profile-1",
            ),
        ]

        index = SearchIndex(
            name=self.config.index_name, fields=fields, vector_search=vector_search
        )

        # Create or update the index
        try:
            client.create_index(index)
            logger.debug("Index '%s' created.", self.config.index_name)
        except Exception as e:
            if "already exists" in str(e):
                logger.debug("Index '%s' already exists.", self.config.index_name)
            else:
                raise

    def delete_memories_index(self) -> None:
        credential = AzureKeyCredential(self.config.api_key)
        client = SearchIndexClient(self.config.endpoint, credential)
        try:
            client.delete_index(self.config.index_name)
            logger.debug("Index '%s' deleted.", self.config.index_name)
        except Exception as e:
            if "not found" in str(e):
                logger.debug("Index '%s' does not exist.", self.config.index_name)
            else:
                raise

    @staticmethod
    def index_exists(client: SearchIndexClient, index_name: str) -> bool:
        try:
            client.get_index(index_name)
            return True
        except Exception:
            return False

    async def store_memory(
        self,
        memory: BaseMemoryInput,
        *,
        embedding_vectors: List[TextEmbedding],
    ) -> str | None:
        id = str(uuid.uuid4())
        doc = {
            "id": id,
            "content": memory.content,
            "created_at": memory.created_at,
            "user_id": memory.user_id,
            "memory_type": memory.memory_type.value,
            "topics": memory.topics,
            "message_attributions": (
                list(memory.message_attributions)
                if memory.message_attributions is not None
                else None
            ),
            "contentVector": (
                embedding_vectors[0].embedding_vector if embedding_vectors else None
            ),
        }
        logger.debug("Uploading document to Azure AI Search: %s", doc)
        result = await self.client.upload_documents(documents=[doc])
        logger.debug("Successfully uploaded document to Azure AI Search")
        if result[0].succeeded:
            return id
        return None

    async def update_memory(
        self,
        memory_id: str,
        updated_memory: str,
        *,
        embedding_vectors: List[TextEmbedding],
    ) -> None:
        doc = {
            "id": memory_id,
            "content": updated_memory,
            "contentVector": (
                embedding_vectors[0].embedding_vector if embedding_vectors else None
            ),
        }
        result = await self.client.merge_documents(documents=[doc])
        if not result[0].succeeded:
            raise ValueError(f"Failed to update memory with id {memory_id}")

    async def get_memories(
        self,
        *,
        memory_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> List[Memory]:
        filters = []
        if memory_ids:
            filters.append(
                "(" + " or ".join([f"id eq {repr(mid)}" for mid in memory_ids]) + ")"
            )
        if user_id:
            filters.append(f"user_id eq '{user_id}'")
        filter_str = " and ".join(filters) if filters else None
        results = self.client.search(search_text="*", filter=filter_str, top=1000)
        memories = []
        async for result in await results:
            memories.append(Memory(**result))
        return memories

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        result = await self.client.get_document(key=memory_id)
        return Memory(**result) if result else None

    async def get_attributed_memories(self, message_ids: List[str]) -> List[Memory]:
        if not message_ids:
            return []
        filter_str = " or ".join(
            [f"message_attributions/any(m: m eq '{mid}')" for mid in message_ids]
        )
        results = self.client.search(
            search_text="*", filter=filter_str, order_by=["created_at desc"], top=1000
        )
        memories = []
        async for result in await results:
            memories.append(Memory(**result))
        return memories

    async def search_memories(
        self,
        *,
        user_id: Optional[str],
        text_embedding: Optional[TextEmbedding] = None,
        topics: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        logger.debug(
            "Searching memories with user_id: %s, topics: %s, limit: %s",
            user_id,
            topics,
            limit,
        )
        try:
            filters = []
            if user_id:
                filters.append(f"user_id eq '{user_id}'")
            if topics:
                topic_filters = [f"topics/any(t: t eq '{topic}')" for topic in topics]
                filters.append(" or ".join(topic_filters))
            filter_str = " and ".join(filters) if filters else None
            select = [
                "id",
                "content",
                "created_at",
                "user_id",
                "memory_type",
                "topics",
                "message_attributions",
            ]
            if text_embedding:
                vector_query = VectorizedQuery(
                    vector=text_embedding.embedding_vector,
                    k_nearest_neighbors=limit or 10,
                    exhaustive=True,
                    fields="contentVector",
                )
                results = self.client.search(
                    # search_text=text_embedding.text,
                    vector_queries=[vector_query],
                    select=select,
                    filter=filter_str,
                )
            else:
                results = self.client.search(
                    search_text="*",
                    include_total_count=True,
                    select=select,
                    filter=filter_str,
                    top=limit or 10,
                )
            memories = []
            async for result in await results:
                logger.debug("Search Result: %s", result)
                memories.append(Memory(**result))
            return memories
        except Exception as e:
            logger.debug("Error: %s", e)
            raise e

    async def delete_memories(
        self,
        *,
        user_id: Optional[str] = None,
        memory_ids: Optional[List[str]] = None,
    ) -> None:
        memories = await self.get_memories(memory_ids=memory_ids, user_id=user_id)
        if not memories:
            return
        docs = [{"id": m.id, "@search.action": "delete"} for m in memories]
        result = await self.client.upload_documents(documents=docs)
        if not all(r.succeeded for r in result):
            failed = [r for r in result if not r.succeeded]
            raise ValueError(f"Failed to delete some memories: {failed}")
