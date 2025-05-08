"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from datetime import datetime
from typing import Dict, List, Optional, cast

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswVectorSearchAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    SearchableField,
    VectorSearchAlgorithmKind,
)
from azure.search.documents.models import Vector

from teams_memory.interfaces.base_memory_storage import BaseMemoryStorage
from teams_memory.interfaces.errors import MemoryNotFoundError, MessageNotFoundError
from teams_memory.interfaces.types import (
    BaseMemoryInput,
    Memory,
    Message,
    MessageInput,
    TextEmbedding,
)


class AzureSearchMemoryStorage(BaseMemoryStorage):
    """Azure AI Search implementation of memory storage."""

    def __init__(
        self,
        search_service_name: str,
        index_name: str = "teams-memories",
        api_key: Optional[str] = None,
        api_version: str = "2023-07-01-Preview",
        endpoint: Optional[str] = None,
    ):
        """Initialize Azure AI Search memory storage.

        Args:
            search_service_name (str): Name of the Azure AI Search service
            index_name (str, optional): Name of the search index. Defaults to "teams-memories".
            api_key (Optional[str], optional): API key for authentication. Defaults to None.
            api_version (str, optional): API version to use. Defaults to "2023-07-01-Preview".
            endpoint (Optional[str], optional): Custom endpoint URL. Defaults to None.
        """
        self.index_name = index_name
        
        # Set up the endpoint URL
        if endpoint is None:
            endpoint = f"https://{search_service_name}.search.windows.net"
            
        # Set up credentials
        if api_key:
            credential = AzureKeyCredential(api_key)
        else:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()

        # Create clients
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential,
            api_version=api_version,
        )
        
        self.index_client = SearchIndexClient(
            endpoint=endpoint,
            credential=credential,
            api_version=api_version,
        )
        
        # Ensure index exists
        self._create_index_if_not_exists()

    def _create_index_if_not_exists(self) -> None:
        """Create the search index if it doesn't exist."""
        if not self.index_client.get_index(self.index_name):
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="user_id", type=SearchFieldDataType.String),
                SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset),
                SimpleField(name="updated_at", type=SearchFieldDataType.DateTimeOffset),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(name="topics", type=SearchFieldDataType.Collection(SearchFieldDataType.String)),
                SimpleField(name="source_ids", type=SearchFieldDataType.Collection(SearchFieldDataType.String)),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=1536,  # Assuming OpenAI's text-embedding-ada-002 model
                    vector_search_profile_name="vector-profile",
                ),
            ]
            
            vector_search = VectorSearch(
                algorithms=[
                    HnswVectorSearchAlgorithmConfiguration(
                        name="vector-config",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters={"m": 4, "efConstruction": 400, "efSearch": 500},
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="vector-config",
                    )
                ],
            )
            
            semantic_config = SemanticConfiguration(
                name="semantic-config",
                prioritized_fields=SemanticField(
                    content_fields=[{"name": "content"}],
                ),
            )
            
            semantic_settings = SemanticSettings(
                configurations=[semantic_config],
            )
            
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_settings=semantic_settings,
            )
            
            self.index_client.create_index(index)

    async def store_memory(
        self,
        memory: BaseMemoryInput,
        *,
        embedding_vectors: List[TextEmbedding],
    ) -> str | None:
        """Store a memory with its embedding vectors in Azure AI Search.

        Args:
            memory (BaseMemoryInput): The Memory object to store
            embedding_vectors (List[TextEmbedding]): List of TextEmbedding objects

        Returns:
            str | None: The ID of the stored memory if successful, None otherwise
        """
        # Combine all embeddings into a single vector (average)
        if not embedding_vectors:
            return None
            
        combined_vector = [
            sum(vec[i] for vec in [e.vector for e in embedding_vectors]) / len(embedding_vectors)
            for i in range(len(embedding_vectors[0].vector))
        ]
        
        document = {
            "id": memory.id,
            "user_id": memory.user_id,
            "content": memory.content,
            "topics": memory.topics,
            "source_ids": memory.source_ids,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
            "embedding": Vector(value=combined_vector),
        }
        
        self.search_client.upload_documents([document])
        return memory.id

    async def update_memory(
        self,
        memory_id: str,
        updated_memory: str,
        *,
        embedding_vectors: List[TextEmbedding],
    ) -> None:
        """Update an existing memory with new content and embeddings.

        Args:
            memory_id (str): ID of the memory to update
            updated_memory (str): New content for the memory
            embedding_vectors (List[TextEmbedding]): New embedding vectors

        Raises:
            MemoryNotFoundError: If the specified memory_id doesn't exist
        """
        try:
            result = self.search_client.get_document(memory_id)
        except Exception:
            raise MemoryNotFoundError(f"Memory with ID {memory_id} not found")
            
        # Combine embeddings
        combined_vector = [
            sum(vec[i] for vec in [e.vector for e in embedding_vectors]) / len(embedding_vectors)
            for i in range(len(embedding_vectors[0].vector))
        ]
        
        document = {
            "id": memory_id,
            "content": updated_memory,
            "updated_at": datetime.utcnow().isoformat(),
            "embedding": Vector(value=combined_vector),
            **{k: v for k, v in result.items() if k not in ["content", "updated_at", "embedding"]},
        }
        
        self.search_client.merge_documents([document])

    async def get_memories(
        self, *, memory_ids: Optional[List[str]] = None, user_id: Optional[str] = None
    ) -> List[Memory]:
        """Retrieve memories by IDs or user.

        Args:
            memory_ids (Optional[List[str]]): Optional list of specific memory IDs to retrieve
            user_id (Optional[str]): Optional user ID to retrieve all memories for

        Returns:
            List[Memory]: List of memory objects matching the criteria

        Raises:
            ValueError: If neither memory_ids nor user_id is provided
        """
        if not memory_ids and not user_id:
            raise ValueError("Either memory_ids or user_id must be provided")
            
        filter_expr = None
        if memory_ids:
            id_list = ", ".join([f"'{id}'" for id in memory_ids])
            filter_expr = f"id in ({id_list})"
        elif user_id:
            filter_expr = f"user_id eq '{user_id}'"
            
        results = list(self.search_client.search("*", filter=filter_expr))
        return [self._document_to_memory(doc) for doc in results]

    async def get_attributed_memories(self, message_ids: List[str]) -> List[Memory]:
        """Retrieve all memories attributed to the provided message IDs.

        Args:
            message_ids (List[str]): List of message IDs to filter memories by source

        Returns:
            List[Memory]: List of memory objects ordered by creation date
        """
        id_list = ", ".join([f"'{id}'" for id in message_ids])
        filter_expr = f"source_ids/any(s: s in ({id_list}))"
        
        results = list(self.search_client.search(
            "*",
            filter=filter_expr,
            order_by=["created_at desc"],
        ))
        return [self._document_to_memory(doc) for doc in results]

    async def search_memories(
        self,
        *,
        user_id: Optional[str],
        text_embedding: Optional[TextEmbedding] = None,
        topics: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """Search memories using semantic similarity and/or topics.

        Args:
            user_id (Optional[str]): Filter memories by specific user ID
            text_embedding (Optional[TextEmbedding]): Vector embedding for semantic search
            topics (Optional[List[str]]): List of topics to filter memories by
            limit (Optional[int]): Maximum number of memories to return

        Returns:
            List[Memory]: List of memories matching the criteria, ordered by relevance

        Raises:
            ValueError: If neither text_embedding nor topics is provided
        """
        if not text_embedding and not topics:
            raise ValueError("Either text_embedding or topics must be provided")
            
        filter_conditions = []
        if user_id:
            filter_conditions.append(f"user_id eq '{user_id}'")
            
        if topics:
            topic_list = ", ".join([f"'{topic}'" for topic in topics])
            filter_conditions.append(f"topics/any(t: t in ({topic_list}))")
            
        filter_expr = " and ".join(filter_conditions) if filter_conditions else None
        
        if text_embedding:
            vector_query = Vector(value=text_embedding.vector)
            results = list(self.search_client.search(
                "*",
                filter=filter_expr,
                vector=vector_query,
                vector_fields="embedding",
                top=limit or self.default_limit,
            ))
        else:
            results = list(self.search_client.search(
                "*",
                filter=filter_expr,
                top=limit or self.default_limit,
            ))
            
        return [self._document_to_memory(doc) for doc in results]

    async def delete_memories(
        self, *, user_id: Optional[str] = None, memory_ids: Optional[List[str]] = None
    ) -> None:
        """Remove memories from storage.

        Args:
            user_id (Optional[str]): Optional user ID to remove all memories for
            memory_ids (Optional[List[str]]): Optional list of specific memory IDs to remove

        Raises:
            ValueError: If neither memory_ids nor user_id is provided
        """
        if not memory_ids and not user_id:
            raise ValueError("Either memory_ids or user_id must be provided")
            
        if memory_ids:
            self.search_client.delete_documents(
                documents=[{"id": id} for id in memory_ids]
            )
        else:
            # First get all memories for the user
            memories = await self.get_memories(user_id=user_id)
            if memories:
                self.search_client.delete_documents(
                    documents=[{"id": memory.id} for memory in memories]
                )

    async def upsert_message(self, message: MessageInput) -> Message:
        """Store or update a message in the storage system.

        Args:
            message (MessageInput): The Message object to store or update

        Returns:
            Message: The stored/updated message with assigned ID and metadata
        """
        # For messages, we'll use a separate index with "-messages" suffix
        messages_index = f"{self.index_name}-messages"
        
        # Ensure messages index exists
        if not self.index_client.get_index(messages_index):
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="conversation_ref", type=SearchFieldDataType.String),
                SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset),
                SearchableField(name="content", type=SearchFieldDataType.String),
            ]
            
            index = SearchIndex(
                name=messages_index,
                fields=fields,
            )
            
            self.index_client.create_index(index)
            
        # Create message client
        message_client = SearchClient(
            endpoint=self.search_client._endpoint,
            index_name=messages_index,
            credential=self.search_client._credential,
        )
        
        document = {
            "id": message.id,
            "conversation_ref": message.conversation_ref,
            "content": message.content,
            "created_at": message.created_at.isoformat(),
        }
        
        message_client.merge_or_upload_documents([document])
        return Message(**document)

    async def get_messages(self, message_ids: List[str]) -> List[Message]:
        """Retrieve messages by their IDs.

        Args:
            message_ids (List[str]): List of message IDs to retrieve

        Returns:
            List[Message]: List of message objects matching the provided IDs

        Raises:
            MessageNotFoundError: If any of the specified message IDs don't exist
        """
        messages_index = f"{self.index_name}-messages"
        message_client = SearchClient(
            endpoint=self.search_client._endpoint,
            index_name=messages_index,
            credential=self.search_client._credential,
        )
        
        id_list = ", ".join([f"'{id}'" for id in message_ids])
        filter_expr = f"id in ({id_list})"
        
        results = list(message_client.search("*", filter=filter_expr))
        if len(results) != len(message_ids):
            missing = set(message_ids) - {doc["id"] for doc in results}
            raise MessageNotFoundError(f"Messages not found: {missing}")
            
        return [Message(**doc) for doc in results]

    async def delete_messages(self, message_ids: List[str]) -> None:
        """Remove messages from storage.

        Args:
            message_ids (List[str]): List of message IDs to remove
        """
        messages_index = f"{self.index_name}-messages"
        message_client = SearchClient(
            endpoint=self.search_client._endpoint,
            index_name=messages_index,
            credential=self.search_client._credential,
        )
        
        message_client.delete_documents(
            documents=[{"id": id} for id in message_ids]
        )

    async def retrieve_conversation_history(
        self,
        conversation_ref: str,
        *,
        n_messages: Optional[int] = None,
        last_minutes: Optional[float] = None,
        before: Optional[datetime] = None,
    ) -> List[Message]:
        """Retrieve conversation history based on specified criteria.

        Args:
            conversation_ref (str): Unique identifier for the conversation
            n_messages (Optional[int]): Number of most recent messages to retrieve
            last_minutes (Optional[float]): Retrieve messages from the last N minutes
            before (Optional[datetime]): Retrieve messages before this timestamp

        Returns:
            List[Message]: List of messages matching the criteria
        """
        messages_index = f"{self.index_name}-messages"
        message_client = SearchClient(
            endpoint=self.search_client._endpoint,
            index_name=messages_index,
            credential=self.search_client._credential,
        )
        
        filter_conditions = [f"conversation_ref eq '{conversation_ref}'"]
        
        if last_minutes:
            cutoff = datetime.utcnow().timestamp() - (last_minutes * 60)
            filter_conditions.append(f"created_at ge {cutoff}")
            
        if before:
            filter_conditions.append(f"created_at lt '{before.isoformat()}'")
            
        filter_expr = " and ".join(filter_conditions)
        
        results = list(message_client.search(
            "*",
            filter=filter_expr,
            order_by=["created_at desc"],
            top=n_messages or 1000,  # Use a reasonable default
        ))
        
        return [Message(**doc) for doc in results]

    def _document_to_memory(self, doc: Dict) -> Memory:
        """Convert a search document to a Memory object."""
        return Memory(
            id=doc["id"],
            user_id=doc["user_id"],
            content=doc["content"],
            topics=doc.get("topics", []),
            source_ids=doc.get("source_ids", []),
            created_at=datetime.fromisoformat(doc["created_at"]),
            updated_at=datetime.fromisoformat(doc["updated_at"]) if doc.get("updated_at") else None,
        ) 