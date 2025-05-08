"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from datetime import datetime
from typing import Dict, List, Optional, Set

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
)

from teams_memory.interfaces.base_message_buffer_storage import BaseMessageBufferStorage
from teams_memory.interfaces.types import Message


class AzureSearchMessageBufferStorage(BaseMessageBufferStorage):
    """Azure AI Search implementation of message buffer storage."""

    def __init__(
        self,
        search_service_name: str,
        index_name: str = "teams-memories-buffer",
        api_key: Optional[str] = None,
        api_version: str = "2023-07-01-Preview",
        endpoint: Optional[str] = None,
    ):
        """Initialize Azure AI Search message buffer storage.

        Args:
            search_service_name (str): Name of the Azure AI Search service
            index_name (str, optional): Name of the search index. Defaults to "teams-memories-buffer".
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
                SimpleField(name="conversation_ref", type=SearchFieldDataType.String),
                SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset),
                SearchableField(name="content", type=SearchFieldDataType.String),
            ]
            
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
            )
            
            self.index_client.create_index(index)

    async def initialize(self) -> None:
        """Initialize the storage. No-op for Azure AI Search."""
        pass

    async def get_messages(self, conversation_ref: str) -> List[Message]:
        """Get all messages for a conversation.

        Args:
            conversation_ref (str): The conversation reference to get messages for

        Returns:
            List[Message]: List of messages for the conversation
        """
        filter_expr = f"conversation_ref eq '{conversation_ref}'"
        results = list(self.search_client.search(
            "*",
            filter=filter_expr,
            order_by=["created_at asc"],
        ))
        return [Message(**doc) for doc in results]

    async def add_message(self, message: Message) -> None:
        """Add a message to the buffer.

        Args:
            message (Message): The message to add
        """
        document = {
            "id": message.id,
            "conversation_ref": message.conversation_ref,
            "content": message.content,
            "created_at": message.created_at.isoformat(),
        }
        
        self.search_client.merge_or_upload_documents([document])

    async def remove_messages(self, message_ids: List[str]) -> None:
        """Remove messages from the buffer.

        Args:
            message_ids (List[str]): List of message IDs to remove
        """
        self.search_client.delete_documents(
            documents=[{"id": id} for id in message_ids]
        )

    async def get_conversation_refs(self) -> Set[str]:
        """Get all conversation references in the buffer.

        Returns:
            Set[str]: Set of conversation references
        """
        results = list(self.search_client.search(
            "*",
            select=["conversation_ref"],
            facets=["conversation_ref"],
        ))
        return {doc["conversation_ref"] for doc in results} 