"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from teams_memory.interfaces.types import Topic


class LLMConfig(BaseModel):
    """Configuration for LLM service."""

    model_config = ConfigDict(extra="allow")  # Allow arbitrary kwargs

    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    embedding_model: Optional[str] = None


class StorageConfig(BaseModel):
    """Base class for storage service configuration. Subclasses should define specific storage types."""

    model_config = ConfigDict(extra="allow")  # Allow arbitrary kwargs


class InMemoryStorageConfig(StorageConfig):
    """Configuration for in-memory storage."""

    storage_type: Literal["in-memory"] = Field(
        default="in-memory", description="The type of storage to use (in-memory)"
    )


class SQLiteStorageConfig(StorageConfig):
    """Configuration for SQLite storage."""

    storage_type: Literal["sqlite"] = Field(
        default="sqlite", description="The type of storage to use (SQLite)"
    )
    db_path: Path = Field(..., description="The path to the SQLite database file")


class AzureAISearchStorageConfig(StorageConfig):
    """Configuration for Azure AI Search storage."""

    storage_type: Literal["azure_ai_search"] = Field(
        description="The type of storage to use (Azure AI Search)",
        default="azure_ai_search",
    )
    endpoint: str = Field(description="Azure Search service endpoint")
    api_key: str = Field(description="Azure Search API key")
    index_name: str = Field(description="Name of the Azure Search index for memories")
    embedding_dimensions: int = Field(
        description="Dimensions of the embedding vectors", default=1536
    )


DEFAULT_TOPICS = [
    Topic(
        name="General Interests and Preferences",
        description="When a user mentions specific events or actions, focus on the underlying interests, hobbies, or preferences they reveal (e.g., if the user mentions attending a conference, focus on the topic of the conference, not the date or location).",  # noqa: E501
    ),
    Topic(
        name="General Facts about the user",
        description="Facts that describe relevant information about the user, such as details about where they live or things they own.",  # noqa: E501
    ),
]


class MemoryModuleConfig(BaseModel):
    """Configuration for memory module components.

    All values are optional and will be merged with defaults if not provided.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    """
    Storage configuration for memories. If this is not provided, the value from `storage` will be used.
    If Azure AI Search is used as the default storage, all other storages must be provided or
    the default storage must be provided.
    """
    memory_storage: Optional[StorageConfig] = Field(
        default=None,
        description="Config for memory storage. Falls back to `storage` if not set.",
    )

    """
    Storage configuration for messages. If this is not provided, the value from `storage` will be used.
    """
    message_storage: Optional[StorageConfig] = Field(
        default=None,
        description="Config for message storage. Falls back to `storage` if not set.",
    )

    """
    Storage configuration for message buffer. If this is not provided, the value from `storage` will be used.
    """
    message_buffer_storage: Optional[StorageConfig] = Field(
        default=None,
        description="Config for message buffer storage. Falls back to `storage` if not set.",
    )

    """
    Storage configuration for scheduled events. If this is not provided, the value from `storage` will be used.
    """
    scheduled_events_storage: Optional[StorageConfig] = Field(
        default=None,
        description="Config for scheduled events storage. Falls back to `storage` if not set.",
    )

    """
    Global storage configuration. Used as a fallback if a per-type storage config is not provided.

    This must be provided if a per-type storage config is not provided.
    If Azure AI Search is used as the default storage, all other storages must be provided.
    """
    storage: Optional[StorageConfig] = Field(
        default=None,
        description="Global storage config (used if per-type not set)",
    )

    """
    Buffer size configuration. This dictates how many messages are collected per conversation before processing.

    The system uses the minimum of this and the `timeout_seconds` to determine when to process the conversation
    for extraction.
    """
    buffer_size: int = Field(
        default=10, description="Number of messages to collect before processing"
    )

    """
    Timeout configuration. This dictates how long the system waits after the first message in a conversation
    before processing for extraction.

    The system uses the minimum of this and the `buffer_size` to determine when to process the conversation
    for extraction.
    """
    timeout_seconds: int = Field(
        default=300,  # 5 minutes
        description="Seconds to wait before processing a conversation",
    )

    """
    LLM configuration.
    """
    llm: LLMConfig = Field(description="LLM service configuration")

    """
    Topics configuration. Use these to specify the topics that the memory module should listen to.
    """
    topics: list[Topic] = Field(
        default=DEFAULT_TOPICS,
        description="List of topics that the memory module should listen to",
        min_length=1,
    )

    """
    Enable logging configuration. If this is set to True, the memory module will log all messages to the console.

    Recommended for debugging.
    """
    enable_logging: bool = Field(
        default=False, description="Enable verbose logging for memory module"
    )

    def get_storage_config(
        self,
        storage_type: Literal[
            "memory", "message", "message_buffer", "scheduled_events"
        ],
    ) -> StorageConfig:
        """
        Returns the storage config for the given type, falling back to global or raising if not set.
        storage_type: one of 'memory', 'message', 'message_buffer', 'scheduled_events'
        """
        per_type: Optional[StorageConfig] = getattr(self, f"{storage_type}_storage")
        if per_type is not None:
            return per_type
        if self.storage is not None:
            return self.storage
        raise ValueError(
            f"No storage config provided for {storage_type}. Please set either the per-type config or the global 'storage' config."  # noqa: E501
        )

    @model_validator(mode="after")
    def validate_storage_configurations(self) -> "MemoryModuleConfig":
        from teams_memory.config import AzureAISearchStorageConfig

        # Helper to check if a config is Azure AI Search
        def is_azure_ai_search(cfg: Any) -> bool:
            return isinstance(cfg, AzureAISearchStorageConfig)

        # Gather all per-type storages and names
        per_type_storages = [
            self.memory_storage,
            self.message_storage,
            self.message_buffer_storage,
            self.scheduled_events_storage,
        ]
        per_type_names = [
            "memory_storage",
            "message_storage",
            "message_buffer_storage",
            "scheduled_events_storage",
        ]
        # Ensure that for each storage type, at least one of per-type or global is provided
        for idx, name in enumerate(per_type_names):
            if per_type_storages[idx] is None and self.storage is None:
                raise ValueError(
                    f"No storage config provided for {name}. Please set either the per-type config or the global 'storage' config."  # noqa: E501
                )
        # 1. If the default storage is Azure AI Search, all other non-memory storages must be set and
        # not Azure AI Search
        if is_azure_ai_search(self.storage):
            non_memory_storages = [
                (self.message_storage, "message_storage"),
                (self.message_buffer_storage, "message_buffer_storage"),
                (self.scheduled_events_storage, "scheduled_events_storage"),
            ]
            for field_value, field_name in non_memory_storages:
                if field_value is None or is_azure_ai_search(field_value):
                    raise ValueError(
                        f"If the default storage is Azure AI Search, you must provide a non-Azure AI Search config for {field_name}."  # noqa: E501
                    )
        # 2. If memory_storage is Azure AI Search, then either the default storage must be provided and not
        # Azure AI Search, or all other storages must be provided and not Azure AI Search
        if is_azure_ai_search(self.memory_storage):
            if self.storage is not None and not is_azure_ai_search(self.storage):
                # OK: default is provided and not Azure AI Search
                return self
            # Otherwise, all other storages must be provided and not Azure AI Search
            non_memory_storages = [
                (self.message_storage, "message_storage"),
                (self.message_buffer_storage, "message_buffer_storage"),
                (self.scheduled_events_storage, "scheduled_events_storage"),
            ]
            for field_value, field_name in non_memory_storages:
                if field_value is None or is_azure_ai_search(field_value):
                    raise ValueError(
                        f"If memory_storage is Azure AI Search, you must provide a non-Azure AI Search config for {field_name} (or provide a non-Azure AI Search default storage)."  # noqa: E501
                    )
        return self
