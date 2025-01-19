from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_module.interfaces.types import Topic


class LLMConfig(BaseModel):
    """Configuration for LLM service."""

    model_config = ConfigDict(extra="allow")  # Allow arbitrary kwargs

    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    embedding_model: Optional[str] = None


class StorageConfig(BaseModel):
    """Configuration for storage service."""

    model_config = ConfigDict(extra="allow")  # Allow arbitrary kwargs

    storage_type: Literal["in-memory", "sqlite"] | str = Field(
        description="The type of storage to use", default="in-memory"
    )

    """
    The path to the database file. Used for SQLite storage.
    """
    db_path: Optional[Path | str] = Field(
        default=None,
        description="The path to the database file",
    )

    @model_validator(mode="before")
    def set_storage_type(cls, values):
        if isinstance(values, dict):
            if values.get("db_path") and "storage_type" not in values:
                values["storage_type"] = "sqlite"
        return values


class InMemoryStorageConfig(StorageConfig):
    """Configuration for in-memory storage."""

    type: str = "in-memory"


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
    Storage configuration. If this is not provided, the memory module will use in-memory storage.
    """
    storage: Optional[StorageConfig] = Field(
        description="Storage configuration",
        default=StorageConfig(),
    )

    """
    Buffer size configuration. This dictates how many messages are collected per conversation before processing.

    The system uses the minimum of this and the `timeout_seconds` to determine when to process the conversation
    for extraction.
    """
    buffer_size: int = Field(
        default=5, description="Number of messages to collect before processing"
    )

    """
    Timeout configuration. This dictates how long the system waits before the first message in a conversation
    before processing for extraction

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
