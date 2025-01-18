from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from memory_module.interfaces.types import Topic


class LLMConfig(BaseModel):
    """Configuration for LLM service."""

    model_config = ConfigDict(extra="allow")  # Allow arbitrary kwargs

    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    embedding_model: Optional[str] = None


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

    # If db_path is empty, use in-memory storage
    db_path: Optional[Path] = Field(
        default_factory=lambda: Path(__file__).parent / "data" / "memory.db",
        description="Path to SQLite database file",
    )
    buffer_size: int = Field(
        default=5, description="Number of messages to collect before processing"
    )
    timeout_seconds: int = Field(
        default=300,  # 5 minutes
        description="Seconds to wait before processing a conversation",
    )
    llm: LLMConfig = Field(description="LLM service configuration")
    topics: list[Topic] = Field(
        default=DEFAULT_TOPICS,
        description="List of topics that the memory module should listen to",
        min_length=1,
    )
    enable_logging: bool = Field(
        default=False, description="Enable verbose logging for memory module"
    )
