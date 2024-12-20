from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    """Configuration for LLM service."""

    model_config = ConfigDict(extra="allow")  # Allow arbitrary kwargs

    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    embedding_model: Optional[str] = None


class MemoryModuleConfig(BaseModel):
    """Configuration for memory module components.

    All values are optional and will be merged with defaults if not provided.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # If db_path is empty, use in-memory storage
    db_path: Optional[Path] = Field(
        default_factory=lambda: Path(__file__).parent / "data" / "memory.db", description="Path to SQLite database file",
    )
    buffer_size: int = Field(default=5, description="Number of messages to collect before processing")
    timeout_seconds: int = Field(
        default=300,  # 5 minutes
        description="Seconds to wait before processing a conversation",
    )
    llm: LLMConfig = Field(description="LLM service configuration")
