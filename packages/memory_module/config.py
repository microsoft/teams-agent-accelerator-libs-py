from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class MemoryModuleConfig(BaseModel):
    """Configuration for memory module components.

    All values are optional and will be merged with defaults if not provided.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    db_path: Path = Field(
        default_factory=lambda: Path(__file__).parent / "data" / "memory.db", description="Path to SQLite database file"
    )
    buffer_size: int = Field(default=5, description="Number of messages to collect before processing")
    timeout_seconds: int = Field(
        default=300,  # 5 minutes
        description="Seconds to wait before processing a conversation",
    )
