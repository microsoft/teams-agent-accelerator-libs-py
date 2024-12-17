from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class User(BaseModel):
    """Represents a user in the system."""

    id: str


class Message(BaseModel):
    """Represents a message in a conversation."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    content: str
    author_id: Optional[str]
    conversation_ref: str
    created_at: datetime
    is_assistant_message: bool = False
    deep_link: Optional[str] = None


class MemoryAttribution(BaseModel):
    memory_id: int
    message_id: str


class MemoryType(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


class BaseMemoryInput(BaseModel):
    """Represents a processed memory."""

    model_config = ConfigDict(from_attributes=True)

    content: str
    created_at: datetime
    memory_type: MemoryType
    user_id: Optional[str] = None
    message_attributions: Optional[List[str]] = Field(default_factory=list)


class Memory(BaseMemoryInput):
    """Represents a processed memory."""

    id: int


class EmbedText(BaseModel):
    text: str
    embedding_vector: List[float]


class ShortTermMemoryRetrievalConfig(BaseModel):
    n_messages: Optional[int] = None  # Number of messages to retrieve
    last_minutes: Optional[Decimal] = None  # Time frame in minutes

    @model_validator(mode="after")
    def check_parameters(self) -> "ShortTermMemoryRetrievalConfig":
        if self.n_messages is None and self.last_minutes is None:
            raise ValueError("Either n_messages or last_minutes must be provided")
        return self
