from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class User(BaseModel):
    """Represents a user in the system."""

    id: str


class Message(BaseModel):
    """Represents a message in a conversation."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    content: str
    author_id: str
    conversation_ref: str
    created_at: datetime


class MemoryAttribution(BaseModel):
    memory_id: int
    message_id: str


class Memory(BaseModel):
    """Represents a processed memory."""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    content: str
    created_at: datetime
    user_id: Optional[str] = None
    message_attributions: Optional[List[str]] = Field(default_factory=list)