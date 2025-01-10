from abc import ABC
from datetime import datetime
from enum import Enum
from typing import ClassVar, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class User(BaseModel):
    """Represents a user in the system."""

    id: str


class BaseMessageInput(ABC, BaseModel):
    content: str
    author_id: str
    conversation_ref: str
    created_at: datetime


class InternalMessageInput(BaseMessageInput):
    """
    Input parameter for an internal message. Used when creating a new message.
    """

    model_config = ConfigDict(from_attributes=True)
    type: ClassVar = "internal"
    deep_link: ClassVar[None] = None


class InternalMessage(InternalMessageInput):
    """
    Represents a message that is not meant to be shown to the user.
    Useful for keeping agentic transcript state.
    These are not used as part of memory extraction
    """

    model_config = ConfigDict(from_attributes=True)

    id: str


class UserMessageInput(BaseMessageInput):
    """
    Input parameter for a user message. Used when creating a new message.
    """

    model_config = ConfigDict(from_attributes=True)
    id: str
    type: ClassVar = "user"
    deep_link: Optional[str] = None


class UserMessage(UserMessageInput):
    """
    Represents a message that was sent by the user.
    """

    model_config = ConfigDict(from_attributes=True)


class AssistantMessageInput(BaseMessageInput):
    """
    Input parameter for an assistant message. Used when creating a new message.
    """

    model_config = ConfigDict(from_attributes=True)
    id: str
    type: ClassVar = "assistant"
    deep_link: Optional[str] = None


class AssistantMessage(AssistantMessageInput):
    """
    Represents a message that was sent by the assistant.
    """

    model_config = ConfigDict(from_attributes=True)


MessageInput = InternalMessageInput | UserMessageInput | AssistantMessageInput
Message = InternalMessage | UserMessage | AssistantMessage


class MemoryAttribution(BaseModel):
    memory_id: str
    message_id: str


class MemoryType(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


class BaseMemoryInput(BaseModel):
    """Represents a processed memory."""

    model_config = ConfigDict(from_attributes=True)

    content: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    memory_type: MemoryType
    user_id: Optional[str] = None
    message_attributions: Optional[List[str]] = Field(default=[])
    topics: Optional[List[str]] = None


class Topic(BaseModel):
    name: str = Field(description="A unique name of the topic that the memory module should listen to")
    description: str = Field(description="Description of the topic")


class Memory(BaseMemoryInput):
    """Represents a processed memory."""

    id: str


class TextEmbedding(BaseModel):
    text: str
    embedding_vector: List[float]


class RetrievalConfig(BaseModel):
    """Configuration for memory retrieval operations.

    This class defines the parameters used to retrieve memories from storage. Memories can be
    retrieved either by a semantic search query or by filtering for a specific topic or both.

    In case of both, the memories are retrieved by the intersection of the two sets.
    """

    query: Optional[str] = Field(
        default=None, description="A natural language query to search for semantically similar memories"
    )
    topic: Optional[Topic] = Field(
        default=None,
        description="Topic to filter memories by. Only memories tagged with this topic will be retrieved",
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of memories to retrieve. If not specified, all matching memories are returned",
    )

    @model_validator(mode="after")
    def check_parameters(self) -> "RetrievalConfig":
        if self.query is None and self.topic is None:
            raise ValueError("Either query or topic must be provided")
        return self


class ShortTermMemoryRetrievalConfig(RetrievalConfig):
    n_messages: Optional[int] = None  # Number of messages to retrieve
    last_minutes: Optional[float] = None  # Time frame in minutes
    before: Optional[datetime] = None  # Retrieve messages up until a specific timestamp

    @model_validator(mode="after")
    def check_parameters(self) -> "ShortTermMemoryRetrievalConfig":
        if self.n_messages is None and self.last_minutes is None:
            raise ValueError("Either n_messages or last_minutes must be provided")
        return self
