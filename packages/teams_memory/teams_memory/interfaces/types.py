"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from abc import ABC
from datetime import datetime
from enum import Enum
from typing import ClassVar, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field


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
    created_at: datetime = Field(default_factory=datetime.now)
    source: Optional[str] = None


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
    user_id: Optional[str] = None


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
    message_attributions: Optional[Set[str]] = set()
    topics: Optional[List[str]] = None


class Topic(BaseModel):
    name: str = Field(
        description="A unique name of the topic that the memory module should listen to"
    )
    description: str = Field(description="Description of the topic")


class Memory(BaseMemoryInput):
    """Represents a processed memory."""

    id: str
    relevance_score: Optional[float] = None


class MemoryWithAttributions(BaseModel):
    """A memory with its attributed messages."""

    memory: Memory
    messages: List[Message]


class TextEmbedding(BaseModel):
    text: str
    embedding_vector: List[float]
