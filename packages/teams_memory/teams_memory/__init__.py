"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from teams_memory.config import (
    AzureAISearchStorageConfig,
    InMemoryStorageConfig,
    LLMConfig,
    MemoryModuleConfig,
    SQLiteStorageConfig,
    StorageConfig,
)
from teams_memory.core.memory_module import MemoryModule
from teams_memory.interfaces.base_memory_module import (
    BaseMemoryModule,
    BaseScopedMemoryModule,
)
from teams_memory.interfaces.types import (
    AssistantMessage,
    AssistantMessageInput,
    InternalMessage,
    InternalMessageInput,
    Memory,
    Message,
    MessageInput,
    Topic,
    UserMessage,
    UserMessageInput,
)
from teams_memory.utils.logging import configure_logging
from teams_memory.utils.teams_bot_middlware import MemoryMiddleware

__all__ = [
    "BaseMemoryModule",
    "MemoryModule",
    "MemoryModuleConfig",
    "AzureAISearchStorageConfig",
    "InMemoryStorageConfig",
    "SQLiteStorageConfig",
    "StorageConfig",
    "LLMConfig",
    "Memory",
    "InternalMessage",
    "InternalMessageInput",
    "UserMessageInput",
    "UserMessage",
    "Message",
    "MessageInput",
    "AssistantMessage",
    "AssistantMessageInput",
    "MemoryMiddleware",
    "Topic",
    "BaseScopedMemoryModule",
    "configure_logging",
]
