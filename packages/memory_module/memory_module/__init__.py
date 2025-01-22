"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from memory_module.config import LLMConfig, MemoryModuleConfig, StorageConfig
from memory_module.core.memory_module import MemoryModule
from memory_module.interfaces.base_memory_module import (
    BaseMemoryModule,
    BaseScopedMemoryModule,
)
from memory_module.interfaces.types import (
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
from memory_module.utils.teams_bot_middlware import MemoryMiddleware

__all__ = [
    "BaseMemoryModule",
    "MemoryModule",
    "MemoryModuleConfig",
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
]
