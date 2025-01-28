"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from unittest.mock import Mock

import pytest
from teams_memory.config import LLMConfig, MemoryModuleConfig, StorageConfig
from teams_memory.core.memory_core import MemoryCore
from teams_memory.core.message_queue import MessageQueue
from teams_memory.interfaces.base_message_buffer_storage import (
    BaseMessageBufferStorage,
)

from tests.teams_memory.utils import (
    create_test_assistant_message,
    create_test_memory,
    create_test_user_message,
)


@pytest.mark.asyncio()
async def test_process_for_semantic_messages_enough_messages():
    config = MemoryModuleConfig(
        buffer_size=4,
        storage=StorageConfig(type="in_memory"),  # use in memory storage
        timeout_seconds=60,
        llm=Mock(spec=LLMConfig),
    )

    core = Mock(spec=MemoryCore)
    message_buffer_storage_mock = Mock(spec=BaseMessageBufferStorage)
    core.process_semantic_messages.return_value = None

    mq = MessageQueue(
        config=config,
        memory_core=core,
        message_buffer_storage=message_buffer_storage_mock,
    )

    # recent messages come first
    messages = [
        create_test_assistant_message(content="That's a long time!"),
        create_test_user_message(content="I've been developing software for 5 years."),
        create_test_assistant_message(
            content="That's great! How long have you been developing software?"
        ),
        create_test_user_message(content="Hey, I'm a software developer."),
    ]

    assert await mq._process_for_semantic_messages(messages) is None
    assert core.process_semantic_messages.call_count == 1


@pytest.mark.asyncio()
async def test_process_for_semantic_messages_less_messages():
    config = MemoryModuleConfig(
        buffer_size=5,
        storage=StorageConfig(type="in_memory"),  # use in-memory storage
        timeout_seconds=60,
        llm=Mock(spec=LLMConfig),
    )

    core = Mock(spec=MemoryCore)

    messages = [
        create_test_assistant_message(content="Hi there, what do you do for a living?"),
        create_test_user_message(
            content="Hey, I'm a software developer.", id="user_msg"
        ),
    ]
    existing_memories = [
        create_test_memory(content="The user is a software developer"),
    ]

    # number of messages is less than buffer size
    buffered_messages = [
        create_test_assistant_message(
            content="That's great! How long have you been developing software?"
        ),
        create_test_user_message(content="I've been developing software for 5 years."),
        create_test_assistant_message(content="That's a long time!"),
    ]

    core.process_semantic_messages.return_value = None
    core.retrieve_conversation_history.return_value = messages

    async def mock_get_memories_from_message(message_id):
        return existing_memories if message_id == "user_msg" else []

    core.get_memories_from_message = mock_get_memories_from_message

    message_buffer_storage_mock = Mock(spec=BaseMessageBufferStorage)
    mq = MessageQueue(
        config=config,
        memory_core=core,
        message_buffer_storage=message_buffer_storage_mock,
    )

    assert await mq._process_for_semantic_messages(buffered_messages) is None
    assert (
        core.retrieve_conversation_history.call_count == 1
    )  # just once to get stored messages
    assert core.process_semantic_messages.call_count == 1
    assert (
        core.process_semantic_messages.call_args.kwargs["messages"]
        == messages + buffered_messages
    )
    assert (
        core.process_semantic_messages.call_args.kwargs["existing_memories"]
        == existing_memories
    )
