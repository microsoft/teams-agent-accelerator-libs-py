"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import os
from datetime import timedelta
from uuid import uuid4

import pytest
from teams_memory.config import StorageConfig
from teams_memory.storage.in_memory_storage import InMemoryStorage
from teams_memory.storage.sqlite_memory_storage import SQLiteMemoryStorage
from teams_memory.storage.sqlite_message_buffer_storage import (
    SQLiteMessageBufferStorage,
)

from tests.teams_memory.utils import create_test_user_message


@pytest.fixture(params=["sqlite", "in_memory"])
def storage(request):
    if request.param == "sqlite":
        db_path = f"storage_{uuid4().hex}.db"
        buffer = SQLiteMessageBufferStorage(StorageConfig(db_path=db_path))
        memory_storage = SQLiteMemoryStorage(StorageConfig(db_path=db_path))
        yield buffer, memory_storage
        os.remove(db_path)
    else:
        buffer = InMemoryStorage()  # type: ignore
        memory_storage = buffer  # type: ignore
        yield buffer, memory_storage


@pytest.mark.asyncio
async def test_store_and_get_buffered_message(storage):
    message = create_test_user_message("Hi")
    message.id = "msg1"
    message.conversation_ref = "conv1"

    buffer, memory_storage = storage
    await memory_storage.upsert_message(message=message)
    await buffer.store_buffered_message(message=message)
    messages = await buffer.get_buffered_messages(
        conversation_ref=message.conversation_ref
    )

    assert len(messages) == 1
    assert messages[0].id == message.id


@pytest.mark.asyncio
async def test_clear_buffered_messages(storage):
    message = create_test_user_message("Hi")
    message.id = "msg1"
    message.conversation_ref = "conv1"

    buffer, memory_storage = storage
    await memory_storage.upsert_message(message=message)
    await buffer.store_buffered_message(message=message)

    # Test clear the buffer
    await buffer.clear_buffered_messages(conversation_ref=message.conversation_ref)
    messages = await buffer.get_buffered_messages(
        conversation_ref=message.conversation_ref
    )

    assert len(messages) == 0


@pytest.mark.asyncio
async def test_clear_buffered_messages_before_time(storage):
    message = create_test_user_message("Hi")
    message.id = "msg1"
    message.conversation_ref = "conv1"

    buffer, memory_storage = storage
    await memory_storage.upsert_message(message=message)
    await buffer.store_buffered_message(message=message)

    # Clear all messages before the message created time, shouldn't clear the message
    before = message.created_at - timedelta(seconds=1)
    await buffer.clear_buffered_messages(
        conversation_ref=message.conversation_ref, before=before
    )
    messages = await buffer.get_buffered_messages(
        conversation_ref=message.conversation_ref
    )
    assert len(messages) == 1

    # Clear all messages before the message created time + 1 second, should clear the message
    before = message.created_at + timedelta(seconds=1)
    await buffer.clear_buffered_messages(
        conversation_ref=message.conversation_ref, before=before
    )
    messages = await buffer.get_buffered_messages(
        conversation_ref=message.conversation_ref
    )
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_count_buffered_messages(storage):
    message = create_test_user_message("Hi")
    message.id = "msg1"
    message.conversation_ref = "conv1"

    buffer, memory_storage = storage
    await memory_storage.upsert_message(message=message)
    await buffer.store_buffered_message(message=message)

    count_ref = await buffer.count_buffered_messages(
        conversation_refs=[message.conversation_ref]
    )
    assert count_ref[message.conversation_ref] == 1


@pytest.mark.asyncio
async def test_get_conversations_from_buffered_messages(storage):
    message1 = create_test_user_message("Hi")
    message1.id = "msg1"
    message1.conversation_ref = "conv1"

    buffer, memory_storage = storage
    await memory_storage.upsert_message(message=message1)
    await buffer.store_buffered_message(message=message1)

    conversation_refs = await buffer.get_conversations_from_buffered_messages(["msg1"])
    assert "conv1" in conversation_refs
    assert "msg1" in conversation_refs["conv1"]


@pytest.mark.asyncio
async def test_get_earliest_buffered_message(storage):
    # Create messages with different timestamps
    message1 = create_test_user_message("First")
    message1.id = "msg1"
    message1.conversation_ref = "conv1"
    message1.created_at = message1.created_at - timedelta(hours=2)

    message2 = create_test_user_message("Second")
    message2.id = "msg2"
    message2.conversation_ref = "conv1"
    message2.created_at = message2.created_at - timedelta(hours=1)

    message3 = create_test_user_message("Third")
    message3.id = "msg3"
    message3.conversation_ref = "conv2"

    buffer, memory_storage = storage
    # Store messages
    for msg in [message1, message2, message3]:
        await memory_storage.upsert_message(message=msg)
        await buffer.store_buffered_message(message=msg)

    # Test getting earliest message from specific conversation
    earliest = await buffer.get_earliest_buffered_message(conversation_refs=["conv1"])
    assert len(earliest) == 1
    assert earliest["conv1"].message_id == "msg1"

    # Test getting earliest messages from all conversations
    all_earliest = await buffer.get_earliest_buffered_message()
    assert len(all_earliest) == 2
    assert all_earliest["conv1"].message_id == "msg1"
    assert all_earliest["conv2"].message_id == "msg3"
