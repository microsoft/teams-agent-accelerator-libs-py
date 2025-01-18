import os
from datetime import timedelta
from uuid import uuid4

import pytest
from memory_module.storage.in_memory_storage import InMemoryStorage
from memory_module.storage.sqlite_memory_storage import SQLiteMemoryStorage
from memory_module.storage.sqlite_message_buffer_storage import (
    SQLiteMessageBufferStorage,
)

from tests.memory_module.utils import create_test_user_message


@pytest.fixture(params=["sqlite", "in_memory"])
def storage(request):
    if request.param == "sqlite":
        db_path = f"storage_{uuid4().hex}.db"
        buffer = SQLiteMessageBufferStorage(db_path=db_path)
        memory_storage = SQLiteMemoryStorage(db_path=db_path)
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
    await memory_storage.store_short_term_memory(message=message)
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
    await memory_storage.store_short_term_memory(message=message)
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
    await memory_storage.store_short_term_memory(message=message)
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
    await memory_storage.store_short_term_memory(message=message)
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
    await memory_storage.store_short_term_memory(message=message1)
    await buffer.store_buffered_message(message=message1)

    conversation_refs = await buffer.get_conversations_from_buffered_messages(["msg1"])
    assert "conv1" in conversation_refs
    assert "msg1" in conversation_refs["conv1"]
