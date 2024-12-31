from datetime import timedelta
import os
from uuid import uuid4

import pytest
from memory_module.storage.in_memory_storage import InMemoryStorage
from memory_module.storage.sqlite_memory_storage import SQLiteMemoryStorage
from memory_module.storage.sqlite_message_buffer_storage import SQLiteMessageBufferStorage
from tests.memory_module.utils import create_test_user_message


@pytest.fixture(params=["sqlite", "in_memory"])
def storage(request):
    if request.param == "sqlite":
        db_path = f'storage_{uuid4().hex}.db'
        buffer = SQLiteMessageBufferStorage(db_path=db_path)
        memory_storage = SQLiteMemoryStorage(db_path=db_path)
        yield buffer, memory_storage
        os.remove(db_path)
    else:
        buffer = InMemoryStorage()
        memory_storage = buffer
        yield buffer, memory_storage


@pytest.mark.asyncio
async def test_store_and_get_buffered_message(storage):
    message = create_test_user_message("Hi")
    message.id = "msg1"
    message.conversation_ref = "conv1"
    
    buffer, memory_storage = storage
    await memory_storage.store_short_term_memory(message=message)
    await buffer.store_buffered_message(message=message)
    messages = await buffer.get_buffered_messages(conversation_ref=message.conversation_ref)
    
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
    messages = await buffer.get_buffered_messages(conversation_ref=message.conversation_ref)
    
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
    await buffer.clear_buffered_messages(conversation_ref=message.conversation_ref, before=message.created_at - timedelta(seconds=1))
    messages = await buffer.get_buffered_messages(conversation_ref=message.conversation_ref)
    assert len(messages) == 1
    
    # Clear all messages before the message created time + 1 second, should clear the message
    await buffer.clear_buffered_messages(conversation_ref=message.conversation_ref, before=message.created_at + timedelta(seconds=1))
    messages = await buffer.get_buffered_messages(conversation_ref=message.conversation_ref)
    assert len(messages) == 0

@pytest.mark.asyncio
async def test_count_buffered_messages(storage):
    message = create_test_user_message("Hi")
    message.id = "msg1"
    message.conversation_ref = "conv1"
    
    buffer, memory_storage = storage
    await memory_storage.store_short_term_memory(message=message)
    await buffer.store_buffered_message(message=message)
    
    count = await buffer.count_buffered_messages(conversation_ref=message.conversation_ref)
    assert count == 1