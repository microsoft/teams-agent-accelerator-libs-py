import os
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from memory_module.interfaces.types import (
    AssistantMessageInput,
    BaseMemoryInput,
    EmbedText,
    MemoryType,
    ShortTermMemoryRetrievalConfig,
    UserMessageInput,
)
from memory_module.storage.in_memory_storage import InMemoryStorage
from memory_module.storage.sqlite_memory_storage import SQLiteMemoryStorage


@pytest.fixture(params=["sqlite", "in_memory"])
def memory_storage(request):
    if request.param == "sqlite":
        name = f"memory_{uuid4().hex}.db"
        storage = SQLiteMemoryStorage(name)
        yield storage
        os.remove(storage.db_path)

    else:
        yield InMemoryStorage()


@pytest.fixture
def sample_memory_input():
    return BaseMemoryInput(
        content="Test memory content",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=["msg1", "msg2"],
    )


@pytest.fixture
def sample_message():
    return UserMessageInput(
        id="msg1",
        content="Test message",
        author_id="user1",
        conversation_ref="conv1",
        created_at=datetime.now(),
    )


@pytest.fixture
def sample_embedding():
    # dims = 1536
    return [[0.1] * 1536]


@pytest.mark.asyncio
async def test_store_and_get_memory(memory_storage, sample_memory_input, sample_embedding):
    # Store memory
    memory_id = await memory_storage.store_memory(sample_memory_input, embedding_vectors=sample_embedding)
    assert memory_id is not None

    # Retrieve memory
    retrieved_memory = await memory_storage.get_memory(memory_id)
    assert retrieved_memory is not None
    assert retrieved_memory.content == sample_memory_input.content
    assert retrieved_memory.user_id == sample_memory_input.user_id
    assert retrieved_memory.memory_type == sample_memory_input.memory_type
    assert set(retrieved_memory.message_attributions) == set(sample_memory_input.message_attributions)


@pytest.mark.asyncio
async def test_update_memory(memory_storage, sample_memory_input, sample_embedding):
    # Store initial memory
    memory_id = await memory_storage.store_memory(sample_memory_input, embedding_vectors=sample_embedding)

    # Update memory
    updated_content = "Updated memory content"
    await memory_storage.update_memory(memory_id, updated_content, embedding_vectors=sample_embedding)

    # Verify update
    updated_memory = await memory_storage.get_memory(memory_id)
    assert updated_memory.content == updated_content


@pytest.mark.asyncio
async def test_retrieve_memories(memory_storage, sample_memory_input, sample_embedding):
    # Store memory
    await memory_storage.store_memory(sample_memory_input, embedding_vectors=sample_embedding)

    # Create query embedding
    query = EmbedText(text="test query", embedding_vector=sample_embedding[0])

    # Retrieve memories
    memories = await memory_storage.retrieve_memories(query, "test_user", limit=1)
    assert len(memories) > 0
    assert memories[0].content == sample_memory_input.content


@pytest.mark.asyncio
async def test_retrieve_memories_multiple_embeddings(memory_storage, sample_memory_input):
    # Test with multiple embeddings per memory
    embeddings = [
        [0.1] * 1536,  # First embedding with low similarity
        [1.0] * 1536,  # Second embedding with high similarity
    ]

    await memory_storage.store_memory(sample_memory_input, embedding_vectors=embeddings)

    # Query should match the second embedding better
    query = EmbedText(text="test query", embedding_vector=[1.0] * 1536)

    memories = await memory_storage.retrieve_memories(query, "test_user", limit=1)
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_clear_memories(memory_storage, sample_memory_input, sample_embedding):
    # Store memory
    await memory_storage.store_memory(sample_memory_input, embedding_vectors=sample_embedding)

    # Clear memories
    await memory_storage.clear_memories(sample_memory_input.user_id)

    # Verify memories are cleared
    memories = await memory_storage.get_all_memories()
    assert len(memories) == 0


@pytest.mark.asyncio
async def test_store_and_retrieve_chat_history(memory_storage, sample_message):
    # Store message
    await memory_storage.store_short_term_memory(sample_message)

    # Retrieve chat history with n_messages
    messages = await memory_storage.retrieve_chat_history(
        sample_message.conversation_ref, ShortTermMemoryRetrievalConfig(n_messages=1)
    )
    assert len(messages) == 1
    assert messages[0].content == sample_message.content

    # Retrieve chat history with last_minutes
    messages = await memory_storage.retrieve_chat_history(
        sample_message.conversation_ref, ShortTermMemoryRetrievalConfig(last_minutes=5)
    )
    assert len(messages) == 1
    assert messages[0].content == sample_message.content

    # Retrieve chat history with `before` parameter set to after the message's creation time
    messages = await memory_storage.retrieve_chat_history(
        sample_message.conversation_ref,
        ShortTermMemoryRetrievalConfig(n_messages=1, before=sample_message.created_at + timedelta(seconds=1)),
    )
    assert len(messages) == 1
    assert messages[0].content == sample_message.content

    # Retrieve chat history with `before` parameter set to before the message's creation time
    messages = await memory_storage.retrieve_chat_history(
        sample_message.conversation_ref,
        ShortTermMemoryRetrievalConfig(n_messages=1, before=sample_message.created_at - timedelta(seconds=1)),
    )
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_get_all_memories(memory_storage, sample_memory_input, sample_embedding):
    # Store multiple memories
    await memory_storage.store_memory(sample_memory_input, embedding_vectors=sample_embedding)

    second_memory = BaseMemoryInput(
        content="Second memory",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=[],
    )
    await memory_storage.store_memory(second_memory, embedding_vectors=sample_embedding)

    # Test without limit
    memories = await memory_storage.get_all_memories()
    assert len(memories) == 2

    # Test with limit
    memories = await memory_storage.get_all_memories(limit=1)
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_get_all_memories_by_message_id(memory_storage, sample_memory_input, sample_message):
    # Store single memory
    await memory_storage.store_memory(sample_memory_input, embedding_vectors=[])
    await memory_storage.store_short_term_memory(sample_message)

    # Get memories by message ID
    memories = await memory_storage.get_all_memories(message_ids=[sample_message.id])

    assert len(memories) == 1
    assert memories[0].content == sample_memory_input.content


@pytest.mark.asyncio
async def test_get_all_memories_by_message_ids(memory_storage, sample_memory_input, sample_message):
    # Store three memories
    await memory_storage.store_memory(sample_memory_input, embedding_vectors=[])
    await memory_storage.store_short_term_memory(sample_message)
    second_memory = BaseMemoryInput(
        content="Second memory",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=["msg2"],
    )
    second_message = UserMessageInput(
        id="msg2",
        content="Test message",
        author_id="user2",
        conversation_ref="conv2",
        created_at=datetime.now(),
    )
    await memory_storage.store_memory(second_memory, embedding_vectors=[])
    await memory_storage.store_short_term_memory(second_message)
    third_memory = BaseMemoryInput(
        content="Third memory",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=["msg3"],
    )
    third_message = UserMessageInput(
        id="msg3",
        content="Test message",
        author_id="user3",
        conversation_ref="conv3",
        created_at=datetime.now(),
    )
    await memory_storage.store_memory(third_memory, embedding_vectors=[])
    await memory_storage.store_short_term_memory(third_message)

    # Get memories by message ID
    memories = await memory_storage.get_all_memories(message_ids=[sample_message.id, second_message.id])

    assert len(memories) == 2
    assert any(second_memory.content in memory.content for memory in memories)


@pytest.mark.asyncio
async def test_get_all_memories_by_message_id_empty(memory_storage, sample_memory_input, sample_message):
    # Store single memory
    await memory_storage.store_memory(sample_memory_input, embedding_vectors=[])
    await memory_storage.store_short_term_memory(sample_message)

    # Get memories by message ID
    memories = await memory_storage.get_all_memories(message_id="incorrect_message_id")

    assert len(memories) == 0


@pytest.mark.asyncio
async def test_get_memories_by_ids(memory_storage, sample_memory_input, sample_embedding):
    # Store memory
    memory_id = await memory_storage.store_memory(sample_memory_input, embedding_vectors=sample_embedding)

    # Retrieve by ID
    memories = await memory_storage.get_memories([memory_id])
    assert len(memories) == 1
    assert memories[0].content == sample_memory_input.content


@pytest.mark.asyncio
async def test_get_messages(memory_storage):
    # Test data
    test_messages = [
        UserMessageInput(
            id="msg1",
            content="Test message 1",
            author_id="user1",
            conversation_ref="conv1",
            created_at=datetime.now(),
            deep_link="link1",
        ),
        AssistantMessageInput(
            id="msg2",
            content="Test message 2",
            author_id="user1",
            conversation_ref="conv1",
            deep_link="link2",
        ),
    ]

    # Store test messages
    for message in test_messages:
        await memory_storage.store_short_term_memory(message)

    # Create memory attributions
    memory_id_1 = await memory_storage.store_memory(
        BaseMemoryInput(
            content="Memory 1",
            created_at=datetime.now(),
            user_id="user1",
            memory_type=MemoryType.SEMANTIC,
            message_attributions=["msg1", "msg2"],
        ),
        embedding_vectors=[],
    )
    memory_id_2 = await memory_storage.store_memory(
        BaseMemoryInput(
            content="Memory 2",
            created_at=datetime.now(),
            user_id="user1",
            memory_type=MemoryType.SEMANTIC,
            message_attributions=["msg2"],
        ),
        embedding_vectors=[],
    )

    # Test retrieving messages
    result = await memory_storage.get_messages([memory_id_1, memory_id_2])

    # Assertions
    assert len(result) == 2
    assert len(result[memory_id_1]) == 2  # Memory 1 should have 2 messages
    assert len(result[memory_id_2]) == 1  # Memory 2 should have 1 message

    # Verify message content
    assert result[memory_id_1][0].id == "msg1"
    assert result[memory_id_1][1].id == "msg2"
    assert result[memory_id_2][0].id == "msg2"
