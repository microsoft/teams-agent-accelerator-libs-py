"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import os
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from teams_memory.config import SQLiteStorageConfig
from teams_memory.interfaces.types import (
    AssistantMessageInput,
    BaseMemoryInput,
    MemoryType,
    TextEmbedding,
    UserMessageInput,
)
from teams_memory.storage.in_memory_storage import InMemoryStorage
from teams_memory.storage.sqlite_memory_storage import SQLiteMemoryStorage
from teams_memory.storage.sqlite_message_storage import SQLiteMessageStorage


@pytest.fixture(params=["sqlite", "in_memory"])
def memory_storage(request):
    if request.param == "sqlite":
        db_path = f"memory_{uuid4().hex}.db"
        storage = SQLiteMemoryStorage(SQLiteStorageConfig(db_path=db_path))
        yield storage
        os.remove(db_path)

    else:
        yield InMemoryStorage()


@pytest.fixture(params=["sqlite", "in_memory"])
def message_storage(request):
    if request.param == "sqlite":
        db_path = f"memory_{uuid4().hex}.db"
        storage = SQLiteMessageStorage(SQLiteStorageConfig(db_path=db_path))
        yield storage
        os.remove(db_path)

    else:
        yield InMemoryStorage()


@pytest.fixture
def sample_memory_input():
    return BaseMemoryInput(
        content="Test memory content",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions={"msg1", "msg2"},
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
    # Create a normalized vector for cosine similarity
    vector = [1.0] * 1536  # All ones
    # Normalize to unit length
    magnitude = (1536) ** 0.5  # sqrt(sum of squares)
    normalized = [x / magnitude for x in vector]
    return [TextEmbedding(text="Test memory content", embedding_vector=normalized)]


@pytest.mark.asyncio
async def test_store_and_get_memory(
    memory_storage, sample_memory_input, sample_embedding
):
    # Store memory
    memory_id = await memory_storage.store_memory(
        sample_memory_input, embedding_vectors=sample_embedding
    )
    assert memory_id is not None

    # Retrieve memory
    retrieved_memory = await memory_storage.get_memory(memory_id)
    assert retrieved_memory is not None
    assert retrieved_memory.content == sample_memory_input.content
    assert retrieved_memory.user_id == sample_memory_input.user_id
    assert retrieved_memory.memory_type == sample_memory_input.memory_type
    assert set(retrieved_memory.message_attributions) == set(
        sample_memory_input.message_attributions
    )


@pytest.mark.asyncio
async def test_update_memory(memory_storage, sample_memory_input, sample_embedding):
    # Store initial memory
    memory_id = await memory_storage.store_memory(
        sample_memory_input, embedding_vectors=sample_embedding
    )

    # Update memory
    updated_content = "Updated memory content"
    await memory_storage.update_memory(
        memory_id, updated_content, embedding_vectors=sample_embedding
    )

    # Verify update
    updated_memory = await memory_storage.get_memory(memory_id)
    assert updated_memory.content == updated_content


@pytest.mark.asyncio
async def test_retrieve_memories(memory_storage, sample_memory_input, sample_embedding):
    # Store memory
    await memory_storage.store_memory(
        sample_memory_input, embedding_vectors=sample_embedding
    )

    # Create query embedding
    query = TextEmbedding(
        text="test query", embedding_vector=sample_embedding[0].embedding_vector
    )

    # Retrieve memories
    memories = await memory_storage.search_memories(
        user_id="test_user", text_embedding=query, limit=1
    )
    assert len(memories) > 0
    assert memories[0].content == sample_memory_input.content


@pytest.mark.asyncio
async def test_retrieve_memories_multiple_embeddings(
    memory_storage, sample_memory_input
):
    # Test with multiple embeddings per memory
    # Create two normalized vectors with different distances
    vector1 = [-1.0] * 1536  # Will give distance of 2 (opposite direction)
    vector2 = [1.0] * 1536  # Will give distance of 0 (same direction)
    magnitude1 = (sum(x * x for x in vector1)) ** 0.5
    magnitude2 = (sum(x * x for x in vector2)) ** 0.5

    embeddings = [
        TextEmbedding(
            text="First embedding", embedding_vector=[x / magnitude1 for x in vector1]
        ),  # High distance
        TextEmbedding(
            text="Second embedding", embedding_vector=[x / magnitude2 for x in vector2]
        ),  # Low distance
    ]

    await memory_storage.store_memory(sample_memory_input, embedding_vectors=embeddings)

    # Query with normalized vector matching second embedding (distance = 0)
    query_vector = [1.0] * 1536
    query_magnitude = (sum(x * x for x in query_vector)) ** 0.5
    query = TextEmbedding(
        text="test query", embedding_vector=[x / query_magnitude for x in query_vector]
    )

    memories = await memory_storage.search_memories(
        user_id="test_user", text_embedding=query, limit=1
    )
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_delete_memories_by_user_id(memory_storage, sample_embedding):
    # Store memories for user
    user_memory1 = BaseMemoryInput(
        content="User memory 1",
        created_at=datetime.now(),
        user_id="user1",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
    )
    user_memory2 = BaseMemoryInput(
        content="User memory 2",
        created_at=datetime.now(),
        user_id="user1",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
    )

    await memory_storage.store_memory(user_memory1, embedding_vectors=sample_embedding)
    await memory_storage.store_memory(user_memory2, embedding_vectors=sample_embedding)

    # Delete all memories for user
    await memory_storage.delete_memories(user_id="user1")

    # Verify memories are deleted
    memories = await memory_storage.get_memories(user_id="user1")
    assert len(memories) == 0


@pytest.mark.asyncio
async def test_delete_specific_memories(memory_storage, sample_embedding):
    # Store memories
    memory1 = BaseMemoryInput(
        content="Memory 1",
        created_at=datetime.now(),
        user_id="user1",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
    )
    memory2 = BaseMemoryInput(
        content="Memory 2",
        created_at=datetime.now(),
        user_id="user1",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
    )

    memory1_id = await memory_storage.store_memory(
        memory1, embedding_vectors=sample_embedding
    )
    memory2_id = await memory_storage.store_memory(
        memory2, embedding_vectors=sample_embedding
    )

    # Delete specific memory
    await memory_storage.delete_memories(memory_ids=[memory1_id])

    # Verify only specified memory is deleted
    memories = await memory_storage.get_memories(user_id="user1")
    assert len(memories) == 1
    assert memories[0].id == memory2_id


@pytest.mark.asyncio
async def test_delete_memories_by_user_and_ids(memory_storage, sample_embedding):
    # Store memories for different users
    user1_memory = BaseMemoryInput(
        content="User 1 memory",
        created_at=datetime.now(),
        user_id="user1",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
    )
    user2_memory = BaseMemoryInput(
        content="User 2 memory",
        created_at=datetime.now(),
        user_id="user2",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
    )

    await memory_storage.store_memory(user1_memory, embedding_vectors=sample_embedding)
    memory2_id = await memory_storage.store_memory(
        user2_memory, embedding_vectors=sample_embedding
    )

    # Try to delete user2's memory using user1's ID
    await memory_storage.delete_memories(user_id="user1", memory_ids=[memory2_id])

    # Verify user2's memory still exists
    memories = await memory_storage.get_memories(user_id="user2")
    assert len(memories) == 1
    assert memories[0].id == memory2_id


@pytest.mark.asyncio
async def test_delete_memories_validation(memory_storage):
    # Test that providing neither parameter raises error
    with pytest.raises(
        ValueError, match="Either user_id or memory_ids must be provided"
    ):
        await memory_storage.delete_memories(user_id=None, memory_ids=None)


@pytest.mark.asyncio
async def test_store_and_retrieve_chat_history(message_storage, sample_message):
    # Store message
    await message_storage.upsert_message(sample_message)

    # Retrieve chat history with n_messages
    messages = await message_storage.retrieve_conversation_history(
        sample_message.conversation_ref, n_messages=1
    )
    assert len(messages) == 1
    assert messages[0].content == sample_message.content

    # Retrieve chat history with last_minutes
    messages = await message_storage.retrieve_conversation_history(
        sample_message.conversation_ref, last_minutes=5
    )
    assert len(messages) == 1
    assert messages[0].content == sample_message.content

    # Retrieve chat history with `before` parameter set to after the message's creation time
    messages = await message_storage.retrieve_conversation_history(
        sample_message.conversation_ref,
        n_messages=1,
        before=sample_message.created_at + timedelta(seconds=1),
    )
    assert len(messages) == 1
    assert messages[0].content == sample_message.content

    # Retrieve chat history with `before` parameter set to before the message's creation time
    messages = await message_storage.retrieve_conversation_history(
        sample_message.conversation_ref,
        n_messages=1,
        before=sample_message.created_at - timedelta(seconds=1),
    )
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_get_attributed_memories_by_message_id(
    memory_storage, message_storage, sample_memory_input, sample_message
):
    # Store single memory
    await memory_storage.store_memory(sample_memory_input, embedding_vectors=[])
    await message_storage.upsert_message(sample_message)

    # Get memories by message ID
    memories = await memory_storage.get_attributed_memories(
        message_ids=[sample_message.id]
    )

    assert len(memories) == 1
    assert memories[0].content == sample_memory_input.content


@pytest.mark.asyncio
async def test_get_attributed_memories_by_message_ids(
    memory_storage, message_storage, sample_memory_input, sample_message
):
    # Store three memories
    await memory_storage.store_memory(sample_memory_input, embedding_vectors=[])
    await message_storage.upsert_message(sample_message)
    second_memory = BaseMemoryInput(
        content="Second memory",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions={"msg2"},
    )
    second_message = UserMessageInput(
        id="msg2",
        content="Test message",
        author_id="user2",
        conversation_ref="conv2",
        created_at=datetime.now(),
    )
    await memory_storage.store_memory(second_memory, embedding_vectors=[])
    await message_storage.upsert_message(second_message)
    third_memory = BaseMemoryInput(
        content="Third memory",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions={"msg3"},
    )
    third_message = UserMessageInput(
        id="msg3",
        content="Test message",
        author_id="user3",
        conversation_ref="conv3",
        created_at=datetime.now(),
    )
    await memory_storage.store_memory(third_memory, embedding_vectors=[])
    await message_storage.upsert_message(third_message)

    # Get memories by message ID
    memories = await memory_storage.get_attributed_memories(
        message_ids=[sample_message.id, second_message.id]
    )

    assert len(memories) == 2
    assert any(second_memory.content in memory.content for memory in memories)


@pytest.mark.asyncio
async def test_get_attributed_memories_by_message_id_empty(
    memory_storage, message_storage, sample_memory_input, sample_message
):
    # Store single memory
    await memory_storage.store_memory(sample_memory_input, embedding_vectors=[])
    await message_storage.upsert_message(sample_message)

    # Get memories by message ID
    memories = await memory_storage.get_attributed_memories(
        message_ids=["incorrect_message_id"]
    )

    assert len(memories) == 0


@pytest.mark.asyncio
async def test_get_memories_by_ids(
    memory_storage, message_storage, sample_memory_input, sample_embedding
):
    # Store memory
    memory_id = await memory_storage.store_memory(
        sample_memory_input, embedding_vectors=sample_embedding
    )

    # Retrieve by ID
    memories = await memory_storage.get_memories(memory_ids=[memory_id])
    assert len(memories) == 1
    assert memories[0].content == sample_memory_input.content


@pytest.mark.asyncio
async def test_get_messages(message_storage):
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
            created_at=datetime.now(),
        ),
    ]

    # Store test messages
    for message in test_messages:
        await message_storage.upsert_message(message)

    # Test retrieving messages
    result = await message_storage.get_messages(["msg1", "msg2"])

    # Assertions
    assert len(result) == 2

    # Verify message content
    message_ids = {msg.id for msg in result}
    assert message_ids == {"msg1", "msg2"}


@pytest.mark.asyncio
async def test_retrieve_memories_by_topic(memory_storage, sample_embedding):
    # Store memories with different topics
    memory1 = BaseMemoryInput(
        content="Memory about AI",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
        topics=["AI"],
    )
    memory2 = BaseMemoryInput(
        content="Memory about nature",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
        topics=["nature"],
    )

    await memory_storage.store_memory(memory1, embedding_vectors=sample_embedding)
    await memory_storage.store_memory(memory2, embedding_vectors=sample_embedding)

    # Retrieve memories by single topic
    memories = await memory_storage.search_memories(
        user_id="test_user", topics=["AI"], limit=10
    )
    assert len(memories) == 1
    assert memories[0].content == "Memory about AI"
    assert "AI" in memories[0].topics

    # Test with non-existent topic
    memories = await memory_storage.search_memories(
        user_id="test_user",
        topics=["non_existent_topic"],
        limit=10,
    )
    assert len(memories) == 0


@pytest.mark.asyncio
async def test_retrieve_memories_by_topic_and_embedding(
    memory_storage, sample_embedding
):
    # Store memories with different topics
    memory1 = BaseMemoryInput(
        content="Technical discussion about artificial intelligence",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
        topics=["AI"],
    )
    memory2 = BaseMemoryInput(
        content="Another AI related memory but less relevant",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
        topics=["AI"],
    )

    await memory_storage.store_memory(memory1, embedding_vectors=sample_embedding)

    # Create vector with higher distance (opposite direction)
    vector = [-1.0] * 1536
    magnitude = (sum(x * x for x in vector)) ** 0.5
    await memory_storage.store_memory(
        memory2,
        embedding_vectors=[
            TextEmbedding(
                text="Less relevant", embedding_vector=[x / magnitude for x in vector]
            )
        ],
    )

    # Create query embedding (same as sample_embedding for low distance)
    query = TextEmbedding(
        text="AI technology", embedding_vector=sample_embedding[0].embedding_vector
    )

    memories = await memory_storage.search_memories(
        user_id="test_user",
        text_embedding=query,
        topics=["AI"],
        limit=2,
    )

    assert len(memories) == 1
    assert memories[0].content == "Technical discussion about artificial intelligence"


@pytest.mark.asyncio
async def test_retrieve_memories_with_multiple_topics(memory_storage, sample_embedding):
    # Store memories with multiple topics
    memory1 = BaseMemoryInput(
        content="Memory about AI and robotics",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
        topics=["AI", "robotics"],
    )
    memory2 = BaseMemoryInput(
        content="Memory about AI and machine learning",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
        topics=["AI", "machine learning"],
    )
    memory3 = BaseMemoryInput(
        content="Memory about nature",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
        topics=["nature"],
    )

    await memory_storage.store_memory(memory1, embedding_vectors=sample_embedding)
    await memory_storage.store_memory(memory2, embedding_vectors=sample_embedding)
    await memory_storage.store_memory(memory3, embedding_vectors=sample_embedding)

    # Retrieve memories by AI topic (should get both AI-related memories)
    memories = await memory_storage.search_memories(
        user_id="test_user", topics=["AI"], limit=10
    )
    assert len(memories) == 2
    assert all("AI" in memory.topics for memory in memories)

    # Retrieve memories by robotics topic (should get only the robotics memory)
    memories = await memory_storage.search_memories(
        user_id="test_user", topics=["robotics"], limit=10
    )
    assert len(memories) == 1
    assert "robotics" in memories[0].topics
    assert memories[0].content == "Memory about AI and robotics"


@pytest.mark.asyncio
async def test_retrieve_memories_with_multiple_topics_parameter(
    memory_storage, sample_embedding
):
    # Store memories with multiple topics
    memory1 = BaseMemoryInput(
        content="Memory about AI and robotics",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
        topics=["AI", "robotics"],
    )
    memory2 = BaseMemoryInput(
        content="Memory about AI and machine learning",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
        topics=["AI", "machine learning"],
    )
    memory3 = BaseMemoryInput(
        content="Memory about nature",
        created_at=datetime.now(),
        user_id="test_user",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
        topics=["nature"],
    )

    await memory_storage.store_memory(memory1, embedding_vectors=sample_embedding)
    await memory_storage.store_memory(memory2, embedding_vectors=sample_embedding)
    await memory_storage.store_memory(memory3, embedding_vectors=sample_embedding)

    # Retrieve memories by multiple topics
    memories = await memory_storage.search_memories(
        user_id="test_user",
        topics=["AI", "robotics"],
        limit=10,
    )

    # Should get both AI-related memories
    assert len(memories) == 2
    assert any("robotics" in memory.topics for memory in memories)
    assert all("AI" in memory.topics for memory in memories)


@pytest.mark.asyncio
async def test_get_memories_by_user_id(
    memory_storage, sample_memory_input, sample_embedding
):
    # Store memories for different users
    user1_memory = BaseMemoryInput(
        content="User 1 memory",
        created_at=datetime.now(),
        user_id="user1",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
    )
    user2_memory = BaseMemoryInput(
        content="User 2 memory",
        created_at=datetime.now(),
        user_id="user2",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
    )
    user1_memory2 = BaseMemoryInput(
        content="User 1 second memory",
        created_at=datetime.now(),
        user_id="user1",
        memory_type=MemoryType.SEMANTIC,
        message_attributions=set(),
    )

    await memory_storage.store_memory(user1_memory, embedding_vectors=sample_embedding)
    await memory_storage.store_memory(user2_memory, embedding_vectors=sample_embedding)
    await memory_storage.store_memory(user1_memory2, embedding_vectors=sample_embedding)

    # Get memories for user1
    memories = await memory_storage.get_memories(user_id="user1")
    assert len(memories) == 2
    assert all(memory.user_id == "user1" for memory in memories)
    assert {memory.content for memory in memories} == {
        "User 1 memory",
        "User 1 second memory",
    }

    # Get memories for user2
    memories = await memory_storage.get_memories(user_id="user2")
    assert len(memories) == 1
    assert memories[0].user_id == "user2"
    assert memories[0].content == "User 2 memory"

    # Get memories for non-existent user
    memories = await memory_storage.get_memories(user_id="user3")
    assert len(memories) == 0

    # Test that providing neither parameter raises error
    with pytest.raises(
        ValueError, match="Either memory_ids or user_id must be provided"
    ):
        await memory_storage.get_memories()
