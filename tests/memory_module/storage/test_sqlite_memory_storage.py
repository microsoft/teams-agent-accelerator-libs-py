from datetime import datetime

from memory_module.interfaces.types import Message
from memory_module.storage.sqlite_memory_storage import SQLiteMemoryStorage


async def test_get_messages():
    # Initialize storage
    storage = SQLiteMemoryStorage(":memory:")

    # Test data
    test_messages = [
        Message(
            id="msg1",
            content="Test message 1",
            author_id="user1",
            conversation_ref="conv1",
            created_at=datetime.now(),
            is_assistant_message=False,
            deep_link="link1",
        ),
        Message(
            id="msg2",
            content="Test message 2",
            author_id="user1",
            conversation_ref="conv1",
            created_at=datetime.now(),
            is_assistant_message=True,
            deep_link="link2",
        ),
    ]

    # Store test messages
    for message in test_messages:
        await storage.store_short_term_memory(message)

    # Create memory attributions
    async with storage.storage.transaction() as cursor:
        # Create test memories
        await cursor.execute(
            "INSERT INTO memories (id, content, created_at, user_id, memory_type) VALUES (?, ?, ?, ?, ?)",
            (1, "Memory 1", datetime.now(), "user1", "EXTRACTED"),
        )
        await cursor.execute(
            "INSERT INTO memories (id, content, created_at, user_id, memory_type) VALUES (?, ?, ?, ?, ?)",
            (2, "Memory 2", datetime.now(), "user1", "EXTRACTED"),
        )

        # Link messages to memories
        await cursor.execute("INSERT INTO memory_attributions (memory_id, message_id) VALUES (?, ?)", (1, "msg1"))
        await cursor.execute("INSERT INTO memory_attributions (memory_id, message_id) VALUES (?, ?)", (1, "msg2"))
        await cursor.execute("INSERT INTO memory_attributions (memory_id, message_id) VALUES (?, ?)", (2, "msg2"))

    # Test retrieving messages
    result = await storage.get_messages([1, 2])

    # Assertions
    assert len(result) == 2
    assert len(result[1]) == 2  # Memory 1 should have 2 messages
    assert len(result[2]) == 1  # Memory 2 should have 1 message

    # Verify message content
    assert result[1][0].id == "msg1"
    assert result[1][1].id == "msg2"
    assert result[2][0].id == "msg2"
