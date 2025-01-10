import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from uuid import uuid4

import pytest
import pytest_asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory_module.config import DEFAULT_TOPICS, MemoryModuleConfig, Topic  # noqa: I001
from memory_module.core.memory_core import (
    EpisodicMemoryExtraction,
    MemoryCore,
    MessageDigest,
    SemanticFact,
    SemanticMemoryExtraction,
)
from memory_module.core.memory_module import MemoryModule
from memory_module.interfaces.types import (
    AssistantMessageInput,
    RetrievalConfig,
    ShortTermMemoryRetrievalConfig,
    UserMessageInput,
)

from tests.memory_module.utils import build_llm_config

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.asyncio(scope="session")


@pytest.fixture
def config(request):
    """Fixture to create test config."""
    params = request.param if hasattr(request, "param") else {}
    llm_config = build_llm_config()
    buffer_size = params.get("buffer_size", 5)
    timeout_seconds = params.get("timeout_seconds", 60)
    topics = params.get("topics", DEFAULT_TOPICS)
    if not llm_config.api_key:
        pytest.skip("OpenAI API key not provided")
    return MemoryModuleConfig(
        db_path=Path(__file__).parent / "data" / "tests" / "memory_module.db",
        buffer_size=buffer_size,
        timeout_seconds=timeout_seconds,
        llm=llm_config,
        topics=topics,
    )


@pytest.fixture
def memory_module(config, monkeypatch):
    """Fixture to create a fresh MemoryModule instance for each test."""
    # Delete the db file if it exists
    if config.db_path.exists():
        config.db_path.unlink()

    memory_module = MemoryModule(config=config)

    # Only mock if api_key is not available
    if not config.llm.api_key:

        async def _mock_extract_semantic_fact_from_messages(messages, **kwargs):
            return SemanticMemoryExtraction(
                action="add",
                reason_for_action="Mocked LLM response about pie",
                facts=[
                    SemanticFact(
                        text="Mocked LLM response about pie",
                        message_indices=[0, 1],
                    )
                ],
            )

        monkeypatch.setattr(
            memory_module.memory_core, "_extract_semantic_fact_from_messages", _mock_extract_semantic_fact_from_messages
        )

        async def _mock_episodic_memory_extraction(messages, **kwargs):
            return EpisodicMemoryExtraction(
                action="add",
                reason_for_action="Mocked LLM response about pie",
                summary="Mocked LLM response about pie",
            )

        monkeypatch.setattr(
            memory_module.memory_core, "_extract_episodic_memory_from_messages", _mock_episodic_memory_extraction
        )

        async def _mock_extract_metadata_from_fact(fact: SemanticFact, **kwargs):
            return MessageDigest(
                topic="Mocked LLM response about pie",
                summary="Mocked LLM response about pie",
                keywords=["pie", "apple pie"],
                hypothetical_questions=["What food does the user like?"],
            )

        monkeypatch.setattr(memory_module.memory_core, "_extract_metadata_from_fact", _mock_extract_metadata_from_fact)

        async def _mock_embedding(**kwargs):
            return type(
                "EmbeddingResponse",
                (object,),
                {
                    "data": [
                        {"embedding": [0.1, 0.2, 0.3]},
                        {"embedding": [0.4, 0.5, 0.6]},
                        {"embedding": [0.7, 0.8, 0.9]},
                        {"embedding": [1.0, 1.1, 1.2]},
                    ]
                },
            )

        assert isinstance(memory_module.memory_core, MemoryCore)
        monkeypatch.setattr(memory_module.memory_core.lm, "embedding", _mock_embedding)

    return memory_module


@pytest_asyncio.fixture(autouse=True)
async def cleanup_scheduled_events(memory_module):
    """Fixture to cleanup scheduled events after each test."""
    try:
        yield
    finally:
        await memory_module.message_queue.message_buffer.scheduler.cleanup()


@pytest.mark.asyncio
async def test_simple_conversation(memory_module):
    """Test a simple conversation about pie."""
    conversation_id = str(uuid4())
    messages = [
        UserMessageInput(
            id=str(uuid4()),
            content="I love pie!",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
        ),
        UserMessageInput(
            id=str(uuid4()),
            content="Apple pie is the best!",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
        ),
    ]

    for message in messages:
        await memory_module.add_message(message)

    await memory_module.message_queue.message_buffer.scheduler.flush()
    stored_memories = await memory_module.memory_core.storage.get_all_memories()
    assert len(stored_memories) >= 1
    assert any("pie" in message.content for message in stored_memories)
    assert any(message.id in stored_memories[0].message_attributions for message in messages)
    assert all(memory.memory_type == "semantic" for memory in stored_memories)

    result = await memory_module.retrieve_memories("user-123", RetrievalConfig(query="apple pie", limit=1))
    assert len(result) == 1
    assert result[0].id == next(memory.id for memory in stored_memories if "apple pie" in memory.content)


@pytest.mark.asyncio
async def test_no_memories_found():
    # TODO: Implement test for no memories found
    pass


@pytest.mark.asyncio
async def test_episodic_memory_timeout(memory_module, config, monkeypatch):
    """Test that episodic memory is triggered after timeout."""
    pytest.skip(
        "Skipping episodic memory timeout test. We are debating if we need to build long-term episodic memories or not."
    )
    # Mock the episodic memory extraction
    extraction_called = False

    async def mock_extract_episodic(*args, **kwargs):
        nonlocal extraction_called
        extraction_called = True

    monkeypatch.setattr(memory_module.memory_core, "_extract_episodic_memory_from_messages", mock_extract_episodic)

    conversation_id = str(uuid4())
    messages = [
        UserMessageInput(
            id=str(uuid4()),
            content=f"Message {i} about pie",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
        )
        for i in range(3)
    ]

    for message in messages:
        await memory_module.add_message(message)

    await memory_module.message_queue.message_buffer.scheduler.flush()
    assert extraction_called, "Episodic memory extraction should have been triggered by timeout"


@pytest.mark.asyncio
async def test_update_memory(memory_module):
    """Test memory update"""
    conversation_id = str(uuid4())
    messages = [
        UserMessageInput(
            id=str(uuid4()),
            content="Seattle is my favorite city!",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
        ),
    ]

    for message in messages:
        await memory_module.add_message(message)

    await memory_module.message_queue.message_buffer.scheduler.flush()
    stored_memories = await memory_module.memory_core.storage.get_all_memories()
    assert len(stored_memories) >= 1

    memory_id = next(memory.id for memory in stored_memories if "Seattle" in memory.content)
    await memory_module.update_memory(memory_id, "The user like San Diego city")
    updated_message = await memory_module.memory_core.storage.get_memory(memory_id)
    assert "San Diego" in updated_message.content


@pytest.mark.asyncio
async def test_remove_memory(memory_module):
    """Test a simple conversation removal based on user id."""
    conversation_id = str(uuid4())
    messages = [
        UserMessageInput(
            id=str(uuid4()),
            content="I like pho a lot!",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
        ),
    ]

    for message in messages:
        await memory_module.add_message(message)
    await memory_module.message_queue.message_buffer.scheduler.flush()
    stored_messages = await memory_module.memory_core.storage.get_all_memories()
    assert len(stored_messages) >= 0

    await memory_module.remove_memories("user-123")

    stored_messages = await memory_module.memory_core.storage.get_all_memories()
    assert len(stored_messages) == 0


@pytest.mark.asyncio
async def test_short_term_memory(memory_module):
    """Test that messages are stored in short-term memory."""
    conversation_id = str(uuid4())
    messages = [
        UserMessageInput(
            id=str(uuid4()),
            content=f"Test message {i}",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now() + timedelta(seconds=-i * 25),
        )
        for i in range(4)
    ]

    # Add messages one by one
    for message in messages:
        await memory_module.add_message(message)

    # Check short-term memory using retrieve method
    chat_history_messages = await memory_module.retrieve_chat_history(
        conversation_id, ShortTermMemoryRetrievalConfig(last_minutes=1)
    )
    assert len(chat_history_messages) == 3

    # Verify messages are in reverse order
    expected_messages = messages[1:3][::-1]  # 3 messages because we only have 3 messages in the last minute
    for i, _msg in enumerate(expected_messages):
        assert chat_history_messages[i].id == expected_messages[i].id


@pytest.mark.asyncio
async def test_add_memory_processing_decision(memory_module):
    """Test whether to process adding memory"""

    async def _validate_decision(memory_module, message: List[UserMessageInput], expected_decision: str):
        extraction = await memory_module.memory_core._extract_semantic_fact_from_messages(message)
        assert extraction.action == "add" and extraction.facts
        for fact in extraction.facts:
            decision = await memory_module.memory_core._get_add_memory_processing_decision(fact, "user-123")
            if decision.decision != expected_decision:
                # Adding this because this test is flaky and it would be good to know why.
                print(f"Decision: {decision}, Expected: {expected_decision}", fact, decision)
            assert decision.decision == expected_decision

    conversation_id = str(uuid4())
    old_messages = [
        UserMessageInput(
            id=str(uuid4()),
            content="I have a Pokemon limited version Macbook.",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now() - timedelta(minutes=3),
        ),
        UserMessageInput(
            id=str(uuid4()),
            content="I bought a pink iphone.",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now() - timedelta(minutes=2),
        ),
        UserMessageInput(
            id=str(uuid4()),
            content="I just bought a Macbook.",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now() - timedelta(minutes=1),
        ),
    ]
    new_messages = [
        [
            UserMessageInput(
                id=str(uuid4()),
                content="I have a Macbook",
                author_id="user-123",
                conversation_ref=conversation_id,
                created_at=datetime.now(),
            )
        ],
        [
            UserMessageInput(
                id=str(uuid4()),
                content="I like cats",
                author_id="user-123",
                conversation_ref=conversation_id,
                created_at=datetime.now(),
            )
        ],
    ]

    for message in old_messages:
        await memory_module.add_message(message)

    await memory_module.message_queue.message_buffer.scheduler.flush()

    await _validate_decision(memory_module, new_messages[0], "ignore")
    await _validate_decision(memory_module, new_messages[1], "add")


@pytest.mark.asyncio
async def test_remove_messages(memory_module):
    conversation1_id = str(uuid4())
    conversation2_id = str(uuid4())
    conversation3_id = str(uuid4())
    message1_id = str(uuid4())
    message2_id = str(uuid4())
    message3_id = str(uuid4())
    message4_id = str(uuid4())
    messages = [
        UserMessageInput(
            id=message1_id,
            content="I like strawberry flavor ice cream a lot.",
            author_id="user-123",
            conversation_ref=conversation1_id,
            created_at=datetime.now(),
        ),
        UserMessageInput(
            id=message2_id,
            content="I like eating noodle.",
            author_id="user-123",
            conversation_ref=conversation1_id,
            created_at=datetime.now(),
        ),
    ]
    for message in messages:
        await memory_module.add_message(message)
    await memory_module.message_queue.message_buffer.scheduler.flush()

    messages2 = [
        UserMessageInput(
            id=message3_id,
            content="I like to go TT for grocery shopping.",
            author_id="user-123",
            conversation_ref=conversation2_id,
            created_at=datetime.now(),
        ),
        UserMessageInput(
            id=message4_id,
            content="I like pancake from TT.",
            author_id="user-123",
            conversation_ref=conversation3_id,
            created_at=datetime.now(),
        ),
    ]

    for message in messages2:
        await memory_module.add_message(message)

    stored_memories = await memory_module.memory_core.storage.get_all_memories()
    assert len(stored_memories) == 2
    stored_buffer = await memory_module.message_queue.message_buffer.storage.get_conversations_from_buffered_messages(
        [message3_id, message4_id]
    )
    assert len(list(stored_buffer.keys())) == 2

    remove_messages = [message1_id, message3_id]

    await memory_module.remove_messages(remove_messages)

    updated_memories = await memory_module.memory_core.storage.get_all_memories()
    assert len(updated_memories) == 1
    assert any("noodle" in memory.content for memory in updated_memories)
    assert not any("strawberry" in memory.content for memory in updated_memories)

    updated_buffer = await memory_module.message_queue.message_buffer.storage.get_conversations_from_buffered_messages(
        [message3_id, message4_id]
    )
    conversation_refs = list(updated_buffer.keys())
    assert len(conversation_refs) == 1
    assert conversation_refs[0] == conversation3_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {
            "topics": [
                Topic(name="Device Type", description="The type of device the user has"),
                Topic(name="Operating System", description="The user's operating system"),
                Topic(name="Device year", description="The year of the user's device"),
            ],
            "buffer_size": 10,
        }
    ],
    indirect=True,
)
async def test_topic_extraction(memory_module):
    conversation_id = str(uuid4())
    messages = [
        {"role": "user", "content": "I need help with my device..."},
        {"role": "assistant", "content": "I'm sorry to hear that. What device do you have?"},
        {"role": "user", "content": "I have a Macbook"},
        {"role": "assistant", "content": "What is the year of your device?"},
        {"role": "user", "content": "2024"},
    ]

    for message in messages:
        if message["role"] == "user":
            input = UserMessageInput(
                id=str(uuid4()),
                content=message["content"],
                author_id="user-123",
                conversation_ref=conversation_id,
                created_at=datetime.now(),
            )
        else:
            input = AssistantMessageInput(
                id=str(uuid4()),
                content=message["content"],
                author_id="user-123",
                conversation_ref=conversation_id,
                created_at=datetime.now(),
            )
        await memory_module.add_message(input)

    await memory_module.message_queue.message_buffer.scheduler.flush()
    stored_memories = await memory_module.memory_core.storage.get_all_memories()
    assert any("macbook" in message.content.lower() for message in stored_memories)
    assert any("2024" in message.content for message in stored_memories)

    # Add assertions for topics
    device_type_memory = next((m for m in stored_memories if "macbook" in m.content.lower()), None)
    year_memory = next((m for m in stored_memories if "2024" in m.content), None)

    assert device_type_memory is not None and "Device Type" in device_type_memory.topics
    assert year_memory is not None and "Device year" in year_memory.topics


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {
            "topics": [
                Topic(name="Device Type", description="The type of device the user has"),
                Topic(name="Operating System", description="The user's operating system"),
                Topic(name="Device year", description="The year of the user's device"),
            ],
            "buffer_size": 10,
        }
    ],
    indirect=True,
)
async def test_retrieve_memories_by_topic(memory_module):
    """Test retrieving memories by topic only."""
    conversation_id = str(uuid4())
    messages = [
        UserMessageInput(
            id=str(uuid4()),
            content="I use Windows 11 on my PC",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now() - timedelta(minutes=5),
        ),
        UserMessageInput(
            id=str(uuid4()),
            content="I have a MacBook Pro from 2023",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now() - timedelta(minutes=3),
        ),
        UserMessageInput(
            id=str(uuid4()),
            content="My MacBook runs macOS Sonoma",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now() - timedelta(minutes=1),
        ),
    ]

    for message in messages:
        await memory_module.add_message(message)
    await memory_module.message_queue.message_buffer.scheduler.flush()

    # Retrieve memories by Operating System topic
    os_memories = await memory_module.retrieve_memories(
        "user-123",
        RetrievalConfig(topic=Topic(name="Operating System", description="The user's operating system")),
    )

    assert all("Operating System" in memory.topics for memory in os_memories)
    assert any("windows 11" in memory.content.lower() for memory in os_memories)
    assert any("sonoma" in memory.content.lower() for memory in os_memories)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {
            "topics": [
                Topic(name="Device Type", description="The type of device the user has"),
                Topic(name="Operating System", description="The user's operating system"),
                Topic(name="Device year", description="The year of the user's device"),
            ],
            "buffer_size": 10,
        }
    ],
    indirect=True,
)
async def test_retrieve_memories_by_topic_and_query(memory_module):
    """Test retrieving memories using both topic and semantic search."""
    conversation_id = str(uuid4())
    messages = [
        UserMessageInput(
            id=str(uuid4()),
            content="I use Windows 11 on my gaming PC",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now() - timedelta(minutes=5),
        ),
        UserMessageInput(
            id=str(uuid4()),
            content="I have a MacBook Pro from 2023",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now() - timedelta(minutes=3),
        ),
        UserMessageInput(
            id=str(uuid4()),
            content="My MacBook runs macOS Sonoma",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now() - timedelta(minutes=1),
        ),
    ]

    for message in messages:
        await memory_module.add_message(message)
    await memory_module.message_queue.message_buffer.scheduler.flush()

    # make sure we have memories
    stored_memories = await memory_module.memory_core.storage.get_all_memories()
    assert any("macbook" in memory.content.lower() for memory in stored_memories)
    assert any("windows" in memory.content.lower() for memory in stored_memories)

    # Retrieve memories by Operating System topic AND query about Mac
    memories = await memory_module.retrieve_memories(
        "user-123",
        RetrievalConfig(
            topic=Topic(name="Operating System", description="The user's operating system"),
            query="MacBook",
        ),
    )
    assert len(memories) > 0, f"No memories found for MacBook, check out stored memories: {stored_memories}"
    most_relevant_memory = memories[0]
    assert "sonoma" in most_relevant_memory.content.lower()
    assert "windows" not in most_relevant_memory.content.lower()

    # Try another query within the same topic
    windows_memories = await memory_module.retrieve_memories(
        "user-123",
        RetrievalConfig(
            topic=Topic(name="Operating System", description="The user's operating system"),
            query="PC",
        ),
    )
    assert len(windows_memories) > 0, f"No memories found for Windows, check out stored memories: {stored_memories}"
    most_relevant_memory = windows_memories[0]
    assert "windows" in most_relevant_memory.content.lower()
    assert "sonoma" not in most_relevant_memory.content.lower()
