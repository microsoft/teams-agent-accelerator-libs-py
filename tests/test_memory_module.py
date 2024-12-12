import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import litellm
import pytest
import pytest_asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory_module import MemoryModule
from memory_module.config import MemoryModuleConfig
from memory_module.core.memory_core import (
    EpisodicMemoryExtraction,
    SemanticFact,
    SemanticMemoryExtraction,
)
from memory_module.interfaces.types import Message

from tests.utils import build_llm_config

litellm.set_verbose = True


@pytest.fixture
def config():
    """Fixture to create test config."""
    llm_config = build_llm_config({"model": "gpt-4o-mini"})
    if not llm_config.api_key:
        pytest.skip("OpenAI API key not provided")
    return MemoryModuleConfig(
        db_path=Path(__file__).parent / "data" / "tests" / "memory_module.db",
        buffer_size=5,
        timeout_seconds=1,  # Short timeout for testing
        llm=llm_config,
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

        async def _mock_semantic_memory_extraction(message, **kwargs):
            return SemanticMemoryExtraction(
                action="add",
                reason_for_action="Mocked LLM response about pie",
                interesting_facts=[
                    SemanticFact(
                        text="Mocked LLM response about pie",
                        tags=[],
                    )
                ],
            )

        monkeypatch.setattr(
            memory_module.memory_core, "_extract_semantic_fact_from_message", _mock_semantic_memory_extraction
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

    return memory_module


@pytest_asyncio.fixture(autouse=True)
async def cleanup_scheduled_events(memory_module):
    """Fixture to cleanup scheduled events after each test."""
    yield
    await memory_module.message_queue.message_buffer.scheduler.cleanup()


@pytest.mark.asyncio
async def test_simple_conversation(memory_module):
    """Test a simple conversation about pie."""
    conversation_id = str(uuid4())
    messages = [
        Message(
            id=str(uuid4()),
            content="I love pie!",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
        ),
        Message(
            id=str(uuid4()),
            content="Apple pie is the best!",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
        ),
    ]

    for message in messages:
        await memory_module.add_message(message)

    stored_messages = await memory_module.memory_core.storage.get_all_memories()
    assert len(stored_messages) == 2
    assert any("pie" in message.content for message in stored_messages)
    assert any(message.id in stored_messages[0].message_attributions for message in messages)
    assert all(memory.memory_type == "semantic" for memory in stored_messages)

    result = await memory_module.retrieve_memories("apple pie", "", 1)
    assert len(result) == 1
    assert result[0].id == 2


@pytest.mark.asyncio
async def test_no_memories_found():
    # TODO: Implement test for no memories found
    pass


# TODO: Add test for episodic memory extraction once `MemoryCore.process_episodic_messages` is implemented.
# @pytest.mark.asyncio
# async def test_episodic_memory_creation(memory_module):
#     """Test that episodic memory creation raises NotImplementedError."""
#     conversation_id = str(uuid4())

#     messages = [
#         Message(
#             id=str(uuid4()),
#             content=f"Message {i} about pie",
#             author_id="user-123",
#             conversation_ref=conversation_id,
#             created_at=datetime.now(),
#             role="user",
#         )
#         for i in range(5)
#     ]

#     for i, message in enumerate(messages):
#         if i < 4:
#             await memory_module.add_message(message)
#         else:
#             with pytest.raises(NotImplementedError, match="Episodic memory extraction not yet implemented"):
#                 await memory_module.add_message(message)


@pytest.mark.asyncio
async def test_episodic_memory_timeout(memory_module, config, monkeypatch):
    """Test that episodic memory is triggered after timeout."""
    # Mock the episodic memory extraction
    extraction_called = False

    async def mock_extract_episodic(*args, **kwargs):
        nonlocal extraction_called
        extraction_called = True

    monkeypatch.setattr(memory_module.memory_core, "_extract_episodic_memory_from_messages", mock_extract_episodic)

    conversation_id = str(uuid4())
    messages = [
        Message(
            id=str(uuid4()),
            content=f"Message {i} about pie",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
            role="user",
        )
        for i in range(3)
    ]

    for message in messages:
        await memory_module.add_message(message)

    await asyncio.sleep(1.5)

    assert extraction_called, "Episodic memory extraction should have been triggered by timeout"


@pytest.mark.asyncio
async def test_update_memory(memory_module):
    """Test memory update"""
    conversation_id = str(uuid4())
    messages = [
        Message(
            id=str(uuid4()),
            content="Seattle is my favorite city!",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
        ),
    ]

    for message in messages:
        await memory_module.add_message(message)

    await memory_module.update_memories(1, "The user like San Diego city")
    updated_message = await memory_module.memory_core.storage.get_memory(1)
    assert "San Diego" in updated_message.content


@pytest.mark.asyncio
async def test_remove_memory(memory_module):
    """Test a simple conversation removal based on user id."""
    conversation_id = str(uuid4())
    messages = [
        Message(
            id=str(uuid4()),
            content="I like pho a lot!",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
        ),
        Message(
            id=str(uuid4()),
            content="I prefer noodle soup!",
            author_id="user-456",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
        ),
        Message(
            id=str(uuid4()),
            content="Dim sum is also my favorite!",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
        ),
    ]

    for message in messages:
        await memory_module.add_message(message)

    stored_messages = await memory_module.memory_core.storage.get_all_memories()
    assert len(stored_messages) == 3

    await memory_module.remove_memories("user-123")

    stored_messages = await memory_module.memory_core.storage.get_all_memories()
    assert len(stored_messages) == 1


@pytest.mark.asyncio
async def test_short_term_memory(memory_module):
    """Test that messages are stored in short-term memory."""
    conversation_id = str(uuid4())
    messages = [
        Message(
            id=str(uuid4()),
            content=f"Test message {i}",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
            role="user",
        )
        for i in range(3)
    ]

    # Add messages one by one
    for message in messages:
        await memory_module.add_message(message)

    # Check short-term memory using retrieve method
    short_term_memories = await memory_module.retrieve_short_term_memories()
    assert len(short_term_memories) == 3
    assert all(msg in short_term_memories for msg in messages)

    # Verify messages are in correct order
    for i, msg in enumerate(messages):
        assert short_term_memories[i] == messages[i]
