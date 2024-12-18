import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory_module.config import MemoryModuleConfig
from memory_module.core.memory_core import (
    EpisodicMemoryExtraction,
    MessageDigest,
    SemanticFact,
    SemanticMemoryExtraction,
)
from memory_module.core.memory_module import MemoryModule
from memory_module.interfaces.types import (
    ShortTermMemoryRetrievalConfig,
    UserMessageInput,
)

from tests.memory_module.utils import build_llm_config

logger = logging.getLogger(__name__)


@pytest.fixture
def config():
    """Fixture to create test config."""
    llm_config = build_llm_config()
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

        async def _mock_extract_semantic_fact_from_messages(messages, **kwargs):
            return SemanticMemoryExtraction(
                action="add",
                reason_for_action="Mocked LLM response about pie",
                facts=[
                    SemanticFact(
                        text="Mocked LLM response about pie",
                        tags=[],
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

        monkeypatch.setattr(memory_module.memory_core.lm, "embedding", _mock_embedding)

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
    assert len(stored_memories) == 2
    assert any("pie" in message.content for message in stored_memories)
    assert any(message.id in stored_memories[0].message_attributions for message in messages)
    assert all(memory.memory_type == "semantic" for memory in stored_memories)

    result = await memory_module.retrieve_memories("apple pie", "", 1)
    assert len(result) == 1
    assert result[0].id == next(memory.id for memory in stored_memories if "apple pie" in memory.content)


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
            created_at=datetime.now(),
        )
        for i in range(3)
    ]

    # Add messages one by one
    for message in messages:
        await memory_module.add_message(message)

    # Check short-term memory using retrieve method
    chat_history_messages = await memory_module.retrieve_chat_history(
        conversation_id, ShortTermMemoryRetrievalConfig(last_minutes=1)
    )
    assert len(chat_history_messages) == 3
    assert all(msg in chat_history_messages for msg in messages)

    # Verify messages are in reverse order
    reversed_messages = messages[::-1]
    for i, _msg in enumerate(reversed_messages):
        assert chat_history_messages[i].id == reversed_messages[i].id
