import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import litellm
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory_module import MemoryModule
from memory_module.core.memory_core import (
    MemoryCore,
    SemanticFact,
    SemanticMemoryExtraction,
)
from memory_module.core.message_queue import MessageQueue
from memory_module.interfaces.types import Message
from memory_module.services.llm_service import LLMService
from memory_module.storage.sqlite_memory_storage import SQLiteMemoryStorage
from memory_module.storage.sqlite_message_buffer_storage import (
    SQLiteMessageBufferStorage,
)

from tests.utils import build_llm_config


@pytest.fixture
def memory_module(monkeypatch):
    """Fixture to create a fresh MemoryModule instance for each test"""
    # path should be relative to the project root
    db_path = Path(__file__).parent / "data" / "tests" / "memory_module.db"
    # delete the db file if it exists
    if db_path.exists():
        db_path.unlink()
    storage = SQLiteMemoryStorage(db_path)
    config = build_llm_config({"model": "gpt-4o-mini"})

    # Only mock if api_key is not available
    if not config.get("api_key"):

        async def _mock_completion(**kwargs):
            return type(
                "MockModelResponse",
                (),
                {
                    "choices": [
                        type(
                            "MockChoices",
                            (),
                            {
                                "message": type(
                                    "MockChoiceMessage",
                                    (),
                                    {
                                        "content": SemanticMemoryExtraction(
                                            action="add",
                                            reason_for_action="Mocked LLM response about pie",
                                            interesting_facts=[
                                                SemanticFact(
                                                    text="Mocked LLM response about pie",
                                                    tags=[],
                                                )
                                            ],
                                        ).model_dump_json()
                                    },
                                )
                            },
                        )
                    ]
                },
            )

        monkeypatch.setattr(litellm, "acompletion", _mock_completion)

    llm_service = LLMService(**config)
    memory_core = MemoryCore(llm_service=llm_service, storage=storage)
    message_buffer_storage = SQLiteMessageBufferStorage(db_path)
    message_queue = MessageQueue(memory_core=memory_core, message_buffer_storage=message_buffer_storage)
    return MemoryModule(llm_service=llm_service, memory_core=memory_core, message_queue=message_queue)


@pytest.mark.asyncio
async def test_simple_conversation(memory_module):
    """Test a simple conversation about pie"""
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
    # contains pie
    assert any("pie" in message.content for message in stored_messages)
    # contains one of the messages at least in its attributions
    assert any(message.id in stored_messages[0].message_attributions for message in messages)
    # all stored_memories are memory_type == "semantic"
    assert all(memory.memory_type == "semantic" for memory in stored_messages)


@pytest.mark.asyncio
async def test_episodic_memory_creation(memory_module):
    """Test that episodic memory creation raises NotImplementedError"""
    conversation_id = str(uuid4())

    # Create 5 messages (default buffer size) for the same conversation
    messages = [
        Message(
            id=str(uuid4()),
            content=f"Message {i} about pie",
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
            role="user",
        )
        for i in range(5)
    ]

    # Add messages one by one, expecting the last one to raise NotImplementedError
    # when it tries to process the episodic memory
    for i, message in enumerate(messages):
        if i < 4:
            await memory_module.add_message(message)
        else:
            with pytest.raises(
                NotImplementedError,
                match="Episodic memory extraction not yet implemented",
            ):
                await memory_module.add_message(message)
