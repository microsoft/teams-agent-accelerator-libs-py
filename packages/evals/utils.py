import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory_module.config import LLMConfig, MemoryModuleConfig
from memory_module.core.memory_module import MemoryModule
from memory_module.interfaces.types import AssistantMessage, UserMessage

from .helpers import SessionMessage


class MemoryModuleManager:
    def __init__(self, buffer_size=5):
        self._buffer_size = buffer_size
        self._memory_module: Optional[MemoryModule] = None
        self._db_path = Path(__file__).parent / "data" / f"memory_{uuid.uuid4().hex}.db"

    def __enter__(self):
        # Create memory module
        llm = LLMConfig(
            model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        config = MemoryModuleConfig(db_path=self._db_path, buffer_size=self._buffer_size, llm=llm)

        self._memory_module = MemoryModule(config=config)
        return self._memory_module

    def __exit__(self, exc_type, exc_value, traceback):
        # Destroy memory module and database
        del self._memory_module
        os.remove(self._db_path)


async def add_messages(memory_module: MemoryModule, messages: List[SessionMessage]):
    def create_message(**kwargs):
        params = {
            "id": str(uuid.uuid4()),
            "content": kwargs["content"],
            "author_id": "user",
            "created_at": datetime.now(),
            "conversation_ref": "conversation_ref",
        }
        if kwargs["type"] == "assistant":
            return AssistantMessage(**params)
        else:
            return UserMessage(**params)

    for message in messages:
        type = "assistant" if message["role"] == "assistant" else "user"
        msg = create_message(content=message["content"], type=type)
        await memory_module.add_message(msg)
