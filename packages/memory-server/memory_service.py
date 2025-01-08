from datetime import datetime
import os
import sys
from typing import Optional
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from memory_module import MemoryModule, MemoryModuleConfig, LLMConfig
from memory_module.interfaces.types import MessageInput, AssistantMessageInput, UserMessageInput

class MemoryService:
    def __init__(self, openai_api_key: str):
        # Initialize memory module
        config = MemoryModuleConfig(
            db_path=os.path.join(os.path.dirname(__file__), "../memory_module/data/memory.db"),
            llm=LLMConfig(
            model="gpt-4o",
            embedding_model="text-embedding-3-small",
            api_key=openai_api_key
        ))
        self._memory = MemoryModule(config)

    async def add_message(self, type: str, content: str):
        message: MessageInput
        if type == "assistant":
            message = AssistantMessageInput(id=str(uuid.uuid4()), content=content, author_id="1", conversation_ref="1")
        elif type == "user":
            message = UserMessageInput(id=str(uuid.uuid4()), content=content, author_id="1", conversation_ref="1", created_at=datetime.now())
        else:
            raise ValueError("Invalid message type")

        await self._memory.add_message(message)
    
    async def retrieve_memories(self, query: str, user_id: str, limit: Optional[int] = None):
        return await self._memory.retrieve_memories(query, user_id=user_id, limit=limit)
    
    async def get_all_memories(self, user_id: str):
        return await self._memory.get_user_memories(user_id=user_id)