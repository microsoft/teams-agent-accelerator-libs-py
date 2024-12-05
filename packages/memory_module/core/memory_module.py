from pathlib import Path
from typing import List, Optional

from memory_module.core.memory_core import MemoryCore
from memory_module.interfaces.base_memory_module import BaseMemoryModule
from memory_module.interfaces.types import Memory, Message
from memory_module.services.llm_service import LLMService


class MemoryModule(BaseMemoryModule):
    """Implementation of the memory module interface."""

    def __init__(
        self,
        llm_service: LLMService,
        db_path: Optional[str | Path] = None,
        memory_core: Optional[MemoryCore] = None,
    ):
        """Initialize the memory module.

        Args:
            db_path: Path to the SQLite database
            memory_core: Optional MemoryCore instance. If not provided, one will be created.
        """
        self.llm_service = llm_service
        if memory_core is None:
            self.memory_core = MemoryCore(llm_service=llm_service)
        else:
            self.memory_core = memory_core

    async def add_message(self, message: Message) -> None:
        """Add a message to be processed into memory."""
        await self.memory_core.process_messages([message])

    async def retrieve_memories(
        self, query: str, user_id: Optional[str]
    ) -> List[Memory]:
        """Retrieve relevant memories based on a query."""
        return await self.memory_core.retrieve(query, user_id)