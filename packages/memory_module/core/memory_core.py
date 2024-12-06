from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from memory_module.interfaces.base_memory_core import BaseMemoryCore
from memory_module.interfaces.types import Memory, Message
from memory_module.services.llm_service import LLMService
from memory_module.storage.sqlite_memory_storage import SQLiteMemoryStorage


class MessageMemoryExtraction(BaseModel):
    """Model for LLM extraction of memories from messages."""

    summary: str
    importance: float
    key_points: List[str]


class SemanticFact(BaseModel):
    text: str = Field(
        ...,
        description="The text of the fact. Do not use real names (you can say 'The user' instead) and avoid pronouns.",
    )
    tags: List[str]


class SemanticMemoryExtraction(BaseModel):
    action: Literal["add", "ignore"] = Field(..., description="Action to take on the extracted fact")
    reason_for_action: Optional[str] = Field(
        ...,
        description="Reason for the action taken on the extracted fact or the reason it was ignored.",
    )
    interesting_facts: Optional[List[SemanticFact]] = Field(
        ..., description="One or more interesting fact extracted from the message."
    )


class MemoryCore(BaseMemoryCore):
    """Implementation of the memory core component."""

    def __init__(
        self,
        llm_service: LLMService,
        storage: Optional[SQLiteMemoryStorage] = None,
    ):
        self.lm = llm_service
        self.storage = storage if storage is not None else SQLiteMemoryStorage()

    async def process_semantic_messages(self, messages: List[Message]) -> None:
        """Process multiple messages into semantic memories (general facts, preferences)."""
        for message in messages:
            extraction = await self._extract_semantic_fact_from_message(message)

            if extraction.action == "add" and extraction.interesting_facts:
                for fact in extraction.interesting_facts:
                    memory = Memory(
                        content=fact.text,
                        created_at=message.created_at,
                        message_attributions=[message.id],
                        memory_type="semantic",
                    )
                    await self.storage.store_memory(memory)

    async def process_episodic_messages(self, messages: List[Message]) -> None:
        """Process multiple messages into episodic memories (specific events, experiences)."""
        # TODO: Implement episodic memory processing
        await self._extract_episodic_memory_from_message(messages)

    async def retrieve(self, query: str) -> List[Memory]:
        """Retrieve memories based on a query.

        Steps:
        1. Convert query to embedding
        2. Find relevant memories
        3. Possibly rerank or filter results
        """
        # TODO: Implement memory retrieval logic
        pass

    async def _extract_memory_from_messages(self, messages: List[Message]) -> MessageMemoryExtraction:
        """Extract meaningful information from messages using LLM."""
        # TODO: Implement LLM-based extraction
        pass

    async def _create_memory_embedding(self, content: str) -> List[float]:
        """Create embedding for memory content."""
        # TODO: Implement embedding creation
        pass

    async def _extract_semantic_fact_from_message(
        self, message: Message, memory_message: str = ""
    ) -> SemanticMemoryExtraction:
        """Extract semantic facts from a message using LLM.

        Args:
            message: The message to extract facts from
            memory_message: Optional context from previous memories

        Returns:
            SemanticMemoryExtraction containing the action and extracted facts
        """
        system_message = f"""You are a semantic memory management agent. Your goal is to extract meaningful, long-term facts and preferences from user messages. Focus on recognizing general patterns and interests that will remain relevant over time, even if the user is mentioning short-term plans or events.

Prioritize:
General Interests and Preferences: When a user mentions specific events or actions, focus on the underlying interests, hobbies, or preferences they reveal (e.g., if the user mentions attending a conference, focus on the topic of the conference, not the date or location).
Facts or Details about user: Extract facts that describe long-term information about the user, such as details about things they own.
Long-Term Facts: Extract facts that describe long-term information about the user, such as their likes, dislikes, or ongoing activities.
Ignore Short-Term Details: Avoid storing short-term specifics like dates or locations unless they reflect a recurring activity or long-term plan.

{memory_message}
Here is the latest message that was sent:
User: {message.content}
"""  # noqa: E501

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": "Please analyze this message and decide whether to extract facts or ignore it. If extracting facts, provide one or more semantic facts focusing on long-term, meaningful information.",  # noqa: E501
            },
        ]

        res = await self.lm.completion(messages=messages, response_model=SemanticMemoryExtraction)

        return res

    async def _extract_episodic_memory_from_message(
        self, message: Message, memory_message: str = ""
    ) -> SemanticMemoryExtraction:
        """Extract episodic memories from a message using LLM."""
        # TODO: Implement episodic memory extraction
        raise NotImplementedError("Episodic memory extraction not yet implemented")
