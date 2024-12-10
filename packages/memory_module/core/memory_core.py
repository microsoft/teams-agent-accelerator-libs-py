from typing import List, Literal, Optional

from litellm import EmbeddingResponse
from pydantic import BaseModel, Field

from memory_module.config import MemoryModuleConfig
from memory_module.interfaces.base_memory_core import BaseMemoryCore
from memory_module.interfaces.types import Memory, Message
from memory_module.services.llm_service import LLMService
from memory_module.storage.sqlite_memory_storage import SQLiteMemoryStorage


class MessageDigest(BaseModel):
    topic: str = Field(..., description="The general category of the message(s).")
    summary: str = Field(..., description="A summary of the message(s).")
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that the message(s) is about. These can range from very specific to very general.",
    )


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
        default=None,
        description="One or more interesting fact extracted from the message. If the action is 'ignore',"
        "this field should be empty.",
    )


class EpisodicMemoryExtraction(BaseModel):
    action: Literal["add", "update", "ignore"] = Field(..., description="Action to take on the extracted fact")
    reason_for_action: Optional[str] = Field(
        ..., description="Reason for the action taken on the extracted fact or the reason it was ignored."
    )
    summary: Optional[str] = Field(
        ...,
        description="Summary of the extracted episodic memory. In case of update,"
        "include some details about the update including the latest state. Do not"
        "use real names (you can say 'The user' instead) and avoid pronouns. Be concise.",
    )


class MemoryCore(BaseMemoryCore):
    """Implementation of the memory core component."""

    def __init__(
        self,
        config: MemoryModuleConfig,
        llm_service: LLMService,
        storage: Optional[SQLiteMemoryStorage] = None,
    ):
        """Initialize the memory core.

        Args:
            config: Memory module configuration
            llm_service: LLM service instance
            storage: Optional storage implementation for memory persistence
        """
        self.lm = llm_service
        self.storage = storage or SQLiteMemoryStorage(db_path=config.db_path)

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
                    # TODO: Figure out embedding vectors
                    await self.storage.store_memory(memory, embedding_vector=[])

    async def process_episodic_messages(self, messages: List[Message]) -> None:
        """Process multiple messages into episodic memories (specific events, experiences)."""
        # TODO: Implement episodic memory processing
        await self._extract_episodic_memory_from_messages(messages)

    async def retrieve(self, query: str) -> List[Memory]:
        """Retrieve memories based on a query.

        Steps:
        1. Convert query to embedding
        2. Find relevant memories
        3. Possibly rerank or filter results
        """
        # TODO: Implement memory retrieval logic
        pass

    async def _extract_information_from_messages(self, messages: List[Message]) -> MessageDigest:
        """Extract meaningful information from messages using LLM.

        Args:
            messages: The list of messages to extract meaningful information from.

        Returns:
            MemoryDigest containing the summary, importance, and key points from the list of messages.
        """
        system_message = f"""You are an expert memory extractor. Given a list of messages, your task is to extract
        meaningful information by providing the following:

Summary: A concise summary of the overall content or key theme of the messages.
Importance: A numeric score between 1 and 10 (1 = least important, 10 = most important) that reflects how significant
this information is.
Key Points: A list of key points or notable facts extracted from the messages.

Here's the list of messages you need to analyze:
{[message.content for message in messages]}
"""
        # TODO: Fix the above prompt so that the messages are displayed correctly.
        # Ex "User: I love pie!", "Assitant: I love pie!"

        messages = [{"role": "system", "content": system_message}]

        return await self.lm.completion(messages=messages, response_model=MessageDigest)

    async def _create_memory_embedding(self, content: str) -> List[float]:
        """Create embedding for memory content."""
        res: EmbeddingResponse = await self.lm.embedding(input=[content])
        return res.data[0]["embedding"]

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
        system_message = f"""You are a semantic memory management agent. Your goal is to extract meaningful,
long-term facts and preferences from user messages. Focus on recognizing general patterns and interests
that will remain relevant over time, even if the user is mentioning short-term plans or events.

Prioritize:
- General Interests and Preferences: When a user mentions specific events or actions, focus on the underlying
interests, hobbies, or preferences they reveal (e.g., if the user mentions attending a conference, focus on the topic of the conference,
not the date or location).
- Facts or Details about user: Extract facts that describe long-term information about the user, such as details about things they own.
- Long-Term Facts: Extract facts that describe long-term information about the user, such as their likes, dislikes, or ongoing activities.
- Ignore Short-Term Details: Avoid storing short-term specifics like dates or locations unless they reflect a recurring activity or long-term plan.

{memory_message}
Here is the latest message that was sent:
User: {message.content}
"""  # noqa: E501

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": "Please analyze this message and decide whether to extract facts or ignore it. If extracting"
                "facts, provide one or more semantic facts focusing on long-term, meaningful information.",  # noqa: E501
            },
        ]

        res = await self.lm.completion(messages=messages, response_model=SemanticMemoryExtraction)

        return res

    async def _extract_episodic_memory_from_messages(self, messages: List[Message]) -> EpisodicMemoryExtraction:
        """Extract episodic memory from a list of messages.

        Args:
            messages: The list of messages to extract memories from

        Returns:
            EpisodicMemoryExtraction containing relevant details
        """
        system_message = f"""You are an episodic memory management agent. Your goal is to extract detailed memories of
specific events or experiences from user messages. Focus on capturing key actions and important contextual details that
the user may want to recall later.

Prioritize:
•	Key Events and Experiences: Focus on significant events or interactions the user mentions (e.g., attending an event,
participating in an activity, or experiencing something noteworthy).
•	Specific Details: Include relevant time markers, locations, people involved, and specific actions or outcomes if
they seem central to the memory. However, avoid storing every minor detail unless it helps reconstruct the experience.
•	Ignore Generalized Information: Do not focus on general interests or preferences, unless they are crucial to
understanding the specific event.

Here are the incoming messages:
{[message.content for message in messages]}
"""
        # TODO: Fix the above prompt so that the messages are displayed correctly.
        # Ex "User: I love pie!", "Assitant: I love pie!"
        messages = [{"role": "system", "content": system_message}]

        return await self.lm.completion(messages=messages, response_model=EpisodicMemoryExtraction)
