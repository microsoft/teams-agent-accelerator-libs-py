import datetime
import logging
from typing import Dict, List, Literal, Optional

from litellm.types.utils import EmbeddingResponse
from pydantic import BaseModel, Field

from memory_module.config import MemoryModuleConfig
from memory_module.interfaces.base_memory_core import BaseMemoryCore
from memory_module.interfaces.base_memory_storage import BaseMemoryStorage
from memory_module.interfaces.types import (
    BaseMemoryInput,
    EmbedText,
    Memory,
    MemoryType,
    Message,
    MessageInput,
    ShortTermMemoryRetrievalConfig,
)
from memory_module.services.llm_service import LLMService
from memory_module.storage.in_memory_storage import InMemoryStorage
from memory_module.storage.sqlite_memory_storage import SQLiteMemoryStorage

logger = logging.getLogger(__name__)


class MessageDigest(BaseModel):
    topic: str = Field(..., description="The general category of the message(s).")
    summary: str = Field(..., description="A summary of the message(s).")
    keywords: list[str] = Field(
        default_factory=list,
        min_length=2,
        max_length=5,
        description="Keywords that the message(s) is about. These can range from very specific to very general.",
    )
    hypothetical_questions: list[str] = Field(
        default_factory=list,
        min_length=2,
        max_length=5,
        description="Hypothetical questions about this memory that someone might ask to query for it. "
        "These can range from very specific to very general.",
    )


class SemanticFact(BaseModel):
    text: str = Field(
        ...,
        description="The text of the fact. Do not use real names (you can say 'The user' instead) and avoid pronouns.",
    )
    message_indices: List[int] = Field(
        default_factory=list,
        description="The indices of the messages that the fact was extracted from.",
    )


class SemanticMemoryExtraction(BaseModel):
    action: Literal["add", "ignore"] = Field(..., description="Action to take on the extracted fact")
    reason_for_action: Optional[str] = Field(
        ...,
        description="Reason for the action taken on the extracted fact or the reason it was ignored.",
    )
    facts: Optional[List[SemanticFact]] = Field(
        default=None,
        description="One or more facts about the user. If the action is 'ignore'," "this field should be empty.",
    )


class ProcessSemanticMemoryDecision(BaseModel):
    decision: Literal["add", "ignore"] = Field(..., description="Action to take on the new memory")
    reason_for_decision: Optional[str] = Field(
        ...,
        description="Reason for the action.",
    )
    ids: Optional[list[str]] = Field(
        default_factory=list[str],
        description="When decision is 'ignore', list of old memory ids that duplicate with new memory.",
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
        storage: Optional[BaseMemoryStorage] = None,
    ):
        """Initialize the memory core.

        Args:
            config: Memory module configuration
            llm_service: LLM service instance
            storage: Optional storage implementation for memory persistence
        """
        self.lm = llm_service
        self.storage: BaseMemoryStorage = storage or (
            SQLiteMemoryStorage(db_path=config.db_path) if config.db_path is not None else InMemoryStorage()
        )

    async def process_semantic_messages(self, messages: List[Message]) -> None:
        """Process multiple messages into semantic memories (general facts, preferences)."""
        # make sure there is an author, and only one author
        author_id = next(
            (message.author_id for message in messages if message.author_id and message.type == "user"), None
        )
        if not author_id:
            logger.error("No author found in messages")
            return
        # check if there are any other authors
        other_authors = [
            message.author_id for message in messages if message.type == "user" and message.author_id != author_id
        ]
        if other_authors:
            logger.error("Multiple authors found in messages")
            return

        extraction = await self._extract_semantic_fact_from_messages(messages)

        if extraction.action == "add" and extraction.facts:
            for fact in extraction.facts:
                message_ids = [messages[idx].id for idx in fact.message_indices if idx < len(messages)]
                if await self._should_process_new_memory(fact.text, message_ids, author_id) is False:
                    continue
                metadata = await self._extract_metadata_from_fact(fact.text)
                memory = BaseMemoryInput(
                    content=fact.text,
                    created_at=messages[0].created_at or datetime.datetime.now(),
                    user_id=author_id,
                    message_attributions=message_ids,
                    memory_type=MemoryType.SEMANTIC,
                )
                embed_vectors = await self._get_semantic_fact_embeddings(fact.text, metadata)
                await self.storage.store_memory(memory, embedding_vectors=embed_vectors)

    async def process_episodic_messages(self, messages: List[Message]) -> None:
        """Process multiple messages into episodic memories (specific events, experiences)."""
        # TODO: Implement episodic memory processing
        await self._extract_episodic_memory_from_messages(messages)

    async def retrieve_memories(self, query: str, user_id: Optional[str], limit: Optional[int]) -> List[Memory]:
        """Retrieve memories based on a query.

        Steps:
        1. Convert query to embedding
        2. Find relevant memories
        3. Possibly rerank or filter results
        """
        embedText = EmbedText(text=query, embedding_vector=await self._get_query_embedding(query))
        return await self.storage.retrieve_memories(embedText, user_id, limit)

    async def update_memory(self, memory_id: str, updated_memory: str) -> None:
        metadata = await self._extract_metadata_from_fact(updated_memory)
        embed_vectors = await self._get_semantic_fact_embeddings(updated_memory, metadata)
        await self.storage.update_memory(memory_id, updated_memory, embedding_vectors=embed_vectors)

    async def remove_memories(self, user_id: str) -> None:
        await self.storage.clear_memories(user_id)

    async def _should_process_new_memory(
        self, new_memory_fact: str, message_ids: list[str], user_id: Optional[str]
    ) -> bool:
        similar_memories = await self.retrieve_memories(new_memory_fact, user_id, None)
        decision = await self._extract_memory_processing_decision(new_memory_fact, similar_memories, user_id)
        if decision.decision == "ignore":
            logger.info(f"Decision to ignore fact {new_memory_fact}")
            if decision.ids is not None:
                await self.storage.add_memory_attributions(decision.ids, message_ids)
            return False
        return True

    async def _extract_memory_processing_decision(
        self, new_memory: str, old_memories: List[Memory], user_id: Optional[str]
    ) -> ProcessSemanticMemoryDecision:
        """Determine whether to add, replace or drop this memory"""

        # created at time format: YYYY-MM-DD HH:MM:SS.sssss in UTC.
        old_memory_content = "\n".join(
            [
                f"<MEMORY created_at={str(memory.created_at)} id={str(memory.id)}>{memory.content}</MEMORY>"
                for memory in old_memories
            ]  # noqa: E501
        )
        system_message = f"""You are a semantic memory management agent. Your goal is to determine whether this new memory is duplicated with existing old memories.
Considerations:
- Time-based order: Each old memory has a creation time. Please take creation time into consideration.
- Repeated behavior: If the new memory indicates a repeated idea over a period of time, it should be added to reflect the pattern.
Return value:
- Add: add new memory while keep old memories
- Ignore: indicates that this memory is similar to an older memory and should be ignored
Here are the old memories:
{old_memory_content}
Here is the new memory:
<MEMORY created_at={str(datetime.datetime.now())}>{new_memory}</MEMORY>
"""  # noqa: E501
        messages = [{"role": "system", "content": system_message}]

        decision = await self.lm.completion(messages=messages, response_model=ProcessSemanticMemoryDecision)
        return decision

    async def _extract_metadata_from_fact(self, fact: str) -> MessageDigest:
        """Extract meaningful information from messages using LLM.

        Args:
            messages: The list of messages to extract meaningful information from.

        Returns:
            MemoryDigest containing the summary, importance, and key points from the list of messages.
        """
        return await self.lm.completion(
            messages=[
                {
                    "role": "system",
                    "content": "Your role is to rephrase the text in your own words and provide a summary of the text.",
                },
                {"role": "user", "content": fact},
            ],
            response_model=MessageDigest,
        )

    async def _get_query_embedding(self, query: str) -> List[float]:
        """Create embedding for memory content."""
        res: EmbeddingResponse = await self.lm.embedding(input=[query])
        return res.data[0]["embedding"]

    async def _get_semantic_fact_embeddings(
        self, fact: str, metadata: Optional[MessageDigest] = None
    ) -> List[List[float]]:
        """Create embedding for semantic fact and metadata."""
        embedding_input = [fact]
        if metadata is not None:
            embedding_input.extend(
                [metadata.topic, metadata.summary, *metadata.keywords, *metadata.hypothetical_questions]
            )
        res: EmbeddingResponse = await self.lm.embedding(input=embedding_input)
        return [data["embedding"] for data in res.data]

    async def _extract_semantic_fact_from_messages(
        self, messages: List[Message], memory_message: str = ""
    ) -> SemanticMemoryExtraction:
        """Extract semantic facts from messages using LLM.

        Args:
            message: The message to extract facts from
            memory_message: Optional context from previous memories

        Returns:
            SemanticMemoryExtraction containing the action and extracted facts
        """
        logger.info("Extracting semantic facts from messages")
        messages_str = ""
        for idx, message in enumerate(messages):
            if message.type == "user":
                messages_str += f"{idx}. User: {message.content}\n"
            elif message.type == "assistant":
                messages_str += f"{idx}. Assistant: {message.content}\n"
            else:
                # we explicitly ignore internal messages
                continue

        system_message = f"""You are a semantic memory management agent. Your goal is to extract meaningful, facts and preferences from user messages. Focus on recognizing general patterns and interests
that will remain relevant over time, even if the user is mentioning short-term plans or events.

Prioritize:
- General Interests and Preferences: When a user mentions specific events or actions, focus on the underlying
interests, hobbies, or preferences they reveal (e.g., if the user mentions attending a conference, focus on the topic of the conference,
not the date or location).
- Facts or Details about user: Extract facts that describe relevant information about the user, such as details about things they own.
- Facts about the user that the assistant might find useful.

{memory_message}
Here is the transcript of the conversation:
<TRANSCRIPT>
{messages_str}
</TRANSCRIPT>
"""  # noqa: E501

        llm_messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": "Please analyze this message and decide whether to extract facts or ignore it."
                "For each fact, include the indices of the messages that the fact was extracted from.",
            },
        ]

        res = await self.lm.completion(messages=llm_messages, response_model=SemanticMemoryExtraction)
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
        llm_messages = [{"role": "system", "content": system_message}]

        return await self.lm.completion(messages=llm_messages, response_model=EpisodicMemoryExtraction)

    async def add_short_term_memory(self, message: MessageInput) -> Message:
        return await self.storage.store_short_term_memory(message)

    async def retrieve_chat_history(
        self, conversation_ref: str, config: ShortTermMemoryRetrievalConfig
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        return await self.storage.retrieve_chat_history(conversation_ref, config)

    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
        return await self.storage.get_memories(memory_ids)

    async def get_user_memories(self, user_id: str) -> List[Memory]:
        return await self.storage.get_user_memories(user_id)

    async def get_messages(self, memory_ids: List[str]) -> Dict[str, List[Message]]:
        return await self.storage.get_messages(memory_ids)
