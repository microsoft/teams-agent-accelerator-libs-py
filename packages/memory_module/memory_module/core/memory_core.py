import datetime
import logging
from typing import Dict, List, Literal, Optional, Set

from litellm.types.utils import EmbeddingResponse
from pydantic import BaseModel, Field, create_model, field_validator

from memory_module.config import MemoryModuleConfig
from memory_module.interfaces.base_memory_core import BaseMemoryCore
from memory_module.interfaces.base_memory_storage import BaseMemoryStorage
from memory_module.interfaces.types import (
    BaseMemoryInput,
    Memory,
    MemoryType,
    Message,
    MessageInput,
    TextEmbedding,
    Topic,
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
    message_ids: Set[int] = Field(
        default_factory=set,
        description="The ids of the messages that the fact was extracted from.",
    )
    # TODO: Add a validator to ensure that topics are valid
    topics: Optional[List[str]] = Field(
        default=None,
        description="The name of the topic that the fact is most relevant to.",  # noqa: E501
    )


class SemanticMemoryExtraction(BaseModel):
    action: Literal["add", "ignore"] = Field(
        ..., description="Action to take on the extracted fact"
    )
    facts: Optional[List[SemanticFact]] = Field(
        default=None,
        description="One or more facts about the user. If the action is 'ignore', this field should be empty.",
    )


class ProcessSemanticMemoryDecision(BaseModel):
    decision: Literal["add", "ignore"] = Field(
        ..., description="Action to take on the new memory"
    )
    reason_for_decision: Optional[str] = Field(
        ...,
        description="Reason for the action.",
    )


class EpisodicMemoryExtraction(BaseModel):
    action: Literal["add", "update", "ignore"] = Field(
        ..., description="Action to take on the extracted fact"
    )
    reason_for_action: Optional[str] = Field(
        ...,
        description="Reason for the action taken on the extracted fact or the reason it was ignored.",
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
            SQLiteMemoryStorage(db_path=config.db_path)
            if config.db_path is not None
            else InMemoryStorage()
        )
        self.topics = config.topics

    async def process_semantic_messages(
        self,
        messages: List[Message],
        existing_memories: Optional[List[Memory]] = None,
    ) -> None:
        """Process multiple messages into semantic memories (general facts, preferences)."""
        # make sure there is an author, and only one author
        author_id = next(
            (
                message.author_id
                for message in messages
                if message.author_id and message.type == "user"
            ),
            None,
        )
        if not author_id:
            logger.error("No author found in messages")
            return

        # check if there are any other authors
        other_authors = [
            message.author_id
            for message in messages
            if message.type == "user" and message.author_id != author_id
        ]
        if other_authors:
            logger.error("Multiple authors found in messages")
            return

        extraction = await self._extract_semantic_fact_from_messages(
            messages, existing_memories
        )

        if extraction.action == "add" and extraction.facts:
            for fact in extraction.facts:
                decision = await self._get_add_memory_processing_decision(
                    fact, author_id
                )
                if decision.decision == "ignore":
                    logger.info("Decision to ignore fact: %s", fact.text)
                    continue
                topics = (
                    [topic for topic in self.topics if topic.name in fact.topics]
                    if fact.topics
                    else None
                )
                metadata = await self._extract_metadata_from_fact(fact.text, topics)
                message_ids = set(
                    messages[idx].id for idx in fact.message_ids if idx < len(messages)
                )
                memory = BaseMemoryInput(
                    content=fact.text,
                    created_at=messages[0].created_at or datetime.datetime.now(),
                    user_id=author_id,
                    message_attributions=message_ids,
                    memory_type=MemoryType.SEMANTIC,
                    topics=fact.topics,
                )
                embed_vectors = await self._get_semantic_fact_embeddings(
                    fact.text, metadata
                )
                logger.info("Storing memory: %s", memory)
                await self.storage.store_memory(memory, embedding_vectors=embed_vectors)

    async def process_episodic_messages(self, messages: List[Message]) -> None:
        """Process multiple messages into episodic memories (specific events, experiences)."""
        # TODO: Implement episodic memory processing
        await self._extract_episodic_memory_from_messages(messages)

    async def search_memories(
        self,
        *,
        user_id: Optional[str],
        query: Optional[str] = None,
        topic: Optional[Topic] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        return await self._retrieve_memories(
            user_id,
            query,
            [topic] if topic else None,
            limit,
        )

    async def _retrieve_memories(
        self,
        user_id: Optional[str],
        query: Optional[str],
        topics: Optional[List[Topic]],
        limit: Optional[int],
    ) -> List[Memory]:
        """Retrieve memories based on a query.

        Steps:
        1. Convert query to embedding
        2. Find relevant memories
        3. Possibly rerank or filter results
        """
        if query:
            text_embedding = await self._get_query_embedding(query)
        else:
            text_embedding = None

        return await self.storage.search_memories(
            user_id=user_id, text_embedding=text_embedding, topics=topics, limit=limit
        )

    async def update_memory(self, memory_id: str, updated_memory: str) -> None:
        metadata = await self._extract_metadata_from_fact(updated_memory)
        embed_vectors = await self._get_semantic_fact_embeddings(
            updated_memory, metadata
        )
        await self.storage.update_memory(
            memory_id, updated_memory, embedding_vectors=embed_vectors
        )

    async def remove_memories(self, user_id: str) -> None:
        await self.storage.clear_memories(user_id)

    async def remove_messages(self, message_ids: List[str]) -> None:
        # Get list of memories that need to be updated/removed with removed messages
        remove_memories_list = await self.storage.get_all_memories(
            message_ids=message_ids
        )

        # Loop each memory and determine whether to remove the memory
        removed_memory_ids = []
        for memory in remove_memories_list:
            if not memory.message_attributions:
                removed_memory_ids.append(memory.id)
                logger.info(
                    "memory %s will be removed since no associated messages", memory.id
                )
                continue
            # If all messages associated with a memory are removed, remove that memory too
            if all(item in message_ids for item in memory.message_attributions):
                removed_memory_ids.append(memory.id)
                logger.info(
                    "memory %s will be removed since all associated messages are removed",
                    memory.id,
                )

        # Remove selected messages and related old memories
        await self.storage.remove_memories(removed_memory_ids)
        await self.storage.remove_messages(message_ids)
        logger.info("messages %s are removed", ",".join(message_ids))

    async def _get_add_memory_processing_decision(
        self, new_memory_fact: SemanticFact, user_id: Optional[str]
    ) -> ProcessSemanticMemoryDecision:
        # topics = (
        #     [topic for topic in self.topics if topic.name in new_memory_fact.topics] if new_memory_fact.topics else None # noqa: E501
        # )
        similar_memories = await self._retrieve_memories(
            user_id, new_memory_fact.text, None, None
        )
        if len(similar_memories) > 0:
            decision = await self._extract_memory_processing_decision(
                new_memory_fact.text, similar_memories, user_id
            )
        else:
            decision = ProcessSemanticMemoryDecision(
                decision="add", reason_for_decision="No similar memories found"
            )
        logger.debug("Decision: %s", decision)
        return decision

    async def _extract_memory_processing_decision(
        self, new_memory: str, old_memories: List[Memory], user_id: Optional[str]
    ) -> ProcessSemanticMemoryDecision:
        """Determine whether to add, replace or drop this memory"""

        # created at time format: YYYY-MM-DD HH:MM:SS.sssss in UTC.
        old_memory_content = "\n".join(
            [
                f"<MEMORY created_at={str(memory.created_at)}>{memory.content}</MEMORY>"
                for memory in old_memories
            ]
        )
        system_message = f"""You are a semantic memory management agent. Your task is to decide whether the new memory should be added to the memory system or ignored as a duplicate.

Considerations:
1.	Context Overlap:
If the new memory conveys information that is substantially covered by an existing memory, it should be ignored.
If the new memory adds unique or specific information not present in any old memory, it should be added.
2.	Granularity of Detail:
Broader or more general memories should not replace specific ones. However, a specific detail can replace a general statement if it conveys the same underlying idea.
For example:
Old memory: “The user enjoys hiking in national parks.”
New memory: “The user enjoys hiking in Yellowstone National Park.”
Result: Ignore (The older memory already encompasses the specific case).
3.	Repeated Patterns:
If the new memory reinforces a pattern of behavior over time (e.g., multiple mentions of a recurring habit, preference, or routine), it should be added to reflect this trend.
4.	Temporal Relevance:
If the new memory reflects a significant change or update to the old memory, it should be added.
For example:
Old memory: “The user is planning a trip to Japan.”
New memory: “The user has canceled their trip to Japan.”
Result: Add (The new memory reflects a change).

Process:
	1.	Compare the specificity, unique details, and time relevance of the new memory against old memories.
	2.	Decide whether to add or ignore based on the considerations above.
	3.	Provide a clear and concise justification for your decision.

Here are the old memories:
{old_memory_content}

Here is the new memory:
{new_memory} created at {str(datetime.datetime.now())}
"""  # noqa: E501
        messages = [{"role": "system", "content": system_message}]

        decision = await self.lm.completion(
            messages=messages, response_model=ProcessSemanticMemoryDecision
        )
        logger.debug("Decision: %s", decision)
        return decision

    async def _extract_metadata_from_fact(
        self, fact: str, topics: Optional[List[Topic]] = None
    ) -> MessageDigest:
        """Extract meaningful information from the fact using LLM.

        Args:
            fact: The fact to extract meaningful information from.

        Returns:
            MemoryDigest containing the summary, importance, and key points from the fact.
        """
        if topics:
            topics_str = "\n".join(
                [f"{topic.name}: {topic.description}" for topic in topics]
            )
            topics_str = f"This specific fact is related to the following topics:\n{topics_str}\nConsider these when extracting the metadata."  # noqa: E501
        else:
            topics_str = ""

        return await self.lm.completion(
            messages=[
                {
                    "role": "system",
                    "content": f"""Your role is to rephrase the text in your own words and provide a summary of the text.\n{topics_str}""",  # noqa: E501
                },
                {"role": "user", "content": fact},
            ],
            response_model=MessageDigest,
        )

    async def _get_query_embedding(self, query: str) -> TextEmbedding:
        """Create embedding for memory content."""
        res: EmbeddingResponse = await self.lm.embedding(input=[query])
        return TextEmbedding(text=query, embedding_vector=res.data[0]["embedding"])

    async def _get_semantic_fact_embeddings(
        self, fact: str, metadata: MessageDigest
    ) -> List[TextEmbedding]:
        """Create embedding for semantic fact and metadata."""
        embedding_input = [fact]  # fact is always included

        if metadata.topic:
            embedding_input.append(metadata.topic)
        if metadata.summary:
            embedding_input.append(metadata.summary)
        embedding_input.extend(kw for kw in metadata.keywords if kw)
        embedding_input.extend(q for q in metadata.hypothetical_questions if q)

        res: EmbeddingResponse = await self.lm.embedding(input=embedding_input)

        return [
            TextEmbedding(text=text, embedding_vector=data["embedding"])
            for text, data in zip(embedding_input, res.data, strict=False)
        ]

    async def _extract_semantic_fact_from_messages(
        self, messages: List[Message], existing_memories: Optional[List[Memory]] = None
    ) -> SemanticMemoryExtraction:
        """Extract semantic facts from messages using LLM.

        Args:
            message: The message to extract facts from
            existing_memories: Optional context from previous memories

        Returns:
            SemanticMemoryExtraction containing the action and extracted facts
        """
        logger.info("Extracting semantic facts from messages")
        messages_str = ""
        for idx, message in enumerate(messages):
            if message.type == "user":
                messages_str += (
                    f"<USER_MESSAGE id={idx}>{message.content}</USER_MESSAGE>\n"
                )
            elif message.type == "assistant":
                messages_str += f"<ASSISTANT_MESSAGE id={idx}>{message.content}</ASSISTANT_MESSAGE>\n"
            else:
                # we explicitly ignore internal messages
                continue
        topics_str = "\n".join(
            [
                f"<MEMORY_TOPIC NAME={topic.name}>{topic.description}</MEMORY_TOPIC>"
                for topic in self.topics
            ]
        )

        existing_memories_str = ""
        if existing_memories:
            for memory in existing_memories:
                existing_memories_str = "\n".join(
                    [
                        f"<EXISTING MEMORY>{memory.content}</EXISTING MEMORY>"
                        for memory in existing_memories
                    ]
                )
        else:
            existing_memories_str = "NO EXISTING MEMORIES"

        system_message = f"""You are a semantic memory management agent. Your goal is to extract meaningful, facts and preferences from user messages. Focus on recognizing general patterns and interests that will remain relevant over time, even if the user is mentioning short-term plans or events.

<TOPICS>
{topics_str}
</TOPICS>

<EXISTING_MEMORIES>
{existing_memories_str}
</EXISTING_MEMORIES>
"""  # noqa: E501
        llm_messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"""
<TRANSCRIPT>
{messages_str}
</TRANSCRIPT>

<INSTRUCTIONS>
Extract new FACTS from the user messages in the TRANSCRIPT.
FACTS are patterns and interests that will remain relevant over time, even if the user is mentioning short-term plans or events.
Treat FACTS independently, even if multiple facts relate to the same topic.
Ignore FACTS found in EXISTING_MEMORIES.
""",  # noqa: E501
            },
        ]

        def topics_validator(cls, v):
            # Fix the casing if that's the only issue
            validated_topics = []
            for topic in v:
                config_topic = next(
                    (t for t in self.topics if t.name.lower() == topic.lower()), None
                )
                if config_topic:
                    validated_topics.append(config_topic.name)
                else:
                    raise ValueError(f"Topic {topic} not found in topics")
            return validated_topics

        ValidatedSemanticMemoryFact = create_model(
            "ValidatedSemanticMemoryFact",
            __base__=SemanticFact,
            __validators__={
                "validate_topics": field_validator("topics")(topics_validator)
            },
        )

        # Dynamically create validated model
        ValidatedSemanticMemoryExtraction = create_model(
            "ValidatedSemanticMemoryExtraction",
            __base__=SemanticMemoryExtraction,
            facts=(List[ValidatedSemanticMemoryFact], Field(description="List of extracted facts")),  # type: ignore[valid-type]
        )

        logger.debug("LLM messages: %s", llm_messages)
        res = await self.lm.completion(
            messages=llm_messages, response_model=ValidatedSemanticMemoryExtraction
        )
        logger.info("Extracted semantic memory: %s", res)
        return res

    async def _extract_episodic_memory_from_messages(
        self, messages: List[Message]
    ) -> EpisodicMemoryExtraction:
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

        return await self.lm.completion(
            messages=llm_messages, response_model=EpisodicMemoryExtraction
        )

    async def add_short_term_memory(self, message: MessageInput) -> Message:
        return await self.storage.store_short_term_memory(message)

    async def retrieve_conversation_history(
        self,
        conversation_ref: str,
        *,
        n_messages: Optional[int] = None,
        last_minutes: Optional[float] = None,
        before: Optional[datetime.datetime] = None,
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        return await self.storage.retrieve_conversation_history(
            conversation_ref,
            n_messages=n_messages,
            last_minutes=last_minutes,
            before=before,
        )

    async def get_memories(self, memory_ids: List[str]) -> List[Memory]:
        return await self.storage.get_memories(memory_ids)

    async def get_user_memories(self, user_id: str) -> List[Memory]:
        return await self.storage.get_user_memories(user_id)

    async def get_messages(self, memory_ids: List[str]) -> Dict[str, List[Message]]:
        return await self.storage.get_messages(memory_ids)

    async def get_memories_from_message(self, message_id):
        return await self.storage.get_all_memories(message_ids=[message_id])
