"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import datetime
import logging
from typing import Any, List, Literal, Optional, Set, Tuple

from litellm.types.utils import EmbeddingResponse
from pydantic import BaseModel, Field, create_model, field_validator, model_validator

from teams_memory.config import (
    AzureAISearchStorageConfig,
    InMemoryStorageConfig,
    MemoryModuleConfig,
    SQLiteStorageConfig,
)
from teams_memory.core.prompts import (
    ANSWER_QUESTION_PROMPT,
    ANSWER_QUESTION_USER_PROMPT,
    MEMORY_PROCESSING_DECISION_PROMPT,
    METADATA_EXTRACTION_PROMPT,
    SEMANTIC_FACT_EXTRACTION_PROMPT,
    SEMANTIC_FACT_USER_PROMPT,
)
from teams_memory.interfaces.base_memory_core import BaseMemoryCore
from teams_memory.interfaces.base_memory_storage import BaseMemoryStorage
from teams_memory.interfaces.base_message_storage import BaseMessageStorage
from teams_memory.interfaces.errors import MemoryNotFoundError
from teams_memory.interfaces.types import (
    BaseMemoryInput,
    Memory,
    MemoryType,
    Message,
    MessageInput,
    TextEmbedding,
    Topic,
)
from teams_memory.services.llm_service import LLMService
from teams_memory.storage.in_memory_storage import InMemoryStorage
from teams_memory.storage.sqlite_memory_storage import SQLiteMemoryStorage
from teams_memory.storage.sqlite_message_storage import SQLiteMessageStorage

logger = logging.getLogger(__name__)


class MessageDigest(BaseModel):
    reworded_facts: list[str] = Field(
        default_factory=list,
        description="A list of reworded facts that are similar to the original fact. These should be in the same language as the original fact.",  # noqa: E501
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


class Answer(BaseModel):
    answer: Optional[str] = Field(..., description="The answer to the question")
    fact_ids: Optional[List[str]] = Field(
        ..., description="The fact ids that were used to answer the question"
    )

    @model_validator(mode="after")
    def validate_answer(self) -> "Answer":
        if self.answer and self.answer.lower() == "unknown":
            self.answer = None
            self.fact_ids = None
        return self


class MemoryCore(BaseMemoryCore):
    """Implementation of the memory core component."""

    def __init__(
        self,
        config: MemoryModuleConfig,
        llm_service: LLMService,
        memory_storage: Optional[BaseMemoryStorage] = None,
        message_storage: Optional[BaseMessageStorage] = None,
    ):
        """Initialize the memory core.

        Args:
            config: Memory module configuration
            llm_service: LLM service instance
            storage: Optional storage implementation for memory persistence
        """
        self.lm = llm_service
        self.memory_storage: BaseMemoryStorage = (
            memory_storage or self._build_memory_storage(config)
        )
        self.message_storage: BaseMessageStorage = (
            message_storage or self._build_message_storage(config)
        )
        self.topics = config.topics

    def _build_memory_storage(self, config: MemoryModuleConfig) -> BaseMemoryStorage:
        storage_config = config.get_storage_config("memory")
        if isinstance(storage_config, InMemoryStorageConfig):
            return InMemoryStorage()
        if isinstance(storage_config, SQLiteStorageConfig):
            return SQLiteMemoryStorage(storage_config)
        if isinstance(storage_config, AzureAISearchStorageConfig):
            # Importing it conditionally because it's an optional dependency
            from teams_memory.storage.azure_ai_search import (
                AzureAISearchMemoryStorage,
            )

            return AzureAISearchMemoryStorage(storage_config)

        raise ValueError(f"Invalid storage type: {config}")

    def _build_message_storage(self, config: MemoryModuleConfig) -> BaseMessageStorage:
        storage_config = config.get_storage_config("message")
        if isinstance(storage_config, InMemoryStorageConfig):
            return InMemoryStorage()
        if isinstance(storage_config, SQLiteStorageConfig):
            return SQLiteMessageStorage(storage_config)

        raise ValueError(f"Invalid storage type: {storage_config}")

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
                logger.info("Storing memory: %s", memory.model_dump_json())
                await self.memory_storage.store_memory(
                    memory, embedding_vectors=embed_vectors
                )

    async def ask(
        self,
        *,
        user_id: Optional[str],
        question: str,
        query: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> Optional[Tuple[str, List[Memory]]]:
        memories = await self.search_memories(user_id=user_id, query=query, topic=topic)
        if not memories:
            return None
        return await self._answer_question_from_memories(memories, question)

    async def _answer_question_from_memories(
        self, memories: List[Memory], question: str
    ) -> Optional[Tuple[str, List[Memory]]]:
        sorted_memories = sorted(memories, key=lambda x: x.created_at, reverse=False)
        facts_str = "\n".join(
            [
                f"<FACT id={memory.id} created_at={memory.created_at}>{memory.content}</FACT>"
                for memory in sorted_memories
            ]
        )
        system_prompt = ANSWER_QUESTION_PROMPT.format(existing_facts=facts_str)
        user_prompt = ANSWER_QUESTION_USER_PROMPT.format(question=question)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        answer = await self.lm.completion(messages=messages, response_model=Answer)
        if answer.answer and answer.fact_ids:
            logger.debug("Question: %s, Answer: %s", question, answer)
            return answer.answer, [
                memory for memory in memories if memory.id in answer.fact_ids
            ]
        else:
            logger.debug("No answer found for question: %s", question)
            return None

    async def search_memories(
        self,
        *,
        user_id: Optional[str],
        query: Optional[str] = None,
        topic: Optional[str] = None,
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
        topics: Optional[List[str]],
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

        return await self.memory_storage.search_memories(
            user_id=user_id, text_embedding=text_embedding, topics=topics, limit=limit
        )

    async def update_memory(self, memory_id: str, updated_memory: str) -> None:
        # Verify that the memory exists
        memory = await self.get_memories(memory_ids=[memory_id])
        if not memory:
            raise MemoryNotFoundError(f"Memory with id {memory_id} not found")

        metadata = await self._extract_metadata_from_fact(updated_memory)
        embed_vectors = await self._get_semantic_fact_embeddings(
            updated_memory, metadata
        )
        await self.memory_storage.update_memory(
            memory_id, updated_memory, embedding_vectors=embed_vectors
        )

    async def remove_memories(
        self, *, user_id: Optional[str] = None, memory_ids: Optional[List[str]] = None
    ) -> None:
        if not memory_ids and not user_id:
            raise ValueError("Either memory_ids or user_id must be provided")
        await self.memory_storage.delete_memories(
            user_id=user_id, memory_ids=memory_ids
        )

    async def remove_messages(self, message_ids: List[str]) -> None:
        # Get list of memories that need to be updated/removed with removed messages
        remove_memories_list = await self.memory_storage.get_attributed_memories(
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
        await self.memory_storage.delete_memories(memory_ids=removed_memory_ids)
        await self.message_storage.delete_messages(message_ids)
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

        system_message = MEMORY_PROCESSING_DECISION_PROMPT.format(
            old_memory_content=old_memory_content,
            new_memory=new_memory,
            created_at=str(datetime.datetime.now()),
        )
        messages = [{"role": "system", "content": system_message}]

        decision: ProcessSemanticMemoryDecision = await self.lm.completion(
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
        topics_context = ""
        if topics:
            topics_str = "\n".join(
                [f"{topic.name}: {topic.description}" for topic in topics]
            )
            topics_str = f"This specific fact is related to the following topics:\n{topics_str}\nConsider these when extracting the metadata."  # noqa: E501
        else:
            topics_str = ""

        result: MessageDigest = await self.lm.completion(
            messages=[
                {
                    "role": "system",
                    "content": METADATA_EXTRACTION_PROMPT.format(
                        topics_context=topics_context
                    ),
                },
                {"role": "user", "content": fact},
            ],
            response_model=MessageDigest,
        )
        return result

    async def _get_query_embedding(self, query: str) -> TextEmbedding:
        """Create embedding for memory content."""
        res: EmbeddingResponse = await self.lm.embedding(input=[query])
        return TextEmbedding(text=query, embedding_vector=res.data[0]["embedding"])

    async def _get_semantic_fact_embeddings(
        self, fact: str, metadata: MessageDigest
    ) -> List[TextEmbedding]:
        """Create embedding for semantic fact and metadata."""
        embedding_input = [fact]  # fact is always included

        if metadata.reworded_facts:
            embedding_input.extend(metadata.reworded_facts)

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

        system_message = SEMANTIC_FACT_EXTRACTION_PROMPT.format(
            topics_str=topics_str, existing_memories_str=existing_memories_str
        )

        user_message = SEMANTIC_FACT_USER_PROMPT.format(messages_str=messages_str)

        llm_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        def topics_validator(cls: Any, v: List[str]) -> List[str]:
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

    async def add_message(self, message: MessageInput) -> Message:
        return await self.message_storage.upsert_message(message)

    async def retrieve_conversation_history(
        self,
        conversation_ref: str,
        *,
        n_messages: Optional[int] = None,
        last_minutes: Optional[float] = None,
        before: Optional[datetime.datetime] = None,
    ) -> List[Message]:
        """Retrieve short-term memories based on configuration (N messages or last_minutes)."""
        if not n_messages and not last_minutes and not before:
            raise ValueError(
                "At least one of n_messages, last_minutes, or before must be provided"
            )
        return await self.message_storage.retrieve_conversation_history(
            conversation_ref,
            n_messages=n_messages,
            last_minutes=last_minutes,
            before=before,
        )

    async def get_memories(
        self,
        *,
        memory_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> List[Memory]:
        """Get memories based on memory ids or user id."""
        if memory_ids is None and user_id is None:
            raise ValueError("Either memory_ids or user_id must be provided")
        return await self.memory_storage.get_memories(
            memory_ids=memory_ids, user_id=user_id
        )

    async def get_messages(self, memory_ids: List[str]) -> List[Message]:
        return await self.message_storage.get_messages(memory_ids)

    async def get_memories_from_message(self, message_id: str) -> List[Memory]:
        return await self.memory_storage.get_attributed_memories(
            message_ids=[message_id]
        )
