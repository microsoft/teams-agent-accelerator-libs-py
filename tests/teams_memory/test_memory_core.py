"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from typing import Any, Dict, List
from unittest import mock

import pytest
from teams_memory.config import LLMConfig, MemoryModuleConfig
from teams_memory.core.memory_core import Answer, MemoryCore
from teams_memory.interfaces.types import TextEmbedding
from teams_memory.services.llm_service import LLMService

from tests.teams_memory.utils import (
    create_test_memory,
    create_test_user_message,
    get_env_llm_config,
)


def includes(text: str, phrase):
    return text.find(phrase) != -1


@pytest.fixture()
def config():
    return get_env_llm_config()


@pytest.mark.asyncio()
async def test_extract_memory_from_messages(config):
    if not config.openai_api_key:
        pytest.skip("OpenAI API key not provided")

    lm_config = LLMConfig(model="gpt-4o-mini", api_key=config.openai_api_key)
    lm = LLMService(config=lm_config)

    storage = mock.Mock()
    config = MemoryModuleConfig(llm=lm_config)
    memory_core = MemoryCore(config=config, llm_service=lm, storage=storage)

    message = create_test_user_message(content="Hey, I'm a software developer.")
    res = await memory_core._extract_semantic_fact_from_messages(messages=[message])

    assert res.facts is not None
    assert any(includes(fact.text, "software developer") for fact in res.facts)


@pytest.mark.asyncio()
async def test_extract_memory_from_messages_with_existing_memories_included(config):
    if not config.openai_api_key:
        pytest.skip("OpenAI API key not provided")

    lm_config = LLMConfig(model="gpt-4o-mini", api_key=config.openai_api_key)
    lm = LLMService(config=lm_config)

    storage = mock.Mock()
    config = MemoryModuleConfig(llm=lm_config)
    memory_core = MemoryCore(config=config, llm_service=lm, storage=storage)

    message = create_test_user_message(content="Hey, I'm a software developer.")
    existing_memory = create_test_memory(content="The user is a software developer.")
    res = await memory_core._extract_semantic_fact_from_messages(
        messages=[message], existing_memories=[existing_memory]
    )

    assert res.facts is None or len(res.facts) == 0
    assert res.action == "ignore"


@pytest.mark.asyncio()
async def test_extract_metadata_from_fact(config):
    if not config.openai_api_key:
        pytest.skip("OpenAI API key not provided")

    lm_config = LLMConfig(model="gpt-4o-mini", api_key=config.openai_api_key)
    lm = LLMService(config=lm_config)

    storage = mock.Mock()
    config = MemoryModuleConfig(llm=lm_config)
    memory_core = MemoryCore(config=config, llm_service=lm, storage=storage)

    res = await memory_core._extract_metadata_from_fact(
        fact="The user is a software developer."
    )

    assert any(includes(keyword, "software") for keyword in res.keywords)


@pytest.mark.asyncio()
async def test_get_query_embedding_from_messages(config):
    if not config.openai_api_key:
        pytest.skip("OpenAI API key not provided")

    lm_config = LLMConfig(
        embedding_model="text-embedding-3-small", api_key=config.openai_api_key
    )
    lm = LLMService(config=lm_config)

    storage = mock.Mock()
    config = MemoryModuleConfig(llm=lm_config)
    memory_core = MemoryCore(config=config, llm_service=lm, storage=storage)

    query = "Which country has a maple leaf in its flag?"
    res: TextEmbedding = await memory_core._get_query_embedding(query=query)

    assert (
        len(res.embedding_vector) >= 512
    )  # 512 is the smallest configurable embedding size for the text-embedding-3-small model


def test_answer_validation():
    """Test the Answer model validation for handling 'unknown' responses."""
    test_cases: List[Dict[str, Any]] = [
        {
            "input": {"answer": "unknown", "fact_ids": ["1", "2"]},
            "expected": {"answer": None, "fact_ids": None},
        },
        {
            "input": {"answer": "UNKNOWN", "fact_ids": ["1"]},
            "expected": {"answer": None, "fact_ids": None},
        },
        {
            "input": {"answer": "The user likes Python", "fact_ids": ["1"]},
            "expected": {"answer": "The user likes Python", "fact_ids": ["1"]},
        },
        {
            "input": {"answer": None, "fact_ids": None},
            "expected": {"answer": None, "fact_ids": None},
        },
    ]

    for case in test_cases:
        model = Answer(**case["input"])
        assert model.answer == case["expected"]["answer"]
        assert model.fact_ids == case["expected"]["fact_ids"]
