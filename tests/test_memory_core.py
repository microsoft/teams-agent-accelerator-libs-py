import os
import sys
from unittest import mock

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../packages"))

from memory_module.config import LLMConfig, MemoryModuleConfig
from memory_module.core.memory_core import EpisodicMemoryExtraction, MemoryCore
from memory_module.services.llm_service import LLMService

from .utils import create_test_message, get_env_llm_config


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

    # TODO: Mocking storage this way doesn't seem right but it works for now.
    storage = mock.Mock()
    config = MemoryModuleConfig(llm=lm_config)
    memory_core = MemoryCore(config=config, llm_service=lm, storage=storage)

    message = create_test_message(content="Hey, I'm a software developer.")
    res = await memory_core._extract_semantic_fact_from_message(message=message)

    assert any(includes(fact.text, "software developer") for fact in res.interesting_facts)


@pytest.mark.asyncio()
async def test_extract_information_from_messages(config):
    if not config.openai_api_key:
        pytest.skip("OpenAI API key not provided")

    lm_config = LLMConfig(model="gpt-4o-mini", api_key=config.openai_api_key)
    lm = LLMService(config=lm_config)

    # TODO: Mocking storage this way doesn't seem right but it works for now.
    storage = mock.Mock()
    config = MemoryModuleConfig(llm=lm_config)
    memory_core = MemoryCore(config=config, llm_service=lm, storage=storage)

    message = create_test_message(content="Hey, I'm a software developer.")
    res = await memory_core._extract_information_from_messages(messages=[message])

    assert any(includes(keyword, "software") for keyword in res.keywords)


@pytest.mark.asyncio()
async def test_extract_episodic_memory_from_messages(config):
    if not config.openai_api_key:
        pytest.skip("OpenAI API key not provided")

    lm_config = LLMConfig(model="gpt-4o-mini", api_key=config.openai_api_key)
    lm = LLMService(config=lm_config)

    # TODO: Mocking storage this way doesn't seem right but it works for now.
    storage = mock.Mock()
    config = MemoryModuleConfig(llm=lm_config)
    memory_core = MemoryCore(config=config, llm_service=lm, storage=storage)

    def m(c):
        return create_test_message(content=c)

    messages = [
        m("Hey, I'm a software developer."),
        m("That's cool! I'm a software developer too. What brings you to the Microsoft Build conference?"),
        m("I'm here to learn more about Azure and the latest software development tools."),
        m("That's awesome! I'm here to learn more about the latest software development tools too."),
    ]
    res: EpisodicMemoryExtraction = await memory_core._extract_episodic_memory_from_messages(messages=messages)

    assert includes(res.summary, "Azure")
    assert includes(res.summary, "software development tools")


@pytest.mark.asyncio()
async def test_create_memory_embedding_from_messages(config):
    if not config.openai_api_key:
        pytest.skip("OpenAI API key not provided")

    lm_config = LLMConfig(embedding_model="text-embedding-3-small", api_key=config.openai_api_key)
    lm = LLMService(config=lm_config)

    # TODO: Mocking storage this way doesn't seem right but it works for now.
    storage = mock.Mock()
    config = MemoryModuleConfig(llm=lm_config)
    memory_core = MemoryCore(config=config, llm_service=lm, storage=storage)

    content = "Which country has a maple leaf in its flag?"
    res: list[float] = await memory_core._create_memory_embedding(content=content)

    assert len(res) >= 512  # 512 is the smallest configurable embedding size for the text-embedding-3-small model
