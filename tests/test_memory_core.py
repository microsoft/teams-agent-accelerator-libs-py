import os
import sys
from unittest import mock

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory_module.core.memory_core import MemoryCore
from memory_module.services.llm_service import LLMService

from .utils import create_test_message


def includes(text: str, phrase):
    return text.find(phrase) != -1


@pytest.fixture()
def config():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    azure_openai_api_base = os.getenv("AZURE_OPENAI_API_BASE")
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    return {
        "openai_api_key": openai_api_key,
        "azure_openai_api_key": azure_openai_api_key,
        "azure_openai_api_base": azure_openai_api_base,
        "azure_openai_api_version": azure_openai_api_version,
        "azure_openai_deployment": azure_openai_deployment,
        "azure_openai_embedding_deployment": azure_openai_embedding_deployment,
    }


@pytest.mark.asyncio()
async def test_extract_memory_from_messages(config):
    if not config["openai_api_key"]:
        pytest.skip("OpenAI API key not provided")

    lm = LLMService(model="gpt-4o-mini", api_key=config["openai_api_key"])

    # TODO: Mocking storage this way doesn't seem right but it works for now.
    storage = mock.Mock()
    memory_core = MemoryCore(llm_service=lm, storage=storage)

    message = create_test_message(content="Hey, I'm a software developer.")
    res = await memory_core._extract_semantic_fact_from_message(message=message)

    assert any(includes(fact.text, "software developer") for fact in res.interesting_facts)
