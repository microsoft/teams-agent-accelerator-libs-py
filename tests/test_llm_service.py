import os
import sys
from unittest import mock

import instructor
import litellm
import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory_module.config import LLMConfig
from memory_module.services.llm_service import LLMService

from .utils import EnvLLMConfig, get_env_llm_config

litellm.set_verbose = True

load_dotenv()


@pytest.fixture()
def config():
    env_config = get_env_llm_config()
    return env_config


async def _return_arguments(**kwargs):
    return kwargs


@pytest.fixture()
def mock_completion(monkeypatch):
    client = mock.Mock()
    router = mock.Mock()
    monkeypatch.setattr(instructor, "apatch", client)
    monkeypatch.setattr(litellm, "Router", router)

    return client, router


@pytest.fixture()
def mock_embedding(monkeypatch):
    monkeypatch.setattr(litellm, "aembedding", _return_arguments)


def includes(text: str, phrase):
    return text.find(phrase) != -1


@pytest.mark.asyncio
async def test_completion_calls_litellm_client(mock_completion):
    model = "test-model"
    api_base = "api base"
    api_version = "api version"
    api_key = "api key"
    messages = []
    litellm_params = {"test key": "test value"}
    local_args = {"local test key": "local test value"}

    config = LLMConfig(model=model, api_base=api_base, api_version=api_version, api_key=api_key, **litellm_params)
    lm = LLMService(config=config)

    await lm.completion(messages, **local_args)

    client_mock, router_mock = mock_completion
    res = client_mock.mock_calls[1].kwargs
    assert res["model"] == model
    assert res["messages"] == messages
    assert res["response_model"] is None
    assert res["local test key"] == local_args["local test key"]

    config = router_mock.mock_calls[0].kwargs["model_list"][0]
    assert config["model_name"] == model
    assert config["litellm_params"]["model"] == model
    assert config["litellm_params"]["api_key"] == api_key
    assert config["litellm_params"]["api_base"] == api_base
    assert config["litellm_params"]["api_version"] == api_version
    assert config["litellm_params"]["test key"] == litellm_params["test key"]


@pytest.mark.asyncio
async def test_embedding_calls_litellm_aembedding(mock_embedding):
    embedding_model = "test-model"
    api_base = "api base"
    api_version = "api version"
    api_key = "api key"
    input = "test input"
    args = {"test key": "test value"}
    local_args = {"local test key": "local test value"}

    config = LLMConfig(
        embedding_model=embedding_model, api_base=api_base, api_version=api_version, api_key=api_key, **args
    )
    lm = LLMService(config=config)

    res = await lm.embedding(input, **local_args)

    assert res["model"] == embedding_model
    assert res["input"] == input
    assert res["api_base"] == api_base
    assert res["api_version"] == api_version
    assert res["api_key"] == api_key
    assert res["test key"] == args["test key"]
    assert res["local test key"] == local_args["local test key"]


@pytest.mark.asyncio
async def test_completion_openai(config: EnvLLMConfig):
    if not config.openai_api_key:
        pytest.skip("OpenAI API key is missing")

    llm_config = LLMConfig(model="gpt-4o", api_key=config.openai_api_key)
    lm = LLMService(config=llm_config)
    messages = [{"role": "system", "content": "Which country has a maple leaf in its flag?"}]

    res = await lm.completion(messages)
    text = res.choices[0].message.content

    assert includes(text, "Canada")


@pytest.mark.asyncio
async def test_completion_openai_structured_outputs(config: EnvLLMConfig):
    if not config.openai_api_key:
        pytest.skip("OpenAI API key is missing")

    llm_config = LLMConfig(model="gpt-4o", api_key=config.openai_api_key)
    lm = LLMService(config=llm_config)
    messages = [{"role": "system", "content": "Which country has a maple leaf in its flag?"}]

    class Country(BaseModel):
        name: str
        capital: str

    res = await lm.completion(messages, response_model=Country)

    assert res.name == "Canada"


@pytest.mark.asyncio
async def test_embeddings_openai(config: EnvLLMConfig):
    if not config.openai_api_key:
        pytest.skip("OpenAI API key is missing")

    llm_config = LLMConfig(embedding_model="text-embedding-3-small", api_key=config.openai_api_key)
    lm = LLMService(config=llm_config)
    query = "Which country has a maple leaf in its flag?"

    res = await lm.embedding(input=[query])

    assert res.model == "text-embedding-3-small"
    assert (
        len(res.data[0]["embedding"]) >= 512
    )  # 512 is the smallest configurable embedding size for the text-embedding-3-small model


@pytest.mark.asyncio
async def test_completion_azure_openai(config: EnvLLMConfig):
    model = config.azure_openai_deployment
    api_base = config.azure_openai_api_base
    api_version = config.azure_openai_api_version
    api_key = config.azure_openai_api_key

    # TODO: Switch to microsft entra id auth when litellm fixes bug: https://github.com/BerriAI/litellm/pull/6917
    llm_config = LLMConfig(model=f"azure/{model}", api_key=api_key, api_base=api_base, api_version=api_version)
    lm = LLMService(config=llm_config)
    messages = [{"role": "system", "content": "Which country has a maple leaf in its flag?"}]

    res = await lm.completion(messages)
    text = res.choices[0].message.content

    assert includes(text, "Canada")


@pytest.mark.asyncio
async def test_completion_azure_openai_structured_outputs(config: EnvLLMConfig):
    model = config.azure_openai_deployment
    api_base = config.azure_openai_api_base
    api_version = config.azure_openai_api_version
    api_key = config.azure_openai_api_key

    llm_config = LLMConfig(model=f"azure/{model}", api_key=api_key, api_base=api_base, api_version=api_version)
    lm = LLMService(config=llm_config)
    messages = [{"role": "system", "content": "Which country has a maple leaf in its flag?"}]

    class Country(BaseModel):
        name: str
        capital: str

    res = await lm.completion(messages, response_model=Country)

    assert res.name == "Canada"


@pytest.mark.asyncio
async def test_embeddings_azure_openai(config: EnvLLMConfig):
    model = config.azure_openai_embedding_deployment
    api_base = config.azure_openai_api_base
    api_version = config.azure_openai_api_version
    api_key = config.azure_openai_api_key

    lm = LLMService(
        config=LLMConfig(embedding_model=f"azure/{model}", api_key=api_key, api_base=api_base, api_version=api_version)
    )
    query = "Which country has a maple leaf in its flag?"

    res = await lm.embedding(input=[query])

    assert res.model == "text-embedding-3-small"
    assert (
        len(res.data[0]["embedding"]) >= 512
    )  # 512 is the smallest configurable embedding size for the text-embedding-3-small model


@pytest.mark.asyncio
async def test_completion_no_model_provided():
    lm = LLMService(config=LLMConfig())
    try:
        await lm.completion(messages=[])
    except ValueError as e:
        assert includes(str(e), "No LM model provided.")


@pytest.mark.asyncio
async def test_embeddings_no_model_provided():
    lm = LLMService(config=LLMConfig())
    try:
        await lm.embedding(input=[])
    except ValueError as e:
        assert includes(str(e), "No embedding model provided.")


@pytest.mark.asyncio
async def test_embedding_model_override(mock_embedding):
    embedding_model = "test-model"
    override_model = "override model"

    llm_config = LLMConfig(embedding_model=embedding_model)
    lm = LLMService(config=llm_config)

    res = await lm.embedding(input="test input", override_model=override_model)

    assert res["model"] == override_model


@pytest.mark.asyncio
async def test_completion_model_override(mock_completion):
    model = "test-model"
    override_model = "override model"

    llm_config = LLMConfig(model=model)
    lm = LLMService(config=llm_config)

    await lm.completion(messages=[], override_model=override_model)

    client_mock, _ = mock_completion
    res = client_mock.mock_calls[1].kwargs
    assert res["model"] == override_model
