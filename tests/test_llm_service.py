import os
import sys
import pytest
from dotenv import load_dotenv
import litellm
from pydantic import BaseModel
from utils import get_env_llm_config, EnvLLMConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from packages.memory_module.services.llm_service import LLMService
# from azure.identity import DefaultAzureCredential, get_bearer_token_provider

litellm.set_verbose = True

load_dotenv()


@pytest.fixture()
def config() -> EnvLLMConfig:
    return get_env_llm_config()


@pytest.fixture()
def azure_config(config: EnvLLMConfig):
    if not config["azure_openai_api_key"]:
        pytest.skip("Azure OpenAI API key is missing")

    if not config["azure_openai_api_base"]:
        pytest.skip("Azure OpenAI API base is missing")

    if not config["azure_openai_api_version"]:
        pytest.skip("Azure OpenAI API version is missing")

    if not config["azure_openai_deployment"]:
        pytest.skip("Azure OpenAI deployment is missing")

    if not config["azure_openai_embedding_deployment"]:
        pytest.skip("Azure OpenAI embedding deployment is missing")

    return config


async def _return_arguments(**kwargs):
    return kwargs


@pytest.fixture()
def mock_completion(monkeypatch):
    monkeypatch.setattr(litellm, "acompletion", _return_arguments)


@pytest.fixture()
def mock_embedding(monkeypatch):
    monkeypatch.setattr(litellm, "aembedding", _return_arguments)


def includes(text: str, phrase):
    return text.find(phrase) != -1


@pytest.mark.asyncio
async def test_completion_calls_litellm_acompletion(mock_completion):
    model = "test-model"
    api_base = "api base"
    api_version = "api version"
    api_key = "api key"
    messages = []
    args = {"test key": "test value"}
    local_args = {"local test key": "local test value"}

    lm = LLMService(model=model, api_base=api_base, api_version=api_version, api_key=api_key, **args)

    res = await lm.completion(messages, **local_args)

    assert res["model"] == model
    assert res["messages"] == messages
    assert res["api_base"] == api_base
    assert res["api_version"] == api_version
    assert res["api_key"] == api_key
    assert res["test key"] == args["test key"]
    assert res["local test key"] == local_args["local test key"]


@pytest.mark.asyncio
async def test_embedding_calls_litellm_aembedding(mock_embedding):
    embedding_model = "test-model"
    api_base = "api base"
    api_version = "api version"
    api_key = "api key"
    input = "test input"
    args = {"test key": "test value"}
    local_args = {"local test key": "local test value"}

    lm = LLMService(
        embedding_model=embedding_model, api_base=api_base, api_version=api_version, api_key=api_key, **args
    )

    res = await lm.embedding(input, **local_args)

    assert res["model"] == embedding_model
    assert res["input"] == input
    assert res["api_base"] == api_base
    assert res["api_version"] == api_version
    assert res["api_key"] == api_key
    assert res["test key"] == args["test key"]
    assert res["local test key"] == local_args["local test key"]


@pytest.mark.asyncio
async def test_completion_openai(config):
    if not config["openai_api_key"]:
        pytest.skip("OpenAI API key is missing")

    lm = LLMService(model="gpt-4o", api_key=config["openai_api_key"])
    messages = [{"role": "system", "content": "Which country has a maple leaf in its flag?"}]

    res = await lm.completion(messages)
    text = res.choices[0].message.content

    assert includes(text, "Canada")


@pytest.mark.asyncio
async def test_completion_openai_structured_outputs(config):
    if not config["openai_api_key"]:
        pytest.skip("OpenAI API key is missing")

    lm = LLMService(model="gpt-4o", api_key=config["openai_api_key"])
    messages = [{"role": "system", "content": "Which country has a maple leaf in its flag?"}]

    class Country(BaseModel):
        name: str
        capital: str

    res = await lm.completion(messages, response_model=Country)

    assert res.name == "Canada"


@pytest.mark.asyncio
async def test_embeddings_openai(config):
    if not config["openai_api_key"]:
        pytest.skip("OpenAI API key is missing")

    lm = LLMService(embedding_model="text-embedding-3-small", api_key=config["openai_api_key"])
    query = "Which country has a maple leaf in its flag?"

    res = await lm.embedding(input=[query])

    assert res.model == "text-embedding-3-small"
    assert (
        len(res.data[0]["embedding"]) >= 512
    )  # 512 is the smallest configurable embedding size for the text-embedding-3-small model


@pytest.mark.asyncio
async def test_completion_azure_openai(azure_config):
    model = azure_config["azure_openai_deployment"]
    api_base = azure_config["azure_openai_api_base"]
    api_version = azure_config["azure_openai_api_version"]
    api_key = azure_config["azure_openai_api_key"]

    # switch to microsft entra id auth when litellm fixes bug: https://github.com/BerriAI/litellm/pull/6917
    # token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

    lm = LLMService(model=f"azure/{model}", api_key=api_key, api_base=api_base, api_version=api_version)
    messages = [{"role": "system", "content": "Which country has a maple leaf in its flag?"}]

    res = await lm.completion(messages)
    text = res.choices[0].message.content

    assert includes(text, "Canada")


@pytest.mark.asyncio
async def test_completion_azure_openai_structured_outputs(azure_config):
    model = azure_config["azure_openai_deployment"]
    api_base = azure_config["azure_openai_api_base"]
    api_version = azure_config["azure_openai_api_version"]
    api_key = azure_config["azure_openai_api_key"]

    lm = LLMService(model=f"azure/{model}", api_key=api_key, api_base=api_base, api_version=api_version)
    messages = [{"role": "system", "content": "Which country has a maple leaf in its flag?"}]

    class Country(BaseModel):
        name: str
        capital: str

    res = await lm.completion(messages, response_model=Country)

    assert res.name == "Canada"


@pytest.mark.asyncio
async def test_embeddings_azure_openai(azure_config):
    model = azure_config["azure_openai_embedding_deployment"]
    api_base = azure_config["azure_openai_api_base"]
    api_version = azure_config["azure_openai_api_version"]
    api_key = azure_config["azure_openai_api_key"]

    lm = LLMService(embedding_model=f"azure/{model}", api_key=api_key, api_base=api_base, api_version=api_version)
    query = "Which country has a maple leaf in its flag?"

    res = await lm.embedding(input=[query])

    assert res.model == "text-embedding-3-small"
    assert (
        len(res.data[0]["embedding"]) >= 512
    )  # 512 is the smallest configurable embedding size for the text-embedding-3-small model


@pytest.mark.asyncio
async def test_completion_no_model_provided():
    lm = LLMService()
    try:
        await lm.completion(messages=[])
    except ValueError as e:
        assert includes(str(e), "No LM model provided.")


@pytest.mark.asyncio
async def test_embeddings_no_model_provided():
    lm = LLMService()
    try:
        await lm.embedding(input=[])
    except ValueError as e:
        assert includes(str(e), "No embedding model provided.")


@pytest.mark.asyncio
async def test_embedding_model_override(mock_embedding):
    embedding_model = "test-model"
    override_model = "override model"

    lm = LLMService(embedding_model=embedding_model)

    res = await lm.embedding(input="test input", override_model=override_model)

    assert res["model"] == override_model


@pytest.mark.asyncio
async def test_completion_model_override(mock_completion):
    model = "test-model"
    override_model = "override model"

    lm = LLMService(model=model)

    res = await lm.completion(messages=[], override_model=override_model)

    assert res["model"] == override_model
