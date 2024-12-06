import os
from datetime import datetime
from typing import Optional, TypedDict

from dotenv import load_dotenv

from memory_module.interfaces.types import Message


def create_test_message(content: str):
    return Message(
        id="123",
        author_id="123",
        conversation_ref="123",
        created_at=datetime.now(),
        content=content,
    )


# Create TypedDict dynamically based on LLMService's __init__ parameters
class LLMConfig(TypedDict):
    model: Optional[str] = (None,)
    api_key: Optional[str] = (None,)
    api_base: Optional[str] = (None,)
    api_version: Optional[str] = (None,)
    embedding_model: Optional[str] = (None,)


class EnvLLMConfig(TypedDict):
    openai_api_key: Optional[str] = (None,)
    openai_deployment: Optional[str] = (None,)
    azure_openai_api_key: Optional[str] = (None,)
    azure_openai_deployment: Optional[str] = (None,)
    azure_openai_embedding_deployment: Optional[str] = (None,)
    azure_openai_api_base: Optional[str] = (None,)
    azure_openai_api_version: Optional[str] = (None,)


def get_env_llm_config() -> EnvLLMConfig:
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY", None)
    openai_deployment = os.getenv("OPENAI_DEPLOYMENT", None)
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", None)
    azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", None)
    azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", None)
    azure_openai_api_base = os.getenv("AZURE_OPENAI_API_BASE", None)
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", None)

    return EnvLLMConfig(
        openai_api_key=openai_api_key,
        openai_deployment=openai_deployment,
        azure_openai_api_key=azure_openai_api_key,
        azure_openai_deployment=azure_openai_deployment,
        azure_openai_embedding_deployment=azure_openai_embedding_deployment,
        azure_openai_api_base=azure_openai_api_base,
        azure_openai_api_version=azure_openai_api_version,
    )


def build_llm_config(override_config: Optional[LLMConfig] = None) -> LLMConfig:
    env_config = get_env_llm_config()

    config = {
        "model": env_config["openai_deployment"] or env_config["azure_openai_deployment"],
        "api_key": env_config["openai_api_key"] or env_config["azure_openai_api_key"],
        "api_base": env_config["azure_openai_api_base"],
        "api_version": env_config["azure_openai_api_version"],
        "embedding_model": env_config["azure_openai_embedding_deployment"],
    }

    if override_config:
        config.update(override_config)

    return LLMConfig(**config)
