import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from memory_module.interfaces.types import Message
from memory_module.services.llm_service import LLMConfig


def create_test_message(content: str):
    return Message(
        id="123",
        author_id="123",
        conversation_ref="123",
        created_at=datetime.now(),
        content=content,
    )


class EnvLLMConfig:
    openai_api_key: Optional[str] = None
    openai_deployment: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    azure_openai_embedding_deployment: Optional[str] = None
    azure_openai_api_base: Optional[str] = None
    azure_openai_api_version: Optional[str] = None


def get_env_llm_config() -> dict:
    load_dotenv()

    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_deployment": os.getenv("OPENAI_DEPLOYMENT"),
        "azure_openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_openai_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "azure_openai_embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "azure_openai_api_base": os.getenv("AZURE_OPENAI_API_BASE"),
        "azure_openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    }


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
