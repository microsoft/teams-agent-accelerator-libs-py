import os
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../packages"))

from pydantic import BaseModel
from memory_module.interfaces.types import AssistantMessage, Memory, UserMessage
from memory_module.services.llm_service import LLMConfig


def create_test_user_message(content: str, id: str = "123"):
    return UserMessage(
        id=id,
        author_id="123",
        conversation_ref="123",
        created_at=datetime.now(),
        content=content,
    )


def create_test_assistant_message(content: str):
    return AssistantMessage(
        id="123",
        author_id="123",
        conversation_ref="123",
        created_at=datetime.now(),
        content=content,
    )


def create_test_memory(content: str):
    return Memory(
        content=content,
        created_at=datetime.now(),
        memory_type="semantic",
        id="1",
    )


class EnvLLMConfig(BaseModel):
    openai_api_key: Optional[str] = None
    openai_deployment: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    azure_openai_embedding_deployment: Optional[str] = None
    azure_openai_api_base: Optional[str] = None
    azure_openai_api_version: Optional[str] = None


def get_env_llm_config() -> EnvLLMConfig:
    load_dotenv()

    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", None)
    azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", None)

    return EnvLLMConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", None),
        openai_deployment=os.getenv("OPENAI_DEPLOYMENT", None),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", None),
        azure_openai_deployment=f"azure/{azure_deployment}" if azure_deployment else None,
        azure_openai_embedding_deployment=f"azure/{azure_embedding_deployment}" if azure_embedding_deployment else None,
        azure_openai_api_base=os.getenv("AZURE_OPENAI_API_BASE", None),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", None),
    )


def build_llm_config(override_config: Optional[LLMConfig] = None) -> LLMConfig:
    env_config = get_env_llm_config()

    config = {
        "model": env_config.azure_openai_deployment or env_config.openai_deployment,
        "api_key": env_config.azure_openai_api_key or env_config.openai_api_key,
        "api_base": env_config.azure_openai_api_base,
        "api_version": env_config.azure_openai_api_version,
        "embedding_model": env_config.azure_openai_embedding_deployment,
    }

    if override_config:
        config.update(override_config)

    return LLMConfig(**config)
