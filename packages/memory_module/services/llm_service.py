from typing import Any, Coroutine, List, Optional, Union

import litellm
from litellm import BaseModel, CustomStreamWrapper, EmbeddingResponse, ModelResponse
from pydantic import BaseModel

from memory_module.config import LLMConfig


class LLMService:
    """Service for handling LM operations.

    You can use any of the dozens of LM providers supported by LiteLLM.
    Simply follow their instructions for how to pass the `{provider_name}/{model_name}` and the authentication configurations to the constructor.

    For example, to use OpenAI's gpt-4o model with an API key, you would do:

    ```
    lm = LLMService(model="gpt-4o", api_key="the api key")
    ```

    To use an Azure OpenAI gpt-4o-mini deployment with an API key, you would do:

    ```
    lm = LLMService(
        model="azure/gpt-4o-mini", api_key="the api key", api_base="the api base", api_version="the api version"
    )
    ```

    For configuration examples of list of providers see: https://docs.litellm.ai/docs/providers
    """

    def __init__(self, config: LLMConfig):
        """Initialize LLM service with configuration.

        Args:
            config: LLM service configuration
        """
        self.model = config.model
        self.api_key = config.api_key
        self.api_base = config.api_base
        self.api_version = config.api_version
        self.embedding_model = config.embedding_model

        # Get any additional kwargs from the config
        self._kwargs = {
            k: v
            for k, v in config.model_dump().items()
            if k not in {"model", "api_key", "api_base", "api_version", "embedding_model"}
        }

    async def completion(
        self,
        messages: List,
        response_format: Optional[dict | BaseModel] = None,
        override_model: Optional[str] = None,
        **kwargs,
    ) -> Coroutine[Any, Any, ModelResponse | CustomStreamWrapper]:
        """Generate completion from the model. This method is a wrapper around litellm's `acompletion` method."""
        model = override_model or self.model
        if not model:
            raise ValueError("No LM model provided.")

        return await litellm.acompletion(
            messages=messages,
            model=model,
            api_key=self.api_key,
            api_version=self.api_version,
            api_base=self.api_base,
            response_format=response_format,
            **self._kwargs,
            **kwargs,
        )

    async def embedding(
        self, input: Union[str, List[str]], override_model: Optional[str] = None, **kwargs
    ) -> Coroutine[Any, Any, EmbeddingResponse]:
        """Get embeddings from the model. This method is a wrapper around litellm's `aembedding` method."""
        model = override_model or self.embedding_model
        if not model:
            raise ValueError("No embedding model provided.")

        return await litellm.aembedding(
            model=model,
            input=input,
            api_key=self.api_key,
            api_version=self.api_version,
            api_base=self.api_base,
            **self._kwargs,
            **kwargs,
        )
