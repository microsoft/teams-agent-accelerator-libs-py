"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from typing import Any, Dict, List, Optional, TypeVar, Union, cast, overload

import instructor
import litellm
from litellm.types.utils import EmbeddingResponse, ModelResponse
from pydantic import BaseModel

from teams_memory.config import LLMConfig

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """Service for handling LM operations.

    You can use any of the dozens of LM providers supported by LiteLLM.
    Simply follow their instructions for how to pass the `{provider_name}/{model_name}` and the authentication
    configurations to the constructor.

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

        self.client = cast(
            instructor.AsyncInstructor, instructor.from_litellm(litellm.acompletion)
        )  # we need to cast this because acompletion is still a callable, even though it's awaitable

        # Get any additional kwargs from the config
        self._litellm_params = {
            k: v
            for k, v in config.model_dump().items()
            if k
            not in {"model", "api_key", "api_base", "api_version", "embedding_model"}
        }

    @overload
    async def completion(
        self,
        messages: List[Dict[str, str]],
        response_model: None = None,
        override_model: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse: ...

    @overload
    async def completion(
        self,
        messages: List[Dict[str, str]],
        response_model: type[T],
        override_model: Optional[str] = None,
        **kwargs: Any,
    ) -> T: ...

    async def completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[type[T]] = None,
        override_model: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[ModelResponse, T]:
        """Generate completion from the model."""
        model = override_model or self.model
        if not model:
            raise ValueError("No LM model provided.")

        # Start with base parameters
        params: dict[str, Any] = {"messages": messages, "model": model}

        # Add optional parameters only if they are not None
        if self.api_key is not None:
            params["api_key"] = self.api_key
        if self.api_base is not None:
            params["api_base"] = self.api_base
        if self.api_version is not None:
            params["api_version"] = self.api_version

        # Add litellm params and kwargs, which will override any previous values if there are conflicts
        params.update(self._litellm_params)
        params.update(kwargs)

        res = await self.client.chat.completions.create(
            response_model=response_model, **params
        )
        return res

    async def embedding(
        self,
        input: Union[str, List[str]],
        override_model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Get embeddings from the model. This method is a wrapper around litellm's `aembedding` method."""
        model = override_model or self.embedding_model
        if not model:
            raise ValueError("No embedding model provided.")

        result: EmbeddingResponse = await litellm.aembedding(
            model=model,
            input=input,
            api_key=self.api_key,
            api_version=self.api_version,
            api_base=self.api_base,
            **self._litellm_params,
            **kwargs,
        )
        return result
