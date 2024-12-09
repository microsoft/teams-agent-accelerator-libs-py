from typing import List, Optional, Union

import instructor
import litellm
from litellm import BaseModel
from litellm.router import Router
from litellm.types.utils import EmbeddingResponse

from memory_module.config import LLMConfig


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

        # Get any additional kwargs from the config
        self._litellm_params = {
            k: v
            for k, v in config.model_dump().items()
            if k not in {"model", "api_key", "api_base", "api_version", "embedding_model"}
        }

    async def completion(
        self, messages: List, response_model: Optional[BaseModel] = None, override_model: Optional[str] = None, **kwargs
    ):
        """Generate completion from the model."""
        model = override_model or self.model
        if not model:
            raise ValueError("No LM model provided.")

        # TODO: This is hacky. Fix it later.
        client = instructor.apatch(
            Router(
                model_list=[
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": model,
                            "api_key": self.api_key,
                            "api_base": self.api_base,
                            "api_version": self.api_version,
                            **self._litellm_params,
                        },
                    }
                ]
            ) # type: ignore
        )

        return client.chat.completions.create(messages=messages, model=model, response_model=response_model, **kwargs) # type: ignore

    async def embedding(
        self, input: Union[str, List[str]], override_model: Optional[str] = None, **kwargs
    ) -> EmbeddingResponse:
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
            **self._litellm_params,
            **kwargs,
        )
