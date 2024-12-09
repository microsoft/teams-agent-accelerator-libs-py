from typing import List, Optional, Union

import instructor
import litellm
from litellm import BaseModel
from litellm.router import Router
from litellm.types.utils import EmbeddingResponse


# TODO:
# * Implement retrying logic
# * Implement basic costs tracking/logging
# * Do we want to customeize the response models so that litellm types aren't being used around the codebase?
# * When using structured outputs do we want to parse the response into the pydantic model?
# * Think about using litellm's router instead of the litellm module directly
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

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        embedding_model: Optional[str] = None,
        **litellm_params,
    ):
        """Creates a new LLMService instance.

        Args:
            model (str): The model to use. This should be in the format of `{provider_name}/{model_name}`.
            api_key (Optional[str], optional): The api key. Defaults to None.
            api_base (Optional[str], optional): The api base endpoint. Defaults to None.
            api_version (Optional[str], optional): The api version. Defaults to None.
            embedding_model (Optional[str], optional): name of the embedding model to use. Defaults to None.
        """
        self.model = model
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.api_base = api_base
        self.api_version = api_version
        self._litellm_params = litellm_params

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
        self, input: Union[str, List[str]], override_model: Optional[str] = None, **kwargs: object
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
