"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Bot Configuration"""

    PORT = 3978
    APP_ID = os.environ.get("BOT_ID", "")
    APP_PASSWORD = os.environ.get("BOT_PASSWORD", "")
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # OpenAI API key
    OPENAI_MODEL_NAME = "gpt-4o-mini"  # OpenAI model name. You can use any other model name from OpenAI.
    AZURE_OPENAI_API_KEY=os.environ["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_DEPLOYMENT=os.environ["AZURE_OPENAI_DEPLOYMENT"]
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    AZURE_OPENAI_API_BASE=os.environ["AZURE_OPENAI_API_BASE"]
    AZURE_OPENAI_API_VERSION=os.environ["AZURE_OPENAI_API_VERSION"]

