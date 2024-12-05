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
    OPENAI_MODEL_NAME = "gpt-3.5-turbo"  # OpenAI model name. You can use any other model name from OpenAI.