import os
import sys
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), "../packages"))

from botbuilder.core import MemoryStorage, TurnContext
from memory_module import (
    LLMConfig,
    MemoryMiddleware,
    MemoryModule,
    MemoryModuleConfig,
)
from teams import Application, ApplicationOptions, TeamsAdapter
from teams.state import TurnState

from config import Config
from src.tech_assistant_agent.agent import LLMConfig as AgentLLMConfig
from src.tech_assistant_agent.primary_agent import TechAssistantAgent

config = Config()

memory_llm_config = {
    "model": f"azure/{config.AZURE_OPENAI_DEPLOYMENT}" if config.AZURE_OPENAI_DEPLOYMENT else config.OPENAI_MODEL_NAME,
    "api_key": config.AZURE_OPENAI_API_KEY or config.OPENAI_API_KEY,
    "api_base": config.AZURE_OPENAI_API_BASE,
    "api_version": config.AZURE_OPENAI_API_VERSION,
    "embedding_model": f"azure/{config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}"
    if config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    else config.OPENAI_EMBEDDING_MODEL_NAME,
}

agent_llm_config = AgentLLMConfig(
    model=memory_llm_config["model"],
    api_key=memory_llm_config["api_key"],
    api_base=memory_llm_config["api_base"],
    api_version=memory_llm_config["api_version"],
)

memory_module = MemoryModule(
    config=MemoryModuleConfig(
        llm=LLMConfig(**memory_llm_config),
        db_path=os.path.join(os.path.dirname(__file__), "data", "memory.db"),
        timeout_seconds=60,
    )
)

# Define storage and application
storage = MemoryStorage()
bot_app = Application[TurnState](
    ApplicationOptions(
        bot_app_id=config.APP_ID,
        storage=storage,
        adapter=TeamsAdapter(config),
    )
)

bot_app.adapter.use(MemoryMiddleware(memory_module))


@bot_app.conversation_update("membersAdded")
async def on_members_added(context: TurnContext, state: TurnState):
    await context.send_activity("Hello! I am a tech assistant bot. How can I help you today?")
    return True


@bot_app.activity("message")
async def on_message(context: TurnContext, state: TurnState):
    tech_assistant_agent = TechAssistantAgent(agent_llm_config, memory_module)
    await tech_assistant_agent.run(context)
    return True


@bot_app.error
async def on_error(context: TurnContext, error: Exception):
    print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()
    await context.send_activity("The bot encountered an error or bug.")
