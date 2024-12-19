import json
from typing import List

from botbuilder.core import TurnContext
from litellm import acompletion
from litellm.types.utils import Choices, ModelResponse
from memory_module import BaseMemoryModule, InternalMessageInput, ShortTermMemoryRetrievalConfig

from src.tech_assistant_agent.agent import Agent, LLMConfig
from src.tech_assistant_agent.prompts import system_prompt
from src.tech_assistant_agent.tech_agent import TechSupportAgent
from src.tech_assistant_agent.tools import (
    ConfirmMemorizedFields,
    ExecuteTask,
    GetCandidateTasks,
    GetMemorizedFields,
    confirm_memorized_fields,
    get_candidate_tasks,
    get_memorized_fields,
)


class TechAssistantAgent(Agent):
    def __init__(self, llm_config: LLMConfig, memory_module: BaseMemoryModule) -> None:
        self._llm_config = llm_config
        self._memory_module = memory_module
        super().__init__()

    async def run(self, context: TurnContext):
        conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)  # noqa E501
        assert conversation_ref_dict.conversation
        messages = await self._memory_module.retrieve_chat_history(
            conversation_ref_dict.conversation.id, ShortTermMemoryRetrievalConfig(last_minutes=1)
        )
        llm_messages: List = [
            {
                "role": "system",
                "content": system_prompt,
            },
            *[
                {
                    "role": "user" if message.type == "user" else "assistant",
                    "content": message.content,
                }
                for message in messages
            ],
        ]

        max_turns = 5
        should_break = False  # Flag to indicate if we should break the outer loop
        for _ in range(max_turns):
            response = await acompletion(
                **self._llm_config,
                messages=llm_messages,
                tools=self._get_available_functions(),
                tool_choice="auto",
                temperature=0,
            )
            assert isinstance(response, ModelResponse)
            first_choice = response.choices[0]
            assert isinstance(first_choice, Choices)

            message = first_choice.message

            if message.tool_calls is None and message.content is not None:
                await self.send_string_message(context, message.content)
                break
            elif message.tool_calls is None and message.content is None:
                print("No tool calls and no content")
                break
            elif message.tool_calls is None:
                print("Tool calls but no content")
                break

            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments

                if function_name == "get_candidate_tasks":
                    args = GetCandidateTasks.model_validate_json(function_args)
                    res = await get_candidate_tasks(args)
                elif function_name == "get_memorized_fields":
                    args = GetMemorizedFields.model_validate_json(function_args)
                    res = await get_memorized_fields(self._memory_module, args)
                elif function_name == "confirm_memorized_fields":
                    args = ConfirmMemorizedFields.model_validate_json(function_args)
                    res = await confirm_memorized_fields(self._memory_module, args, context)
                    should_break = True
                elif function_name == "execute_task":
                    args = ExecuteTask.model_validate_json(function_args)
                    tech_support_agent = TechSupportAgent(self._llm_config, args)
                    res = await tech_support_agent.run(context)
                    should_break = True
                else:
                    res = None

                if res is not None:
                    llm_messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call],
                        }
                    )
                    llm_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(res)})
                    await self._add_internal_message(
                        context,
                        json.dumps(
                            {
                                "tool_call_name": function_name,
                                "result": res,
                            }
                        ),
                    )
                else:
                    break

                if should_break:
                    break

            if should_break:
                break  # Break the outer loop

    def _get_available_functions(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_candidate_tasks",
                    "description": "Identify the task based on user's query",
                    "parameters": GetCandidateTasks.schema(),
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_memorized_fields",
                    "description": "Retrieve values for fields that have been previously memorized",
                    "parameters": GetMemorizedFields.schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "confirm_memorized_fields",
                    "description": "Confirm the fields that have been previously memorized",
                    "parameters": ConfirmMemorizedFields.schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_task",
                    "description": "Execute a troubleshooting task",
                    "parameters": ExecuteTask.schema(),
                    "strict": True,
                },
            },
        ]

    async def _add_internal_message(self, context: TurnContext, content: str):
        conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)
        if not content:
            print("content is not text, so ignoring...")
            return False
        if conversation_ref_dict is None:
            print("conversation_ref_dict is None")
            return False
        if conversation_ref_dict.bot is None:
            print("conversation_ref_dict.bot is None")
            return False
        if conversation_ref_dict.conversation is None:
            print("conversation_ref_dict.conversation is None")
            return False
        await self._memory_module.add_message(
            InternalMessageInput(
                content=content,
                author_id=conversation_ref_dict.bot.id,
                conversation_ref=conversation_ref_dict.conversation.id,
            )
        )
        return True
