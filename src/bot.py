import datetime
import json
import os
import sys
import traceback
from typing import List, Literal

sys.path.append(os.path.join(os.path.dirname(__file__), "../packages"))

from botbuilder.core import MemoryStorage, TurnContext
from memory_module import LLMConfig, MemoryModule, MemoryModuleConfig, Message
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field
from teams import Application, ApplicationOptions, TeamsAdapter
from teams.state import TurnState

from config import Config

config = Config()
# client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
client = AsyncAzureOpenAI(
    api_key=config.AZURE_OPENAI_API_KEY,
    api_version=config.AZURE_OPENAI_API_VERSION,
    azure_endpoint=config.AZURE_OPENAI_API_BASE,
    azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
)
memory_module = MemoryModule(
    config=MemoryModuleConfig(
        llm=LLMConfig(
            model="azure/gpt-4o",
            api_base=config.AZURE_OPENAI_API_BASE,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            embedding_model="azure/text-embedding-3-small",
        ),
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


class TaskConfig(BaseModel):
    task_name: str
    required_fields: list[str]


tasks_by_config = {
    "troubleshoot_device_issue": TaskConfig(
        task_name="troubleshoot_device_issue", required_fields=["OS", "Device Type", "Year"]
    ),
    "troubleshoot_connectivity_issue": TaskConfig(
        task_name="troubleshoot_connectivity_issue", required_fields=["OS", "Device Type", "Year"]
    ),
    "troubleshoot_access_issue": TaskConfig(
        task_name="troubleshoot_access_issue", required_fields=["OS", "Device Type", "Year"]
    ),
}


class GetCandidateTasks(BaseModel):
    user_query: str = Field(description="A succinct description of the user's issue")
    candidate_task: Literal["troubleshoot_device_issue", "troubleshoot_connectivity_issue", "troubleshoot_access_issue"]


class GetMemorizedFields(BaseModel):
    queries_for_fields: list[str] = Field(
        description="A list of questions to see if any information exists about the fields. These must be questions."
    )


class FieldToMemorize(BaseModel):
    field_name: str
    field_value: str


# class MemorizeFields(BaseModel):
#     fields_to_memorize: list[FieldToMemorize]


class PromptUser(BaseModel):
    query_for_user: str


class InformUser(BaseModel):
    message: str


class UserDetail(BaseModel):
    field_name: str
    field_value: str


class ExecuteTask(BaseModel):
    succint_summary_of_issue: str
    user_details: List[UserDetail] = Field(description="A key value pair of the user's details")


async def get_candidate_tasks(candidate_tasks: GetCandidateTasks) -> str:
    candidate_task = tasks_by_config[candidate_tasks.candidate_task]
    return candidate_task.model_dump_json()


async def get_memorized_fields(fields_to_retrieve: GetMemorizedFields) -> str:
    empty_obj = {}
    for field in fields_to_retrieve.queries_for_fields:
        result = await memory_module.retrieve_memories(field, None, None)
        print(f"result for {field}: {result}")
        if result:
            empty_obj[field] = ", ".join([r.content for r in result])
        else:
            empty_obj[field] = None
    return json.dumps(empty_obj)


# async def memorize_fields(fields_to_memorize: MemorizeFields) -> str:
#     return "Fields memorized"


async def prompt_user(query_for_user: PromptUser, context: TurnContext) -> str:
    await context.send_activity(query_for_user.query_for_user)
    return query_for_user.query_for_user


async def execute_task(task_name: ExecuteTask, context: TurnContext) -> str:
    system_prompt = f"""
You are an IT Support Assistant. You make up some common solutions to common issues. Be creative.

The user's issue is: {task_name.succint_summary_of_issue}

The user's details are: {task_name.user_details}

Come up with a solution to the user's issue.
"""
    res = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.9,
    )
    await context.send_activity(res.choices[0].message.content)
    return res.choices[0].message.content


async def inform_user(message: InformUser, context: TurnContext) -> str:
    await context.send_activity(message.message)
    return message.message


def get_available_functions():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_candidate_tasks",
                "description": "Identify the task based on user's query",
                "parameters": GetCandidateTasks.schema(),
            },
        },
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "inform_user",
        #         "description": "Send a message to the user",
        #         "parameters": InformUser.schema(),
        #     },
        # },
        {
            "type": "function",
            "function": {
                "name": "get_memorized_fields",
                "description": "Retrieve values for fields that have been previously memorized",
                "parameters": GetMemorizedFields.schema(),
            },
        },
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "prompt_user",
        #         "description": "Ask the user for specific information",
        #         "parameters": PromptUser.schema(),
        #     },
        # },
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "memorize_fields",
        #         "description": "Store field values for later use",
        #         "parameters": MemorizeFields.schema(),
        #     },
        # },
        {
            "type": "function",
            "function": {
                "name": "execute_task",
                "description": "Execute a troubleshooting task",
                "parameters": ExecuteTask.schema(),
            },
        },
    ]


async def add_message(
    context: TurnContext,
    content: str,
    is_assistant_message: bool,
    created_at: datetime.datetime | None = None,
    override_message_id: str | None = None,
):
    conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)
    if conversation_ref_dict is None:
        print("conversation_ref_dict is None")
        return False
    if conversation_ref_dict.user is None:
        print("conversation_ref_dict.user is None")
        return False
    if conversation_ref_dict.bot is None:
        print("conversation_ref_dict.bot is None")
        return False
    if conversation_ref_dict.conversation is None:
        print("conversation_ref_dict.conversation is None")
        return False
    user_aad_object_id = conversation_ref_dict.user.aad_object_id
    bot_id = conversation_ref_dict.bot.id
    print(
        "Adding message",
        Message(
            id=override_message_id or context.activity.id,
            content=content,
            author_id=user_aad_object_id,
            conversation_ref=conversation_ref_dict.conversation.id,
            created_at=created_at or datetime.datetime.now(datetime.timezone.utc),
            is_assistant_message=is_assistant_message,
        ),
    )
    await memory_module.add_message(
        Message(
            id=override_message_id or context.activity.id,
            content=content,
            author_id=user_aad_object_id,
            conversation_ref=conversation_ref_dict.conversation.id,
            created_at=created_at or datetime.datetime.now(datetime.timezone.utc),
            is_assistant_message=is_assistant_message,
        )
    )


@bot_app.conversation_update("membersAdded")
async def on_members_added(context: TurnContext, state: TurnState):
    await context.send_activity("Hello! I'm your IT Support Assistant. How can I assist you today?")
    await add_message(context, "Hello! I'm your IT Support Assistant. How can I assist you today?", True)


@bot_app.activity("message")
async def on_message(context: TurnContext, state: TurnState):
    conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)
    await add_message(context, context.activity.text, False)
    system_prompt = """
You are an IT Chat Bot that helps users troubleshoot tasks

<PROGRAM>
Ask the user for their request unless the user has already provided it.

Note: Step 1 - Identify potential tasks based on the user's query.
To identify tasks:
    Step 1a: Use the "get_candidate_tasks" function with the user's query as input.
    Step 1b (If necessary): Display "I'm not sure what task you need help with. Could you clarify your request?"

Note: Step 2 - Gather necessary information for the selected task.
To gather missing fields for the task:
    Step 2a: Use the "get_memorized_fields" function to check if any required fields are already known.
    Step 2b (If necessary): For each missing field, prompt the user to provide the required information.

Note: Step 3 - Execute the task.
To execute the selected task:
    Step 3a: Use the "execute_task" function with the user's query, the selected task, and the list of gathered fields.
    Step 3b: Display the result of the task to the user.

Note: Full process flow.
While the user has requests:
    1. Identify tasks based on the user's query.
    2. Gather any required information for the task.
    3. Execute the task and display the result.
    4. Ask the user if they need help with anything else.

If the user ends the conversation, display "Thank you! Let me know if you need anything else in the future."

<INSTRUCTIONS>
Run the provided PROGRAM by executing each step.
"""
    messages = await memory_module.retrieve_short_term_memories(
        conversation_ref_dict.conversation.id, {"last_minutes": 1}
    )
    print("messages", messages)
    llm_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *[
            {
                "role": "assistant" if message.is_assistant_message else "user",
                "content": message.content,
            }
            for message in messages
        ],
    ]

    max_turns = 5
    should_break = False  # Flag to indicate if we should break the outer loop
    for _ in range(max_turns):
        response: ChatCompletion = await client.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=llm_messages,
            tools=get_available_functions(),
            tool_choice="auto",
            temperature=0,
        )
        print("llm_messages", llm_messages)
        print("tools", get_available_functions())
        print("response", response)

        message = response.choices[0].message

        if message.tool_calls is None and message.content is not None:
            print("Adding dm", response.id)
            await add_message(context, message.content, True, datetime.datetime.now(datetime.timezone.utc), response.id)
            await context.send_activity(message.content)
            break
        elif message.tool_calls is None and message.content is None:
            print("No tool calls and no content")
            break

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments

            print(f"--ToolCall {tool_call.id}--")
            print(function_name)
            print(function_args)
            print(f"--ToolCall {tool_call.id}--")

            if function_name == "get_candidate_tasks":
                args = GetCandidateTasks.model_validate_json(function_args)
                res = await get_candidate_tasks(args)
            elif function_name == "get_memorized_fields":
                args = GetMemorizedFields.model_validate_json(function_args)
                res = await get_memorized_fields(args)
            # elif function_name == "memorize_fields":
            #     args = MemorizeFields.model_validate_json(function_args)
            #     res = await memorize_fields(args)
            elif function_name == "prompt_user":
                args = PromptUser.model_validate_json(function_args)
                res = await prompt_user(args, context)
            elif function_name == "execute_task":
                args = ExecuteTask.model_validate_json(function_args)
                res = await execute_task(args, context)
            elif function_name == "inform_user":
                args = InformUser.model_validate_json(function_args)
                res = await inform_user(args, context)
            else:
                res = None

            if res is not None:
                print("--res--")
                print(res)
                print("--res--")
                llm_messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call],
                    }
                )
                llm_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(res)})
                await add_message(
                    context,
                    json.dumps(
                        {
                            "tool_call_name": function_name,
                            "result": res,
                        }
                    ),
                    True,
                    datetime.datetime.now(datetime.timezone.utc),
                    tool_call.id,
                )
            else:
                break

        if should_break:
            break  # Break the outer loop

    return True


@bot_app.error
async def on_error(context: TurnContext, error: Exception):
    print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()
    await context.send_activity("The bot encountered an error or bug.")
