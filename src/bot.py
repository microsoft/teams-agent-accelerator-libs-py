import datetime
import json
import os
import sys
import traceback
from typing import List, Literal

sys.path.append(os.path.join(os.path.dirname(__file__), "../packages"))

from botbuilder.core import CardFactory, MemoryStorage, TurnContext
from botbuilder.schema import Activity
from litellm import acompletion
from memory_module import (
    AssistantMessageInput,
    InternalMessageInput,
    LLMConfig,
    Memory,
    MemoryModule,
    MemoryModuleConfig,
    UserMessageInput,
)
from pydantic import BaseModel, Field
from teams import Application, ApplicationOptions, TeamsAdapter
from teams.ai.citations import AIEntity, Appearance, ClientCitation
from teams.state import TurnState

from config import Config

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

completions_llm_config = {
    "model": memory_llm_config["model"],
    "api_key": memory_llm_config["api_key"],
    "api_base": memory_llm_config["api_base"],
    "api_version": memory_llm_config["api_version"],
}

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


class TaskConfig(BaseModel):
    task_name: str
    required_fields: list[str]


tasks_by_config = {
    "troubleshoot_device_issue": TaskConfig(
        task_name="troubleshoot_device_issue", required_fields=["OS", "Device Type", "Year"]
    ),
    "troubleshoot_connectivity_issue": TaskConfig(
        task_name="troubleshoot_connectivity_issue", required_fields=["OS", "Device Type", "Router Location"]
    ),
    "troubleshoot_access_issue": TaskConfig(
        task_name="troubleshoot_access_issue", required_fields=["OS", "Device Type", "Year"]
    ),
}


class GetCandidateTasks(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    user_query: str = Field(description="A succinct description of the user's issue")
    candidate_task: Literal["troubleshoot_device_issue", "troubleshoot_connectivity_issue", "troubleshoot_access_issue"]


class GetMemorizedFields(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    queries_for_fields: list[str] = Field(
        description="A list of questions to see if any information exists about the fields. These must be questions."
    )


class FieldToMemorize(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    field_name: str
    field_value: str


class UserDetail(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    field_name: str
    field_value: str
    memory_ids: List[str] = Field(description="A list of memory IDs for the field")


class ExecuteTask(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    succint_summary_of_issue: str
    user_details: List[UserDetail] = Field(description="A key value pair of the user's details")


class ConfirmMemorizedFields(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    fields_to_confirm: List[UserDetail]


async def get_candidate_tasks(candidate_tasks: GetCandidateTasks) -> str:
    candidate_task = tasks_by_config[candidate_tasks.candidate_task]
    return candidate_task.model_dump_json()


async def get_memorized_fields(fields_to_retrieve: GetMemorizedFields) -> str:
    empty_obj = {}
    for query in fields_to_retrieve.queries_for_fields:
        result = await memory_module.retrieve_memories(query, None, None)
        print("Getting memorized queries", query)
        print(result)
        print("---")

        if result:
            empty_obj[query] = ", ".join([f"{r.id}. {r.content}" for r in result])
        else:
            empty_obj[query] = None
    return json.dumps(empty_obj)


confirmation_card = {
    "type": "AdaptiveCard",
    "body": [],
    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
    "version": "1.6",
}


def build_confirmation_card(fields_to_confirm: List[UserDetail]):
    confirmation_card_copy = confirmation_card.copy()
    # databinding is not supported in the card, so we need to manually replace the values
    facts = []
    for field in fields_to_confirm:
        facts.append({"title": field.field_name, "value": field.field_value})
    confirmation_card_copy["body"].append({"type": "FactSet", "facts": facts})
    return CardFactory.adaptive_card(confirmation_card_copy)


async def confirm_memorized_fields(fields_to_confirm: ConfirmMemorizedFields, context: TurnContext) -> str:
    print("Confirming memorized fields", fields_to_confirm)
    flattened_memory_ids = [
        memory_id for user_detail in fields_to_confirm.fields_to_confirm for memory_id in user_detail.memory_ids
    ]
    memories = await memory_module.get_memories(flattened_memory_ids)
    # group memories by field name
    user_details_with_memories: List[tuple[UserDetail, Memory | None]] = []
    for user_detail in fields_to_confirm.fields_to_confirm:
        memories_for_user_detail = [memory for memory in memories if memory.id in user_detail.memory_ids]
        # just take the first one into account for citation (for now)
        user_details_with_memories.append(
            (user_detail, memories_for_user_detail[0] if memories_for_user_detail else None)
        )

    cited_memories: List[Memory] = [memory for _, memory in user_details_with_memories if memory is not None]
    messages_for_cited_memories = await memory_module.get_messages([memory.id for memory in cited_memories])
    print("messages_for_cited_memories", messages_for_cited_memories)
    memory_strs = []
    citations: List[ClientCitation] = []
    for user_detail, associated_memory in user_details_with_memories:
        idx = len(citations) + 1
        if associated_memory:
            memory_strs.append(f"{user_detail.field_name}: {user_detail.field_value} [{idx}]")
            associated_message = (
                messages_for_cited_memories[associated_memory.id][0]
                if associated_memory.id in messages_for_cited_memories
                else None
            )
            citations.append(
                ClientCitation(
                    str(idx),
                    Appearance(
                        name=user_detail.field_name,
                        abstract=associated_memory.content,
                        url=associated_message.deep_link if associated_message else None,
                    ),
                )
            )
        else:
            memory_strs.append(f"{user_detail.field_name}: {user_detail.field_value}")
            continue

    memory_details_str = "<br>".join([memory_str for memory_str in memory_strs])
    ai_entity = AIEntity(
        additional_type=["AIGeneratedContent"],
        citation=citations,
    )
    activity_with_card_attachment = Activity(
        type="message",
        text=f"Can you confirm the following fields?<br>{memory_details_str}",
        entities=[ai_entity] if ai_entity else None,
    )
    await context.send_activity(activity_with_card_attachment)
    return json.dumps(fields_to_confirm.model_dump())


async def send_string_message(context: TurnContext, message: str) -> str | None:
    activity = Activity(
        type="message", text=message, entities=[AIEntity(additional_type=["AIGeneratedContent"], citation=[])]
    )
    res = await context.send_activity(activity)
    if res:
        return res.id


async def execute_task(task_name: ExecuteTask, context: TurnContext) -> str:
    system_prompt = f"""
You are an IT Support Assistant. You make up some common solutions to common issues. Be creative.

The user's issue is: {task_name.succint_summary_of_issue}

The user's details are: {task_name.user_details}

Come up with a solution to the user's issue.
"""
    res = await acompletion(
        **completions_llm_config,
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.9,
    )
    await send_string_message(context, res.choices[0].message.content)
    return res.choices[0].message.content


def get_available_functions():
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


def build_deep_link(context: TurnContext, message_id: str):
    conversation_ref = TurnContext.get_conversation_reference(context.activity)
    if conversation_ref.conversation and conversation_ref.conversation.is_group:
        deeplink_conversation_id = conversation_ref.conversation.id
    elif conversation_ref.user and conversation_ref.bot:
        user_aad_object_id = conversation_ref.user.aad_object_id
        bot_id = conversation_ref.bot.id.replace("28:", "")
        deeplink_conversation_id = f"19:{user_aad_object_id}_{bot_id}@unq.gbl.spaces"
    else:
        return None
    return f"https://teams.microsoft.com/l/message/{deeplink_conversation_id}/{message_id}?context=%7B%22contextType%22%3A%22chat%22%7D"


async def add_user_message(context: TurnContext):
    conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)
    content = context.activity.text
    if not content:
        print("content is not text, so ignoring...")
        return False
    if conversation_ref_dict is None:
        print("conversation_ref_dict is None")
        return False
    if conversation_ref_dict.user is None:
        print("conversation_ref_dict.user is None")
        return False
    if conversation_ref_dict.conversation is None:
        print("conversation_ref_dict.conversation is None")
        return False
    user_aad_object_id = conversation_ref_dict.user.aad_object_id
    message_id = context.activity.id
    await memory_module.add_message(
        UserMessageInput(
            id=message_id,
            content=context.activity.text,
            author_id=user_aad_object_id,
            conversation_ref=conversation_ref_dict.conversation.id,
            created_at=context.activity.timestamp if context.activity.timestamp else datetime.datetime.now(),
            deep_link=build_deep_link(context, context.activity.id),
        )
    )
    return True


async def add_agent_message(context: TurnContext, message_id: str, content: str):
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
    await memory_module.add_message(
        AssistantMessageInput(
            id=message_id,
            content=content,
            author_id=conversation_ref_dict.bot.id,
            conversation_ref=conversation_ref_dict.conversation.id,
            deep_link=build_deep_link(context, message_id),
        )
    )
    return True


async def add_internal_message(context: TurnContext, content: str):
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
    await memory_module.add_message(
        InternalMessageInput(
            content=content,
            author_id=conversation_ref_dict.bot.id,
            conversation_ref=conversation_ref_dict.conversation.id,
        )
    )
    return True


@bot_app.conversation_update("membersAdded")
async def on_members_added(context: TurnContext, state: TurnState):
    result = await send_string_message(context, "Hello! I'm your IT Support Assistant. How can I assist you today?")
    if result:
        await add_agent_message(context, result, "Hello! I'm your IT Support Assistant. How can I assist you today?")


@bot_app.activity("message")
async def on_message(context: TurnContext, state: TurnState):
    conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)
    await add_user_message(context)
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
    Step 2b (If necessary): Use the "confirm_memorized_fields" function to confirm the fields if they are already known.
    Step 2c (If necessary): For each missing field, prompt the user to provide the required information.

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
"""  # noqa E501
    messages = await memory_module.retrieve_chat_history(conversation_ref_dict.conversation.id, {"last_minutes": 1})
    print("messages", messages)
    llm_messages = [
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
            **completions_llm_config,
            messages=llm_messages,
            tools=get_available_functions(),
            tool_choice="auto",
            temperature=0,
        )

        message = response.choices[0].message

        if message.tool_calls is None and message.content is not None:
            agent_message_id = await send_string_message(context, message.content)
            if agent_message_id:
                await add_agent_message(context, agent_message_id, message.content)
            break
        elif message.tool_calls is None and message.content is None:
            print("No tool calls and no content")
            break

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments

            if function_name == "get_candidate_tasks":
                args = GetCandidateTasks.model_validate_json(function_args)
                res = await get_candidate_tasks(args)
            elif function_name == "get_memorized_fields":
                args = GetMemorizedFields.model_validate_json(function_args)
                res = await get_memorized_fields(args)
            elif function_name == "confirm_memorized_fields":
                args = ConfirmMemorizedFields.model_validate_json(function_args)
                res = await confirm_memorized_fields(args, context)
                should_break = True
            elif function_name == "execute_task":
                args = ExecuteTask.model_validate_json(function_args)
                res = await execute_task(args, context)
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
                await add_internal_message(
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
            break  # Break the outer loop

    return True


@bot_app.error
async def on_error(context: TurnContext, error: Exception):
    print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()
    await context.send_activity("The bot encountered an error or bug.")
