import json
from typing import List, Literal

from botbuilder.core import TurnContext
from botbuilder.schema import Activity
from pydantic import BaseModel, Field
from teams.ai.citations import AIEntity, Appearance, ClientCitation

from memory_module import BaseMemoryModule, Memory
from src.tech_assistant_agent.supported_tech_tasks import tasks_by_config


class GetCandidateTasks(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    user_query: str = Field(description="A succinct description of the user's issue")
    candidate_task: Literal["troubleshoot_device_issue", "troubleshoot_connectivity_issue", "troubleshoot_access_issue"]


class GetMemorizedFields(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    queries_for_fields: list[str] = Field(
        description="A list of questions to see if any information exists about the fields. These must be questions."
    )


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


async def get_memorized_fields(memory_module: BaseMemoryModule, fields_to_retrieve: GetMemorizedFields) -> str:
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


async def confirm_memorized_fields(
    memory_module: BaseMemoryModule, fields_to_confirm: ConfirmMemorizedFields, context: TurnContext
) -> str:
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
