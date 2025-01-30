"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import json
import os
import sys
from typing import List, Literal

from botbuilder.core import TurnContext
from botbuilder.schema import Activity
from pydantic import BaseModel, Field
from teams.ai.citations import AIEntity, Appearance, ClientCitation
from teams_memory import BaseScopedMemoryModule, Topic

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tech_assistant_agent.supported_tech_tasks import tasks_by_config

topics = [
    Topic(name="Device Type", description="The type of device the user has"),
    Topic(
        name="Operating System",
        description="The operating system for the user's device",
    ),
    Topic(name="Device year", description="The year of the user's device"),
]


class GetCandidateTasks(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    user_query: str = Field(description="A succinct description of the user's issue")
    candidate_task: Literal[
        "troubleshoot_device_issue",
        "troubleshoot_connectivity_issue",
        "troubleshoot_access_issue",
    ]


class GetMemorizedFields(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    memory_topics: List[Literal["Device Type", "Operating System", "Device year"]] = (
        Field(
            description="Topics for memories that the user may have revealed previously."
        )
    )


class UserDetail(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    field_name: str
    field_value: str
    memory_ids: List[str] = Field(description="A list of memory IDs for the field")


class ExecuteTask(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    succint_summary_of_issue: str
    user_details: List[UserDetail] = Field(
        description="A key value pair of the user's details"
    )


class ConfirmMemorizedFields(BaseModel):
    model_config = {"json_schema_extra": {"additionalProperties": False}}
    fields: List[UserDetail]


async def get_candidate_tasks(candidate_tasks: GetCandidateTasks) -> str:
    candidate_task = tasks_by_config[candidate_tasks.candidate_task]
    return candidate_task.model_dump_json()


async def get_memorized_fields(
    memory_module: BaseScopedMemoryModule, fields_to_retrieve: GetMemorizedFields
) -> str:
    fields: dict = {}
    for topic in fields_to_retrieve.memory_topics:
        result = await memory_module.search_memories(topic=topic)
        print("Getting memorized queries: ", topic)
        print(result)
        print("---")

        if result:
            fields[topic] = ", ".join([f"{r.id}. {r.content}" for r in result])
        else:
            fields[topic] = None
    return json.dumps(fields)


async def confirm_memorized_fields(
    memory_module: BaseScopedMemoryModule,
    fields_to_confirm: ConfirmMemorizedFields,
    context: TurnContext,
) -> str:
    print("Confirming memorized fields", fields_to_confirm)
    if not fields_to_confirm.fields:
        print("No fields to confirm")
        return "No fields to confirm"

    # Get memories and attributed messages
    cited_fields = []
    for user_detail in fields_to_confirm.fields:
        field_name = user_detail.field_name
        field_value = user_detail.field_value
        memories_with_citations = None
        if user_detail.memory_ids:
            memories_with_citations = await memory_module.get_memories_with_citations(
                memory_ids=user_detail.memory_ids
            )
        cited_fields.append((field_name, field_value, memories_with_citations))

    # Build client citations to send in Teams
    memory_strs = []
    citations: List[ClientCitation] = []
    for cited_field in cited_fields:
        idx = len(citations) + 1
        field_name, field_value, memories_with_citations = cited_field

        if memories_with_citations is None or len(memories_with_citations) == 0:
            memory_strs.append(f"{field_name}: {field_value}")
            continue
        else:
            memory_strs.append(f"{field_name}: {field_value} [{idx}]")

            memory = memories_with_citations[0].memory
            messages = memories_with_citations[0].messages  # type: ignore
            citations.append(
                ClientCitation(
                    str(idx),
                    Appearance(
                        name=field_name,
                        abstract=memory.content,
                        url=messages[0].deep_link if messages else None,
                    ),
                )
            )

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
