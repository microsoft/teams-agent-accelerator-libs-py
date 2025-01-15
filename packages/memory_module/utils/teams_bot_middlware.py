import datetime
from asyncio import gather
from typing import Awaitable, Callable, List

from botbuilder.core import TurnContext
from botbuilder.core.middleware_set import Middleware
from botbuilder.core.teams import TeamsInfo
from botbuilder.schema import Activity, ResourceResponse
from memory_module.core.memory_module import ScopedMemoryModule
from memory_module.interfaces.base_memory_module import BaseMemoryModule
from memory_module.interfaces.types import (
    AssistantMessageInput,
    UserMessageInput,
)


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


async def get_roster(conversation_ref: dict, context: TurnContext) -> List[str]:
    conversation_type = conversation_ref.get("conversation", {}).get("conversationType")

    if conversation_type == "personal":
        user = conversation_ref.get("user", {})
        user_id = user.get("id")
        if user_id:
            return [user_id]
        else:
            raise ValueError("User ID not found in conversation reference")
    elif conversation_type == "groupChat":
        roster = await TeamsInfo.get_members(context)
        return [member.id for member in roster]
    else:
        print(f"Unknown conversation type: {conversation_type}")
        return []


async def build_scoped_memory_module(memory_module: BaseMemoryModule, context: TurnContext):
    conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)
    users_in_conversation_scope = await get_roster(conversation_ref_dict, context)
    return ScopedMemoryModule(memory_module, users_in_conversation_scope, conversation_ref_dict.conversation.id)


class MemoryMiddleware(Middleware):
    def __init__(self, memory_module: BaseMemoryModule):
        self.memory_module = memory_module

    async def _add_user_message(self, context: TurnContext):
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
        await self.memory_module.add_message(
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

    async def _add_agent_message(
        self, context: TurnContext, activities: List[Activity], responses: List[ResourceResponse]
    ):
        conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)
        if conversation_ref_dict is None:
            print("conversation_ref_dict is None")
            return False
        if conversation_ref_dict.bot is None:
            print("conversation_ref_dict.bot is None")
            return False
        if conversation_ref_dict.conversation is None:
            print("conversation_ref_dict.conversation is None")
            return False

        tasks = []
        for activity, response in zip(activities, responses, strict=False):
            if activity.text:
                tasks.append(
                    self.memory_module.add_message(
                        AssistantMessageInput(
                            id=response.id,
                            content=activity.text,
                            author_id=conversation_ref_dict.bot.id,
                            conversation_ref=conversation_ref_dict.conversation.id,
                            deep_link=build_deep_link(context, response.id),
                            created_at=activity.timestamp if activity.timestamp else datetime.datetime.now(),
                        )
                    )
                )

        if tasks:
            await gather(*tasks)
        return True

    async def _augment_context(self, context: TurnContext):
        scoped_memory_module = await build_scoped_memory_module(self.memory_module, context)
        context.set("memory_module", scoped_memory_module)

    async def on_turn(self, context: TurnContext, logic: Callable[[], Awaitable]):
        # Handle incoming message
        await self.add_user_message(context)

        # Store the original send_activities method
        original_send_activities = context.send_activities

        # Create a wrapped version that captures the activities
        # We need to do this because bot-framework has a bug with how
        # _on_send_activities middleware is implemented
        # https://github.com/microsoft/botbuilder-python/issues/2197
        async def wrapped_send_activities(activities: List[Activity]):
            responses = await original_send_activities(activities)
            await self.add_agent_message(context, activities, responses)
            return responses

        # Replace the send_activities method
        context.send_activities = wrapped_send_activities

        # Run the bot's logic
        await logic()
