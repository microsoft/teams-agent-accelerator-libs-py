"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import datetime
import logging
import os
import sys
from asyncio import gather
from typing import Awaitable, Callable, List, Optional, cast

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from botbuilder.core import TurnContext
from botbuilder.core.middleware_set import Middleware
from botbuilder.core.teams import TeamsInfo
from botbuilder.schema import Activity, ConversationReference, ResourceResponse
from botframework.connector.models import ChannelAccount, ConversationAccount
from teams_memory.config import MemoryModuleConfig
from teams_memory.core.memory_module import MemoryModule, ScopedMemoryModule
from teams_memory.interfaces.base_memory_module import BaseMemoryModule
from teams_memory.interfaces.types import (
    AssistantMessageInput,
    UserMessageInput,
)

logger = logging.getLogger(__name__)


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


class MemoryMiddleware(Middleware):
    def __init__(
        self,
        *,
        config: Optional[MemoryModuleConfig] = None,
        memory_module: Optional[BaseMemoryModule] = None,
    ):
        if config and memory_module:
            logger.warning(
                "config and memory_module are both provided, using memory_module"
            )
        elif config:
            self.memory_module: BaseMemoryModule = MemoryModule(config=config)
        elif memory_module:
            self.memory_module = memory_module
        else:
            raise ValueError("Either config or memory_module must be provided")

    async def _add_user_message(self, context: TurnContext):
        conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)
        content = context.activity.text
        if not content:
            logger.error("content is not text, so ignoring...")
            return False
        if conversation_ref_dict is None:
            logger.error("conversation_ref_dict is None")
            return False
        if conversation_ref_dict.user is None:
            logger.error("conversation_ref_dict.user is None")
            return False
        if conversation_ref_dict.conversation is None:
            logger.error("conversation_ref_dict.conversation is None")
            return False
        user_aad_object_id = cast(
            ChannelAccount, conversation_ref_dict.user
        ).aad_object_id
        message_id = context.activity.id
        await self.memory_module.add_message(
            UserMessageInput(
                id=message_id,
                content=context.activity.text,
                author_id=user_aad_object_id,
                conversation_ref=conversation_ref_dict.conversation.id,
                created_at=(
                    context.activity.timestamp
                    if context.activity.timestamp
                    else datetime.datetime.now()
                ),
                deep_link=build_deep_link(context, context.activity.id),
            )
        )
        return True

    async def _add_agent_message(
        self,
        context: TurnContext,
        activities: List[Activity],
        responses: List[ResourceResponse],
    ):
        conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)
        if conversation_ref_dict is None:
            logger.error("conversation_ref_dict is None")
            return False
        if conversation_ref_dict.bot is None:
            logger.error("conversation_ref_dict.bot is None")
            return False
        if conversation_ref_dict.conversation is None:
            logger.error("conversation_ref_dict.conversation is None")
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
                            created_at=(
                                activity.timestamp
                                if activity.timestamp
                                else datetime.datetime.now()
                            ),
                        )
                    )
                )

        if tasks:
            await gather(*tasks)
        return True

    async def _augment_context(self, context: TurnContext):
        conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)
        users_in_conversation_scope = await self._get_roster(
            conversation_ref_dict, context
        )
        context.set(
            "memory_module",
            ScopedMemoryModule(
                self.memory_module,
                users_in_conversation_scope,
                conversation_ref_dict.conversation.id,
            ),
        )

    async def on_turn(self, context: TurnContext, logic: Callable[[], Awaitable]):
        await self._augment_context(context)
        # Handle incoming message
        await self._add_user_message(context)

        # Store the original send_activities method
        original_send_activities = context.send_activities

        # Create a wrapped version that captures the activities
        # We need to do this because bot-framework has a bug with how
        # _on_send_activities middleware is implemented
        # https://github.com/microsoft/botbuilder-python/issues/2197
        async def wrapped_send_activities(activities: List[Activity]):
            responses = await original_send_activities(activities)
            await self._add_agent_message(context, activities, responses)
            return responses

        # Replace the send_activities method
        context.send_activities = wrapped_send_activities

        # Run the bot's logic
        await logic()

    async def _get_roster(
        self, conversation_ref: ConversationReference, context: TurnContext
    ) -> List[str]:
        if conversation_ref.conversation is None:
            logger.error("conversation_ref.conversation is None")
            return []

        conversation_type = cast(
            ConversationAccount, conversation_ref.conversation
        ).conversation_type
        if conversation_type == "personal":
            user = cast(ChannelAccount, conversation_ref.user)
            user_id = user.aad_object_id
            return [user_id]
        elif conversation_type == "groupChat":
            roster = await TeamsInfo.get_members(context)
            return [member.aad_object_id for member in roster]
        else:
            logger.warning("Conversation type %s not supported", conversation_type)
            return []
