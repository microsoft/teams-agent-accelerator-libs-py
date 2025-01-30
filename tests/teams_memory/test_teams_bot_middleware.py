"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from botbuilder.core import TurnContext
from botbuilder.schema import Activity, ResourceResponse
from botframework.connector.models import ChannelAccount, ConversationAccount
from teams_memory.config import LLMConfig, MemoryModuleConfig
from teams_memory.core.memory_module import MemoryModule, ScopedMemoryModule
from teams_memory.utils.teams_bot_middlware import MemoryMiddleware, build_deep_link

pytestmark = pytest.mark.asyncio


@pytest.fixture
def turn_context():
    context = MagicMock(spec=TurnContext)
    context.activity = Activity(
        id=str(uuid4()),
        text="Hello!",
        timestamp=datetime.datetime.now(),
        channel_id="msteams",
    )
    context.activity.conversation = ConversationAccount(
        id="conversation-id",
        name="Test Conversation",
        conversation_type="personal",
    )
    context.activity.from_property = ChannelAccount(
        id="user-id",
        name="Test User",
        aad_object_id="user-aad-id",
    )
    context.activity.recipient = ChannelAccount(
        id="28:bot-id",
        name="Test Bot",
    )
    return context


@pytest.fixture
def memory_module_mock():
    return AsyncMock()


@pytest.fixture
def middleware(memory_module_mock):
    return MemoryMiddleware(memory_module=memory_module_mock)


async def test_build_deep_link_personal_chat(turn_context):
    link = build_deep_link(turn_context, "message-id")
    assert "user-aad-id" in link
    assert "bot-id" in link
    assert "message-id" in link


async def test_build_deep_link_group_chat(turn_context):
    turn_context.activity.conversation.is_group = True
    turn_context.activity.conversation.id = "group-conversation-id"

    link = build_deep_link(turn_context, "message-id")
    assert "group-conversation-id" in link
    assert "message-id" in link


async def test_add_user_message(middleware, turn_context):
    result = await middleware._add_user_message(turn_context)
    assert result is True

    middleware.memory_module.add_message.assert_called_once()
    call_args = middleware.memory_module.add_message.call_args[0][0]
    assert call_args.content == "Hello!"
    assert call_args.author_id == "user-aad-id"
    assert call_args.conversation_ref == "conversation-id"


async def test_add_agent_message(middleware, turn_context):
    activities = [
        Activity(text="Bot response 1"),
        Activity(text="Bot response 2"),
    ]
    responses = [
        ResourceResponse(id="response-1"),
        ResourceResponse(id="response-2"),
    ]

    result = await middleware._add_agent_message(turn_context, activities, responses)
    assert result is True

    assert middleware.memory_module.add_message.call_count == 2
    for i, call in enumerate(middleware.memory_module.add_message.call_args_list):
        args = call[0][0]
        assert args.content == f"Bot response {i + 1}"
        assert args.author_id == "28:bot-id"
        assert args.conversation_ref == "conversation-id"


async def test_get_roster_personal_chat(middleware, turn_context):
    conversation_ref = TurnContext.get_conversation_reference(turn_context.activity)
    roster = await middleware._get_roster(conversation_ref, turn_context)

    assert len(roster) == 1
    assert roster[0] == "user-aad-id"


async def test_get_roster_group_chat(middleware, turn_context):
    conversation_ref = TurnContext.get_conversation_reference(turn_context.activity)
    conversation_ref.conversation.conversation_type = "groupChat"

    mock_members = [
        ChannelAccount(id="user1", aad_object_id="aad1"),
        ChannelAccount(id="user2", aad_object_id="aad2"),
    ]

    with patch(
        "botbuilder.core.teams.TeamsInfo.get_members", return_value=mock_members
    ):
        roster = await middleware._get_roster(conversation_ref, turn_context)

    assert len(roster) == 2
    assert roster == ["aad1", "aad2"]


async def test_on_turn(middleware, turn_context):
    logic = AsyncMock()

    await middleware.on_turn(turn_context, logic)

    # Verify context augmentation
    turn_context.set.assert_called_once()
    assert "memory_module" in turn_context.set.call_args[0]
    memory_module = turn_context.set.call_args[0][1]
    assert isinstance(memory_module, ScopedMemoryModule)

    # Verify the conversation id and user roster
    assert memory_module.conversation_ref == "conversation-id"
    assert memory_module.users_in_conversation_scope == ["user-aad-id"]

    # Verify user message was added
    middleware.memory_module.add_message.assert_called_once()

    # Verify bot logic was called
    logic.assert_called_once()


async def test_middleware_initialization_with_config():
    config = MemoryModuleConfig(
        db_path="test.db",
        buffer_size=5,
        timeout_seconds=60,
        llm=LLMConfig(),
    )

    middleware = MemoryMiddleware(config=config)
    assert middleware.memory_module is not None
    assert isinstance(middleware.memory_module, MemoryModule)


async def test_middleware_initialization_error():
    with pytest.raises(
        ValueError, match="Either config or memory_module must be provided"
    ):
        MemoryMiddleware()


async def test_add_user_message_invalid_content(middleware, turn_context):
    turn_context.activity.text = None
    result = await middleware._add_user_message(turn_context)
    assert result is False
    assert not middleware.memory_module.add_message.called
