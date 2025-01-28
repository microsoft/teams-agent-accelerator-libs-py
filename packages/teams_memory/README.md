# What is memory module?

Memory module is a simple yet powerful addition to help manage memories for Teams AI Agents. By offloading responsibility for keeping track of facts about users, it allows developers to create agents that are both more personable and more efficient.

# Features

- Seamless integration with Teams AI SDK.
  - The memory module hooks directly into the Teams AI SDK via a middleware and keeps track of both incoming and outgoing messages
- Automatic memory extraction
  - Give a set of topics (or use default ones) that your application cares more about, and the memory module will automatically begin extracting and storing those memories
- Simple Short Term memory retrieval
  - Simple paradigms for working memory retrievals (last N mins, or last M messages)
- Query Based or Topic Based memory retrieval
  - Search for existing memories using natural language queries or topics

# Integration

Integrating the Memory Module into your Teams AI SDK application (or Bot Framework) is fairly simple.

## Prerequisites

1. Azure Open AI or Open AI keys
   - The LLM Layer for the application is built using [LiteLLM](https://docs.litellm.ai/), so it can technically support any supported [providers](https://docs.litellm.ai/docs/providers), but we have only tested with AOAI and OAI.

## Integrating into a Teams AI SDK Application

### Add messages for extraction

1. After you build your bot `Application`, build a `MemoryMiddleware` which takes in some configs:
   - `llm` - these are configurations for the LLM. This is required.
   - `storage` - these are configurations for the storage layer. By default, it uses InMemoryStorage if no config is provided
   - `buffer_size` - This is the minimum size of the message buffer before memories are extracted using all the messages inside it.
   - `timeout_seconds` - This is the length of time that needs to elapse after the buffer starts filling up for a particular conversation before the extraction takes place.
     - Note: The system uses whichever condition occurs first: The `buffer_size` reaching its limit or the `timeout_seconds` elapsing before the extraction can take place.
   - `topics` - These are topics that the system can use to perform extraction. These are generally relevant to your application. Using specific topics here can help in the extraction process:
     - Focus - it helps the LLM focus on what's actually important and ignore less general facts that your application may never use
     - Storage - General topics cause the LLM to over-extract which can be unnecessary and may use up storage space unnecessarily

```
memory_middleware = MemoryMiddleware(
    config=MemoryModuleConfig(
        llm=LLMConfig(**memory_llm_config),
        storage=StorageConfig(
            db_path=os.path.join(os.path.dirname(__file__), "data", "memory.db")
        ), # if db_path is provided, we switch to using a sqlite db
        timeout_seconds=60, # extraction takes place 60 seconds after the first message comes in
        enable_logging=True, # helpful during debugging
        topics=[
		    Topic(name="Device Type", description="The type of device the user has"),
		    Topic(
		        name="Operating System",
		        description="The operating system for the user's device",
		    ),
		    Topic(name="Device year", description="The year of the user's device"),
		], # my application is a tech-assistant agent that cares to track a user's
    )
)
bot_app.adapter.use(memory_middleware)
```

At this point, the application is automatically listening to all incoming and outgoing messages. It groups messages by conversations and begins to schedule them for extraction. 2. The previous step only automatically stores incoming and outgoing messages from your bot. But sometimes, you may want to store `InternalMessage` as well. These can be used as additional context for extraction of memories, or for your agent to keep track of internal messages that may be required to keep track of the conversation. You can use it after a `tool_call` for example.

```
async def add_internal_message(self, context: TurnContext, tool_call_name: str, tool_call_result: str):
        conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)
        memory_module: BaseScopedMemoryModule = context.get("memory_module")
        await memory_module.add_message(
            InternalMessageInput(
                content=json.dumps({ "tool_call_name": tool_call_name, "result": tool_call_result }),
                author_id=conversation_ref_dict.bot.id,
                conversation_ref=memory_module.conversation_ref,
            )
        )
        return True
```

### Using Short Term Memories / Working Memory

Working Module allows you to store and use messages when using an LLM.

```
async def build_llm_messages(self, context: TurnContext, system_message: str):
        memory_module: BaseScopedMemoryModule = context.get("memory_module")
        assert memory_module
        messages = await memory_module.retrieve_chat_history(
            ShortTermMemoryRetrievalConfig(last_minutes=1)
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
            ], # UserMessages will have a `role` of `user`, `AssistantMessage` and `InternalMessage` objects will have a `role` of `assistant`
        ]
        return llm_messages
```

### Using Extracted Semantic Memory

When it's time to use memories in your application, you may get a `ScopedMemoryModule` from the `TurnContext`:

```
async def retrieve_device_type_memories(context: TurnContext):
	memory_module: ScopedMemoryModule = context.get('memory_module')
	device_type_memories = await memory_module.search_memories(
		topic=Topic(name="Device Type", description="The type of device the user has"),
		query="What device does the user own?"
	)
```

You may search for memories either using a topic or a natural language query (or both, but not none).

> [!IMPORTANT] > _`teams_memory` is in alpha, we are still internally validating and testing!_

## Logging

You can enable logging when setting up the memory module in the config.

```py
config = MemoryModuleConfig()
config.enable_logging=True,
```

### How does it work?

The `teams_memory` library uses
Python's [logging](https://docs.python.org/3.12/library/logging.html) library to facilitate logging. The `teams_memory` logger is configured to log debug messages (and higher serverity) to the console.

To set up the logger in your Python file, use the following code:

```py
import logging

logger = logging.getLogger(__name__)
```

This will create a logger named `teams_memory.<sub_module>.<file_name>`, which is a descendant of the `teams_memory` logger. All logged messages will be passed up to the handler assigned to the `teams_memory` logger.

### How to customize the logging behavior of the library?

Instead of setting `MemoryModuleConfig.enable_logging` to True, directly access the `teams_memory` logger like this:

```py
import logging

logger = logging.getLogger("teams_memory")
```

You can apply customizations to it. All loggers used in the library will be a descendant of it and so logs will be propagated to it.
