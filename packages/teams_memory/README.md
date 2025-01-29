> [!IMPORTANT] > _`teams_memory` is in alpha, we are still internally validating and testing!_

# What is Teams Memory Module?

Teams Memory module is a simple yet powerful addition to help manage memories for Teams AI Agents. By offloading responsibility for keeping track of facts about users, it allows developers to create agents that are both more personable and more efficient.

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

- Azure Open AI or Open AI keys
  - The LLM Layer for the application is built using [LiteLLM](https://docs.litellm.ai/), so it can technically support any supported [providers](https://docs.litellm.ai/docs/providers), but we have only tested with AOAI and OAI.

## Integrating into a Teams AI SDK Application

### Add messages

#### Incoming / Outgoing Messages

Memory extraction requires incoming and outgoing messages to your application. To simlify this, you can use a middlware to automatically do this for you.

After you build your bot `Application`, build a `MemoryMiddleware` which takes in some configs:

- `llm` - these are configurations for the LLM. This is required.
- `storage` - these are configurations for the storage layer. By default, it uses InMemoryStorage if no config is provided
- `buffer_size` - This is the minimum size of the message buffer before memories are extracted using all the messages inside it.
- `timeout_seconds` - This is the length of time that needs to elapse after the buffer starts filling up for a particular conversation before the extraction takes place.
  - Note: The system uses whichever condition occurs first: The `buffer_size` reaching its limit or the `timeout_seconds` elapsing before the extraction can take place.
- `topics` - These are topics that the system can use to perform extraction. These are generally relevant to your application. Using specific topics here can help in the extraction process:
  - Focus - it helps the LLM focus on what's actually important and ignore less general facts that your application may never use
  - Storage - General topics cause the LLM to over-extract which can be unnecessary and may use up storage space unnecessarily

```python
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

At this point, the application is automatically listening to all incoming and outgoing messages.

> [!NOTE]  
> Additionally, doing this will automatically augment the `TurnContext` with a `memory_module` property that is scoped to the conversation for that particular request. During the lifetime of the request, you can access this property to get a `ScopedMemoryModule` that is scoped to the conversation via:
>
> ```python
> memory_module: BaseScopedMemoryModule = context.get("memory_module")
> ```

#### [Optional] Internal Messages

The previous step only automatically stores incoming and outgoing messages from your bot. But sometimes, you may want to store `InternalMessage` as well. These can be used as additional context for extraction of memories, or for your agent to keep track of internal messages that may be required to keep track of the conversation. You can use it after a `tool_call` for example.

```python
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

### Extracting Memories

> [!NOTE]  
> The memory module currently only supports extracting semantic memories about a user. What this means is that that each extracted memory is related to a particular user. We have plans to support extracting memories about a conversation in the future. See [Future Work](#future-work) for more details.

There are two ways to extract memories using the memory module.

1.  Automatically - The memory module will automatically extract memories when the `buffer_size` is reached or the `timeout_seconds` elapses. This is helpful if you want to passively extract memories without having to call any methods.
2.  On-Demand - You can manually trigger the extraction of memories by calling the `memory_module.process_messages()` method. This is helpful if you want to extract memories at a specific point in time. This could be after a specific

#### Automatic Extraction

To enable automatic extraction, you need to call `memory_middleware.memory_module.listen()` when your application starts. This will begin listening to all incoming and outgoing messages and automatically extract memories when the `buffer_size` is reached or the `timeout_seconds` elapses. Here's an example of doing this in a Teams AI SDK application:

```python
async def initialize_memory_module(_app: web.Application):
    await memory_middleware.memory_module.listen()

app.on_startup.append(initialize_memory_module)
```

#### On-Demand Extraction

You might prefer on-demand extraction if you want to extract memories at a specific point in time or after a particular event. For example, you may want to extract memories after a `tool_call` or after a specific message. For this, you can simply call the `process_messages()` method on the `ScopedMemoryModule` that you have access to via the `TurnContext`.

```python
async def extract_memories_after_tool_call(context: TurnContext):
    memory_module: ScopedMemoryModule = context.get('memory_module')
    await memory_module.process_messages() # takes whatever messages are in the buffer and extracts memories
```

> [!NOTE]  
> `memory_module.process_messages()` is not exclusive to `listen()` and can be called at any time to extract memories, even if automatic extraction is enabled.

### Using Short Term Memories / Working Memory

Memory module makes it easy for you to use messages when using an LLM. Since the `memory_module` is already listening to all incoming and outgoing messages, you can simply retrieve the messages from it and use them as context for your LLM.

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

When it's time to use memories in your application, you may use the scoped memory module that you have access to via the `TurnContext`:

```
async def retrieve_device_type_memories(context: TurnContext):
	memory_module: ScopedMemoryModule = context.get('memory_module')
	device_type_memories = await memory_module.search_memories(
		topic=Topic(name="Device Type", description="The type of device the user has"),
		query="What device does the user own?"
	)
```

You may search for memories either using a topic or a natural language query (or both, but not none).

## Logging

You can enable logging when setting up the memory module in the config.

```py
config = MemoryModuleConfig()
config.enable_logging=True,
```

Internally, it uses Python's [logging](https://docs.python.org/3.12/library/logging.html) library to facilitate logging. But setting `MemoryModuleConfig.enable_logging` to True, the module will begin logging all messages to the console.

By default, the logger will log debug messages (and higher serverity) to the console, but you can customize this behavior by setting up the logger in your Python file.

```py
from teams_memory import configure_logging

configure_logging(logging_level=logging.INFO)
```

# Model Performance

We have tested the memory module with `gpt-4o` and `text-embedding-3-small` where it has performed reasonably well with our own datasets and evals. We plan to share details on its performance in the future and also plan to share evaluations for other models.

# Future Work

Teams Memory Module is currently in alpha. We are actively working on improving its performance and adding more features. Here are some of the features we plan to add in the future:

- Evals and performance on other models
- More storage providers (eg. PostgresSQL, CosmosDB, etc.)
- Automatic Message Expiration (eg. messages older than 1 day are automatically deleted)
- Episodic Memory extraction (memories about a conversation, not just a user)
- Sophisticated memory access patterns (eg. memories across multiple groups being shared securely)
