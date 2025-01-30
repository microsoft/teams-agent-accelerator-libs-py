> [!IMPORTANT]  
> _`teams_memory` is in alpha. We are still internally validating and testing!_

# Teams Memory Module

The Teams Memory Module is a simple yet powerful library designed to help manage memories for Teams AI Agents. By offloading the responsibility of tracking user-related facts, it enables developers to create more personable and efficient agents.

# Features

- **Seamless Integration with Teams AI SDK**:  
  The memory module integrates directly with the Teams AI SDK via middleware, tracking both incoming and outgoing messages.

- **Automatic Memory Extraction**:  
  Define a set of topics (or use default ones) relevant to your application, and the memory module will automatically extract and store related memories.

- **Simple Short-Term Memory Retrieval**:  
  Easily retrieve working memory using paradigms like "last N minutes" or "last M messages."

- **Query-Based or Topic-Based Memory Retrieval**:  
  Search for existing memories using natural language queries or predefined topics.

# Integration

Integrating the Memory Module into your Teams AI SDK application (or Bot Framework) is straightforward.

## Prerequisites

- **Azure OpenAI or OpenAI Keys**:  
  The LLM layer is built using [LiteLLM](https://docs.litellm.ai/), which supports multiple [providers](https://docs.litellm.ai/docs/providers). However, only Azure OpenAI (AOAI) and OpenAI (OAI) have been tested.

## Integrating into a Teams AI SDK Application

### Installing the Memory Module

```bash
pip install teams-memory
```

### Adding Messages

#### Incoming / Outgoing Messages

Memory extraction requires incoming and outgoing messages to your application. To simplify this, you can use middleware to automate the process.

After building your bot `Application`, create a `MemoryMiddleware` with the following configurations:

- **`llm`**: Configuration for the LLM (required).
- **`storage`**: Configuration for the storage layer. Defaults to `InMemoryStorage` if not provided.
- **`buffer_size`**: Minimum size of the message buffer before memory extraction is triggered.
- **`timeout_seconds`**: Time elapsed after the buffer starts filling up before extraction occurs.
  - **Note**: Extraction occurs when either the `buffer_size` is reached or the `timeout_seconds` elapses, whichever happens first.
- **`topics`**: Topics relevant to your application. These help the LLM focus on important information and avoid unnecessary extractions.

```python
memory_middleware = MemoryMiddleware(
    config=MemoryModuleConfig(
        llm=LLMConfig(**memory_llm_config),
        storage=StorageConfig(
            db_path=os.path.join(os.path.dirname(__file__), "data", "memory.db")
        ),  # Uses SQLite if `db_path` is provided
        timeout_seconds=60,  # Extraction occurs 60 seconds after the first message
        enable_logging=True,  # Helpful for debugging
        topics=[
            Topic(name="Device Type", description="The type of device the user has"),
            Topic(name="Operating System", description="The operating system for the user's device"),
            Topic(name="Device Year", description="The year of the user's device"),
        ],  # Example topics for a tech-assistant agent
    )
)
bot_app.adapter.use(memory_middleware)
```

At this point, the application automatically listens to all incoming and outgoing messages.

> [!TIP]  
> This integration augments the `TurnContext` with a `memory_module` property, scoped to the conversation for the request. Access it via:
>
> ```python
> memory_module: BaseScopedMemoryModule = context.get("memory_module")
> ```

#### [Optional] Internal Messages

The previous step only stores incoming and outgoing messages. You also have the option to o store `InternalMessage` objects (e.g., for additional context or tracking internal conversation states) via:

```python
async def add_internal_message(self, context: TurnContext, tool_call_name: str, tool_call_result: str):
    conversation_ref_dict = TurnContext.get_conversation_reference(context.activity)
    memory_module: BaseScopedMemoryModule = context.get("memory_module")
    await memory_module.add_message(
        InternalMessageInput(
            content=json.dumps({"tool_call_name": tool_call_name, "result": tool_call_result}),
            author_id=conversation_ref_dict.bot.id,
            conversation_ref=memory_module.conversation_ref,
        )
    )
    return True
```

### Extracting Memories

> [!NOTE]  
> The memory module currently supports extracting **semantic memories** about a user. Future updates will include support for conversation-level memories. See [Future Work](#future-work) for details.

There are two ways to extract memories:

1. **Automatic Extraction**: Memories are extracted when the `buffer_size` is reached or the `timeout_seconds` elapses.
2. **On-Demand Extraction**: Manually trigger extraction by calling `memory_module.process_messages()`.

#### Automatic Extraction

Enable automatic extraction by calling `memory_middleware.memory_module.listen()` when your application starts. This listens to messages and triggers extraction based on the configured conditions.

```python
async def initialize_memory_module(_app: web.Application):
    await memory_middleware.memory_module.listen()

async def shutdown_memory_module(_app: web.Application):
    await memory_middleware.memory_module.shutdown()

app.on_startup.append(initialize_memory_module)
app.on_shutdown.append(shutdown_memory_module)

web.run_app(app, host="localhost", port=Config.PORT)
```

> [!IMPORTANT]  
> When performing automatic extraction via `listen()`, it's important to ensure that you also configure your application to cleanup resources when the application shuts down using the `shutdown()` method.

#### On-Demand Extraction

Use on-demand extraction to trigger memory extraction at specific points, such as after a `tool_call` or a particular message.

```python
async def extract_memories_after_tool_call(context: TurnContext):
    memory_module: ScopedMemoryModule = context.get('memory_module')
    await memory_module.process_messages()  # Extracts memories from the buffer
```

> [!NOTE]  
> `memory_module.process_messages()` can be called at any time, even if automatic extraction is enabled.

### Using Short-Term Memories (Working Memory)

The memory module simplifies the retrieval of recent messages for use as context in your LLM.

```python
async def build_llm_messages(self, context: TurnContext, system_message: str):
    memory_module: BaseScopedMemoryModule = context.get("memory_module")
    assert memory_module
    messages = await memory_module.retrieve_chat_history(
        ShortTermMemoryRetrievalConfig(last_minutes=1)
    )
    llm_messages: List = [
        {"role": "system", "content": system_prompt},
        *[
            {"role": "user" if message.type == "user" else "assistant", "content": message.content}
            for message in messages
        ],  # UserMessages have a `role` of `user`; others are `assistant`
    ]
    return llm_messages
```

### Using Extracted Semantic Memory

Access extracted memories via the `ScopedMemoryModule` available in the `TurnContext`:

```python
async def retrieve_device_type_memories(context: TurnContext):
    memory_module: ScopedMemoryModule = context.get('memory_module')
    device_type_memories = await memory_module.search_memories(
        topic="Device Type", # This name must match the topic name in the config
        query="What device does the user own?"
    )
```

You can search for memories using a topic, a natural language query, or both.

## Logging

Enable logging in the memory module configuration:

```python
config = MemoryModuleConfig()
config.enable_logging = True
```

The module uses Python's [logging](https://docs.python.org/3.12/library/logging.html) library. By default, it logs debug messages (and higher severity) to the console. Customize the logging behavior as follows:

```python
from teams_memory import configure_logging

configure_logging(logging_level=logging.INFO)
```

# Model Performance

| Model  | Embedding Model        | Tested | Notes                                   |
| ------ | ---------------------- | ------ | --------------------------------------- |
| gpt-4o | text-embedding-3-small | ✅     | Tested via both OpenAI and Azure OpenAI |

# Future Work

The Teams Memory Module is in active development. Planned features include:

- **Evals and Performance Testing**: Support for additional models.
- **More Storage Providers**: Integration with PostgreSQL, CosmosDB, etc.
- **Automatic Message Expiration**: Delete messages older than a specified duration (e.g., 1 day).
- **Episodic Memory Extraction**: Memories about conversations, not just users.
- **Sophisticated Memory Access Patterns**: Secure sharing of memories across multiple groups.
