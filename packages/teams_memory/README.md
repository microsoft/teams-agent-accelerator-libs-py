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
        storage=SQLiteStorageConfig(
            db_path=os.path.join(os.path.dirname(__file__), "data", "memory.db")
        ),
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

You can search for memories using a topic, a natural language query, or both. This method returns a list of relevant memories. Internally, it does an embeddings search, so it's possible that there could be some false positives.

If you want use the memories to answer a question, you can use the `ask` method. Internally, it uses the `search_memories` method, but it also uses the LLM to answer the question.

```python
async def retrieve_device_details(context: TurnContext):
    memory_module: ScopedMemoryModule = context.get('memory_module')
    result = await memory_module.ask(
        question="Has the user owned multiple devices in the past?",
        topic="Device Type",
    )
    if result:
        answer, memories = result
        print(answer) # "Yes the user has owned a Macbook and a Windows PC"
        print(memories) # List of memories that were used to answer the question
```

### Memory Attributions

The memory module stores attributions (citations) for its memories. Attributions are the original messages from which a memory was extracted. A single memory can have multiple attributions, as the same information may have been mentioned multiple times or combines information from multiple messages. Attributions are important because they allow users to verify the source and accuracy of memories by seeing the original messages where the information appeared.

The `memory_module.get_memories_with_attributions` method returns a list of `MemoryWithAttributions` objects. Each object contains:

- A `memory` field with the extracted memory
- A `messages` field containing a list of the original messages that were used to create the memory

Each message in the `messages` list includes a `deep_link` property that provides a direct link to view the original message in Teams.

```python
async def build_teams_citations(context: TurnContext, content: str, memory: Memory):
    memory_module: ScopedMemoryModule = context.get('memory_module')
    memories = await memory_module.get_memories_with_attributions(
        memory_ids=[memory.id]
    )

    memory = memories[0].memory
    message = memories[0].messages
    message.deep_link # can be used to navigate to the message in Teams

    # build citations for Teams
    ...
```

For more information on how to build citations for Teams, see [Teams documentation](https://learn.microsoft.com/en-us/microsoftteams/platform/bots/how-to/bot-messages-ai-generated-content?tabs=desktop%2Cbotmessage).

## Storage

The memory module architecture is designed with reasonable storage abstractions that allow you to change the storage layer. There are 4 main parts to storage:

1. **Memory Storage**: Storage for semantic memories.
2. **Message Storage**: Storage for messages.
3. **Message Buffer Storage**: Storage for the message buffer.
4. **Scheduled Events Storage**: Storage for scheduled events.

You can change out the storage layer for each of these components, or use the same storage for all of them. By default, if no storage options are provided, the memory module will use an in-memory storage layer for each of these components. When constructing the `MemoryModuleConfig`, you can provide a custom storage config for each of these components or a global storage config, which will be used for all components. Below are some of the storage options available:

| Storage Type                 | Type of possible storage                          | Description                                                                                                  |
| ---------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `InMemoryStorageConfig`      | memory, message_buffer, scheduled_events          | Uses an in-memory storage layer.                                                                             |
| `SQLiteStorageConfig`        | memory, message, message_buffer, scheduled_events | Uses a SQLite database.                                                                                      |
| `AzureAISearchStorageConfig` | memory                                            | Uses an [Azure AI Search index](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search). |

### In-Memory Storage

In-memory storage is the default storage layer for the memory module. It uses an in-memory storage layer for all components. It's best for testing and development. By default, if no storage options are provided, the memory module will use an in-memory storage layer for all components.

### SQLite Storage

SQLite storage is a good option for small applications. It uses a SQLite database to store values in a single file.

```python
config = MemoryModuleConfig(
    storage=SQLiteStorageConfig(db_path="path/to/db.sqlite")
)
```

### Azure AI Search Storage

[Azure AI Search storage](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search) is a great option for storing memories in a production environment. You will need to create an [Azure AI Search resource](https://learn.microsoft.com/en-us/azure/search/search-create-service-portal) before using this storage. Once you have this, the memory module will automatically construct an index for you, and start using it to store and retrieve memories.

First install the `azure-search` extra:

```bash
pip install teams-memory[azure-search]
```

Then, configure the memory module to use Azure AI Search storage and installs necessary dependencies.

```python
# Assumes you have set the following environment variables:
# AZURE_SEARCH_ENDPOINT
# AZURE_SEARCH_KEY
# AZURE_SEARCH_INDEX_NAME
# They can be found in the Azure AI Search resource in the Azure portal
config = MemoryModuleConfig(
    memory_storage=AzureAISearchStorageConfig(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        key=os.getenv("AZURE_SEARCH_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    ),
    storage=InMemoryStorageConfig(), # you need to provide a storage config for other storage components
)
```

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
