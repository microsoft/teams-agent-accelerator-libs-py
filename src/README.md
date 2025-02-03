# Overview of the Basic AI Chatbot template

This app template is built on top of [Teams AI library](https://aka.ms/teams-ai-library) and [Teams Memory](../packages/teams_memory/README.md).

# What this sample is about

This sample showcases a Tech support Assistant agent which can help users with their device problems. It demonstrates how an agent may use Semantic Memories to answer questions more efficiently.

See [tech_assistant_agent](./tech_assistant_agent/README.md) for more details on the tech support assistant agent. Its [prompts](./tech_assistant_agent/prompts.py) are especially helpful to understand how this agent works.

## How does it work?

### Topics

The sample is initialized with a list of topics that it cares about. These are topics that the agent wants to remember about the user. Specifically, they are:

1. Device Type
2. Operating System
3. Device Year

See [tools.py](./tech_assistant_agent/tools.py) for the definition of the topics.

### Middleware

When you initialize the `MemoryMiddleware`, it will start to record all the messages that are incoming or outgoing from the bot. These messages are then used by the agent as working memory and also for extraction for long term memory.

By setting up the middleware we also get access to a scoped version of the `memory_module` from the TurnContext. This memory module is scoped to the conversation that the TurnContext is built for.

See [bot.py](./bot.py) for the initialization of the `MemoryMiddleware`.

> [!TIP]
> You'll notice that for the sample, the `timeout_seconds` is 60 seconds. The extraction here is set to be a bit aggressive (extract every 1 minute if there is a message in a conversation) to demonstrate memory extraction, but a higher threshhold here is reasonable to set here.

### Automatic extraction

The Memory Module can be set up to automatically extract long term memories from the working memory. When the application server starts up, by calling `memory_middleware.memory_module.listen()`, it will start to trigger extraction of memories in depending on the configuration passed when the `MemoryMiddleware` (or `MemoryModule`) was initialized. This work happens in a background thread and is non-blocking.

See [app.py](./app.py) for the initialization of the `MemoryMiddleware`. Note that when `listen` is called, you also should call `shutdown` when the application is shutting down.

> [!NOTE]
> The alternative to automatic extraction is explicit extraction. This can be accomplished by calling `memory_module.process_messages` which will process all the messages that are in the message buffer.

### Using working memory

The agent can use the conversational messages as working memory to build up contexts for LLM calls for the agent. In addition to the incoming and outgoing messages, the agent can also add internal messages to the working memory.

See [primary_agent.py](./tech_assistant_agent/primary_agent.py) for how working memory is used, and also how internal messages are added to the working memory.

### Using long term semantic memories

The tech support assistant can search for memories from a tool call (See [get_memorized_fields](./tech_assistant_agent/tools.py)). In this tool call, the agent searches memories for a given topic. Depending on if the memories are found or not, the agent can then continue to ask the user for the information or proceed with the flow (like confirming the memories).

### Citing memories

If the agent finds memories that are relevant to the task at-hand, the tech support assistant can ask for confirmations of the memories and cite the original sources of the memories.

See [confirm_memorized_fields](./tech_assistant_agent/tools.py) for the implementation of the tool call.

# Running the sample

## Get started with the sample

> **Prerequisites**
>
> To run the template in your local dev machine, you will need:
>
> - [Python](https://www.python.org/), version 3.8 to 3.11.
> - [Python extension](https://code.visualstudio.com/docs/languages/python), version v2024.0.1 or higher.
> - [Teams Toolkit Visual Studio Code Extension](https://aka.ms/teams-toolkit) latest version or [Teams Toolkit CLI](https://aka.ms/teams-toolkit-cli).
> - An account with [OpenAI](https://platform.openai.com/).
> - [Node.js](https://nodejs.org/) (supported versions: 16, 18) for local debug in Test Tool.

### Configurations

1. create _.env_ in root folder. Copy the below template into it.

```
# AZURE CONFIG

AZURE_OPENAI_API_KEY=<API key>
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_API_BASE=https://<domain name>.openai.azure.com
AZURE_OPENAI_API_VERSION=<version number>

# OPENAI CONFIG

OPENAI_MODEL_NAME=gpt-4o
OPENAI_API_KEY=<API key>
OPENAI_EMBEDDING_MODEL_NAME=text-embedding-3-small
```

Remember, that these are also used by the Memory Module to extract and retrieve memories.

Fill out only one of Azure OpenAI and OpenAI configurations.

### Debug with Teams Test Tool

1. Open a new terminal under root folder.
1. run `npm install -g @microsoft/teamsapp-cli`
1. run `uv sync`
1. run `.venv\Scripts\Activate`
1. run `python src/app.py`  
   If success, server will start on `http://localhost:3978`
1. Open another new Terminal under root folder
1. Install the teams app test tool (if you haven't already done that)
   - run `mkdir -p src/devTool/teamsapptester` (or `New-Item -ItemType Directory -Path src/devTool/teamsapptester -Force` on Powershell)
   - run `npm i @microsoft/teams-app-test-tool --prefix "src/devTools/teamsapptester"`
1. run `node src/devTools/teamsapptester/node_modules/@microsoft/teams-app-test-tool/cli.js start`  
   If success, a test website will show up
   ![alt text](image.png)

### Debug in Teams

1. Open a new terminal under root folder.
1. run `uv sync`
1. run `.venv\Scripts\Activate`
1. Open this folder as a VSCode workspace.
1. Navigate to the `Run and Debug` tab in VSCode, and select `Debug in Teams (Edge)`. This will start the flow to sideload the bot into Teams, start the server locally, and start the tunnel that exposes the server to the web.

### Deploy to Azure

Currently the scaffolding only supports Azure OpenAI related configurations but can be easily update to support OpenAI configuration.

1. Open a new terminal under root folder.
1. run `uv sync`
1. run `.venv\Scripts\Activate`
1. Build the memory module into a distribtuion file by doing `uv build packages/teams_memory`. This should create the artifact `dist/teams_memory-0.1.0.tar.gz`. Copy this into the `src/dist/` folder.
1. Open this folder as a VSCode workspace.
1. Copy the contents of the `.env` file and add it to the `env/.env.dev.user` file.
1. Navigate to the Teams Toolkit extension in VSCode.
1. Under `Lifecycle`, first click `Provision` to provision resources to Azure.
1. Then click `Deploy`, this should deploy the project to the Azure App Service instance, and run the start up script.
1. If the above two steps completed successfully, then click `Publish`. This will create an app package in `./appPackage/build/appPackage.dev.zip`.
1. Sideload the app package in Teams and start chatting with the bot.

**Congratulations**! You are running an application that can now interact with users in Teams:

> For local debugging using Teams Toolkit CLI, you need to do some extra steps described in [Set up your Teams Toolkit CLI for local debugging](https://aka.ms/teamsfx-cli-debugging).

![ai chat bot](https://github.com/OfficeDev/TeamsFx/assets/9698542/9bd22201-8fda-4252-a0b3-79531c963e5e)

## What's included in the template

| Folder       | Contents                                     |
| ------------ | -------------------------------------------- |
| `.vscode`    | VSCode files for debugging                   |
| `appPackage` | Templates for the Teams application manifest |
| `infra`      | Templates for provisioning Azure resources   |
| `src`        | The source code for the application          |

The following files can be customized and demonstrate an example implementation to get you started.

| File                            | Contents                                               |
| ------------------------------- | ------------------------------------------------------ |
| `src/app.py`                    | Hosts an aiohttp api server and exports an app module. |
| `src/bot.py`                    | Handles business logics for the Basic AI Chatbot.      |
| `src/config.py`                 | Defines the environment variables.                     |
| `src/prompts/chat/skprompt.txt` | Defines the prompt.                                    |
| `src/prompts/chat/config.json`  | Configures the prompt.                                 |

The following are Teams Toolkit specific project files. You can [visit a complete guide on Github](https://github.com/OfficeDev/TeamsFx/wiki/Teams-Toolkit-Visual-Studio-Code-v5-Guide#overview) to understand how Teams Toolkit works.

| File                    | Contents                                                                                                                                  |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `teamsapp.yml`          | This is the main Teams Toolkit project file. The project file defines two primary things: Properties and configuration Stage definitions. |
| `teamsapp.local.yml`    | This overrides `teamsapp.yml` with actions that enable local execution and debugging.                                                     |
| `teamsapp.testtool.yml` | This overrides `teamsapp.yml` with actions that enable local execution and debugging in Teams App Test Tool.                              |

## Extend the template

You can follow [Build a Basic AI Chatbot in Teams](https://aka.ms/teamsfx-basic-ai-chatbot) to extend the Basic AI Chatbot template with more AI capabilities, like:

- [Customize prompt](https://aka.ms/teamsfx-basic-ai-chatbot#customize-prompt)
- [Customize user input](https://aka.ms/teamsfx-basic-ai-chatbot#customize-user-input)
- [Customize conversation history](https://aka.ms/teamsfx-basic-ai-chatbot#customize-conversation-history)
- [Customize model type](https://aka.ms/teamsfx-basic-ai-chatbot#customize-model-type)
- [Customize model parameters](https://aka.ms/teamsfx-basic-ai-chatbot#customize-model-parameters)
- [Handle messages with image](https://aka.ms/teamsfx-basic-ai-chatbot#handle-messages-with-image)

## Additional information and references

- [Teams Toolkit Documentations](https://docs.microsoft.com/microsoftteams/platform/toolkit/teams-toolkit-fundamentals)
- [Teams Toolkit CLI](https://aka.ms/teamsfx-toolkit-cli)
- [Teams Toolkit Samples](https://github.com/OfficeDev/TeamsFx-Samples)

## Known issue

- If you use `Debug in Test Tool` to local debug, you might get an error `InternalServiceError: connect ECONNREFUSED 127.0.0.1:3978` in Test Tool log. You can wait for Python launch console ready and then refresh the front end web page.
- When you use `Launch Remote in Teams` to remote debug after deployment, you might loose interaction with your bot. This is because the remote service needs to restart. Please wait for several minutes to retry it.
