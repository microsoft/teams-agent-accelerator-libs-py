# Teams Memory Examples

This directory contains example applications demonstrating how to use the `teams_memory` package.

## Azure AI Search Console Test

The [`azure_search_console_test.py`](azure_search_console_test.py) script demonstrates how to use Azure AI Search storage functionality in the `teams_memory` package. It initializes a `MemoryCore` with Azure AI Search storage, adds sample memories, and retrieves them based on a query.

### Environment Variables

The example requires several environment variables to be set. You can set these variables in two ways:

#### Option 1: Using a `.env` file

1. Create a `.env` file in the `examples/` directory or in the project root.
2. Copy the contents from `.env.sample` and replace the placeholder values with your actual values:

```
# Azure AI Search Configuration
AZURE_SEARCH_SERVICE_NAME="your-search-service-name"
AZURE_SEARCH_INDEX_NAME="test-console-memories"
AZURE_SEARCH_API_KEY="your-search-api-key"
AZURE_SEARCH_ENDPOINT="https://your-search-service-name.search.windows.net"

# OpenAI Configuration (required for embeddings)
OPENAI_API_KEY="your-openai-api-key"
# OPENAI_ENDPOINT="your-custom-openai-endpoint"  # Optional, uncomment if using Azure OpenAI
```

#### Option 2: Setting environment variables directly

You can also set the environment variables directly in your shell:

```bash
export AZURE_SEARCH_SERVICE_NAME="your-search-service-name"
export AZURE_SEARCH_INDEX_NAME="test-console-memories"
export AZURE_SEARCH_API_KEY="your-search-api-key"
export AZURE_SEARCH_ENDPOINT="https://your-search-service-name.search.windows.net"
export OPENAI_API_KEY="your-openai-api-key"
# export OPENAI_ENDPOINT="your-custom-openai-endpoint"  # Optional, uncomment if using Azure OpenAI
```

### Running the Example

Once you've set up the environment variables, you can run the example:

```bash
python azure_search_console_test.py
```

The script will:
1. Initialize a `MemoryCore` with Azure AI Search storage
2. Add sample memories
3. Retrieve memories based on a query
4. Retrieve memories by topic