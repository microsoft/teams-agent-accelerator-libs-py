#!/usr/bin/env python
"""
Azure AI Search Memory Storage Test Console Application

This script demonstrates the use of Azure AI Search storage functionality
in the teams_memory package. It initializes a MemoryCore with Azure AI Search
storage, adds sample memories, and retrieves them based on a query.

Required environment variables:
- AZURE_SEARCH_SERVICE_NAME: The name of the Azure AI Search service
- AZURE_SEARCH_INDEX_NAME: The name of the search index (e.g., "test-console-memories")
- AZURE_SEARCH_API_KEY: The API key for the Azure AI Search service
- AZURE_SEARCH_ENDPOINT: The endpoint URL for the Azure AI Search service
"""
import os
import asyncio
import datetime
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from teams_memory.core.memory_core import MemoryCore
from teams_memory.config import MemoryModuleConfig, StorageConfig, LLMConfig
from teams_memory.services.llm_service import LLMService
from teams_memory.interfaces.types import BaseMemoryInput, MemoryType

# Try to load environment variables from .env files
# First check in the examples directory, then in the project root
env_paths = [
    Path(__file__).parent / ".env",  # examples/.env
    Path(__file__).parent.parent / ".env",  # .env in project root
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print("No .env file found. Using environment variables from the shell.")



async def add_sample_memory(memory_core: MemoryCore, content: str, user_id: str, topics: Optional[list[str]] = None) -> Optional[str]:
    """Add a sample memory to the memory core."""
    print(f"Adding memory: {content}")
    
    memory = BaseMemoryInput(
        content=content,
        created_at=datetime.datetime.now(),
        user_id=user_id,
        memory_type=MemoryType.SEMANTIC,
        topics=topics or [],
    )
    
    # For the sample, we'll use the content as the embedding
    # In a real application, you would use the LLM service to generate embeddings
    embedding_text = await memory_core.lm.embedding(input=[content])
    embedding_vector = embedding_text.data[0]["embedding"]
    
    from teams_memory.interfaces.types import TextEmbedding
    embedding = TextEmbedding(text=content, embedding_vector=embedding_vector)
    
    memory_id = await memory_core.storage.store_memory(memory, embedding_vectors=[embedding])
    if memory_id:
        print(f"Memory added with ID: {memory_id}")
    else:
        print("Failed to add memory")
    return memory_id


async def main():
    """Main function to run the Azure AI Search memory test."""
    # Check for required environment variables
    required_vars = [
        "AZURE_SEARCH_SERVICE_NAME",
        "AZURE_SEARCH_INDEX_NAME",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_ENDPOINT"
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        print("Error: The following required environment variables are not set:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these environment variables and try again.")
        print("You can either:")
        print("1. Set them directly in your shell:")
        print('   export AZURE_SEARCH_SERVICE_NAME="your-search-service-name"')
        print('   export AZURE_SEARCH_INDEX_NAME="test-console-memories"')
        print('   export AZURE_SEARCH_API_KEY="your-search-api-key"')
        print('   export AZURE_SEARCH_ENDPOINT="https://your-search-service-name.search.windows.net"')
        print("\n2. Create a .env file in the examples/ directory or project root with the following content:")
        print('   AZURE_SEARCH_SERVICE_NAME="your-search-service-name"')
        print('   AZURE_SEARCH_INDEX_NAME="test-console-memories"')
        print('   AZURE_SEARCH_API_KEY="your-search-api-key"')
        print('   AZURE_SEARCH_ENDPOINT="https://your-search-service-name.search.windows.net"')
        print('   OPENAI_API_KEY="your-openai-api-key"  # Required for embeddings')
        return
    
    print("Initializing Azure Search Memory...")
    
    # Configure the memory module
    config = MemoryModuleConfig(
        storage=StorageConfig(
            storage_type="azure-search",
            search_service_name=os.environ["AZURE_SEARCH_SERVICE_NAME"],
            search_index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
            search_api_key=os.environ["AZURE_SEARCH_API_KEY"],
            search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        ),
        llm=LLMConfig(
            model="gpt-3.5-turbo",  # Default model for completions
            embedding_model="text-embedding-ada-002",  # Default model for embeddings
            api_key=os.environ.get("OPENAI_API_KEY"),  # Get OpenAI API key from environment
            api_base=os.environ.get("OPENAI_ENDPOINT"),  # Get OpenAI endpoint from environment if set
        ),
        enable_logging=True,  # Enable logging for better visibility
    )
    
    # Initialize LLM service and memory core
    llm_service = LLMService(config.llm)
    memory_core = MemoryCore(config, llm_service)
    
    print("Azure Search Memory initialized successfully.")
    
    # Add sample memories
    user_id = "test-user-123"
    
    await add_sample_memory(
        memory_core,
        "The user enjoys hiking in the mountains on weekends.",
        user_id,
        ["General Interests and Preferences"]
    )
    
    await add_sample_memory(
        memory_core,
        "The user lives in Seattle and works as a software engineer.",
        user_id,
        ["General Facts about the user"]
    )
    
    await add_sample_memory(
        memory_core,
        "The user has a dog named Max that they adopted last year.",
        user_id,
        ["General Facts about the user"]
    )
    
    # Retrieve memories based on a query
    query = "hiking"
    print(f"\nRetrieving memories for query: '{query}'")
    
    memories = await memory_core.search_memories(user_id=user_id, query=query)
    
    if memories:
        print(f"Found {len(memories)} memories:")
        for i, memory in enumerate(memories, 1):
            print(f"{i}. {memory.content} (ID: {memory.id})")
    else:
        print("No memories found for the query.")
    
    # Retrieve memories by topic
    topic = "General Facts about the user"
    print(f"\nRetrieving memories for topic: '{topic}'")
    
    memories = await memory_core.search_memories(user_id=user_id, topic=topic)
    
    if memories:
        print(f"Found {len(memories)} memories:")
        for i, memory in enumerate(memories, 1):
            print(f"{i}. {memory.content} (ID: {memory.id})")
    else:
        print("No memories found for the topic.")


if __name__ == "__main__":
    asyncio.run(main())