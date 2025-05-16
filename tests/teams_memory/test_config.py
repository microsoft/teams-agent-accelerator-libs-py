from pathlib import Path

import pytest
from teams_memory.config import (
    AzureAISearchStorageConfig,
    InMemoryStorageConfig,
    LLMConfig,
    MemoryModuleConfig,
    SQLiteStorageConfig,
    Topic,
)


# Test InMemoryStorageConfig construction
def test_in_memory_storage_config():
    config = InMemoryStorageConfig()
    assert config.storage_type == "in-memory"


# Test AzureAISearchStorageConfig construction
def test_azure_ai_search_storage_config():
    config = AzureAISearchStorageConfig(
        endpoint="https://example.search.windows.net",
        api_key="test-key",
        index_name="test-index",
        embedding_dimensions=1536,
    )
    assert config.storage_type == "azure_ai_search"
    assert config.endpoint == "https://example.search.windows.net"
    assert config.api_key == "test-key"
    assert config.index_name == "test-index"
    assert config.embedding_dimensions == 1536


# Minimal valid LLMConfig for MemoryModuleConfig
def minimal_llm_config():
    return LLMConfig(model="gpt-4", api_key="dummy")


# Test MemoryModuleConfig with only global storage (SQLite)
def test_memory_module_config_global_storage():
    config = MemoryModuleConfig(
        storage=SQLiteStorageConfig(db_path=Path("test.db")),
        llm=minimal_llm_config(),
    )
    assert isinstance(config.get_storage_config("memory"), SQLiteStorageConfig)
    assert config.get_storage_config("message").storage_type == "sqlite"


# Test MemoryModuleConfig with per-type storage
def test_memory_module_config_per_type_storage_missing_some():
    with pytest.raises(
        ValueError,
        match="Please set either the per-type config or the global 'storage' config",
    ):
        MemoryModuleConfig(
            memory_storage=InMemoryStorageConfig(),
            message_storage=SQLiteStorageConfig(db_path=Path("test.db")),
            llm=minimal_llm_config(),
        )


def test_memory_module_config_per_type_storage():
    config = MemoryModuleConfig(
        memory_storage=InMemoryStorageConfig(),
        message_storage=SQLiteStorageConfig(db_path=Path("test.db")),
        message_buffer_storage=SQLiteStorageConfig(db_path=Path("test.db")),
        scheduled_events_storage=SQLiteStorageConfig(db_path=Path("test.db")),
        llm=minimal_llm_config(),
    )
    assert isinstance(config.get_storage_config("memory"), InMemoryStorageConfig)
    assert isinstance(config.get_storage_config("message"), SQLiteStorageConfig)
    assert isinstance(config.get_storage_config("message_buffer"), SQLiteStorageConfig)
    assert isinstance(
        config.get_storage_config("scheduled_events"), SQLiteStorageConfig
    )


# Test get_storage_config error if missing
def test_memory_module_config_missing_storage():
    with pytest.raises(ValueError, match="No storage config provided for memory"):
        MemoryModuleConfig(llm=minimal_llm_config()).get_storage_config("memory")


# Test validation error if Azure AI Search is global and others missing
def test_memory_module_config_azure_global_missing_others():
    with pytest.raises(
        ValueError,
        match="must provide a non-Azure AI Search config for message_storage",
    ):
        MemoryModuleConfig(
            storage=AzureAISearchStorageConfig(
                endpoint="https://example.search.windows.net",
                api_key="test-key",
                index_name="test-index",
            ),
            llm=minimal_llm_config(),
        )


# Test validation error if memory_storage is Azure AI Search and others missing
def test_memory_module_config_azure_memory_missing_others():
    with pytest.raises(
        ValueError,
        match="Please set either the per-type config or the global 'storage' config",
    ):
        MemoryModuleConfig(
            memory_storage=AzureAISearchStorageConfig(
                endpoint="https://example.search.windows.net",
                api_key="test-key",
                index_name="test-index",
            ),
            llm=minimal_llm_config(),
        )


# Test validation passes if memory_storage is Azure AI Search but global is non-Azure
def test_memory_module_config_azure_memory_with_non_azure_global():
    config = MemoryModuleConfig(
        memory_storage=AzureAISearchStorageConfig(
            endpoint="https://example.search.windows.net",
            api_key="test-key",
            index_name="test-index",
        ),
        storage=SQLiteStorageConfig(db_path=Path("test.db")),
        llm=minimal_llm_config(),
    )
    assert isinstance(config.get_storage_config("memory"), AzureAISearchStorageConfig)
    assert isinstance(config.get_storage_config("message"), SQLiteStorageConfig)


# Test topics field with custom topics
def test_memory_module_config_custom_topics():
    topics = [
        Topic(name="A", description="desc A"),
        Topic(name="B", description="desc B"),
    ]
    config = MemoryModuleConfig(
        storage=InMemoryStorageConfig(),
        llm=minimal_llm_config(),
        topics=topics,
    )
    assert config.topics == topics


# Test default topics present
def test_memory_module_config_default_topics():
    config = MemoryModuleConfig(
        storage=InMemoryStorageConfig(),
        llm=minimal_llm_config(),
    )
    assert len(config.topics) >= 1
    assert isinstance(config.topics[0], Topic)
