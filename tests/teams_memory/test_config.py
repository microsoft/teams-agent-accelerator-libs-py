from teams_memory.config import StorageConfig


def test_storage_config_default():
    config = StorageConfig()
    assert config.storage_type == "in-memory"
    assert config.db_path is None


def test_storage_config_explicit_sqlite():
    config = StorageConfig(storage_type="sqlite", db_path="test.db")
    assert config.storage_type == "sqlite"
    assert config.db_path == "test.db"


def test_storage_config_auto_sqlite():
    """Test that storage_type is automatically set to sqlite when db_path is provided"""
    config = StorageConfig(db_path="test.db")
    assert config.storage_type == "sqlite"
    assert config.db_path == "test.db"


def test_storage_config_explicit_type_with_db():
    """Test that explicit storage_type is not overridden even with db_path"""
    config = StorageConfig(storage_type="in-memory", db_path="test.db")
    assert config.storage_type == "in-memory"
    assert config.db_path == "test.db"
