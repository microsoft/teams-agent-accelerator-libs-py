import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from memory_module.config import LLMConfig, MemoryModuleConfig
from memory_module.services.scheduled_events_service import ScheduledEventsService


@pytest.fixture
def config():
    """Fixture that provides a MemoryModuleConfig instance."""
    return MemoryModuleConfig(
        db_path=Path(__file__).parent / "data" / "tests" / "memory_module.db",
        timeout_seconds=1,
        llm=LLMConfig(),
    )


@pytest.fixture
def service(config):
    """Fixture that provides a ScheduledEventsService instance."""
    # Delete the db file if it exists
    if config.db_path.exists():
        config.db_path.unlink()
    return ScheduledEventsService(config=config)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_scheduled_events(service):
    """Fixture to cleanup scheduled events after each test."""
    yield
    await service.cleanup()


@pytest.mark.asyncio
async def test_add_event(service):
    """Test that events are properly added to pending_events."""
    now = datetime.now()
    await service.add_event("test1", {"id": "test1", "data": "test"}, now)

    assert len(service.pending_events) == 1
    assert any(task.id == "test1" for task in service.pending_events)

    await asyncio.sleep(0.1)
    assert len(service.pending_events) == 0


@pytest.mark.asyncio
async def test_callback_execution(config):
    """Test that callbacks are executed when events trigger."""
    service = ScheduledEventsService(config=config)
    try:
        callback_called = False
        callback_data = None

        async def test_callback(id: str, obj: Any, time: datetime):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = obj

        service.callback = test_callback

        # Schedule event for 0.1 seconds from now
        now = datetime.now()
        test_data = {"test": "data"}
        await service.add_event("test1", test_data, now + timedelta(seconds=0.1))

        # Wait a bit longer than the scheduled time
        await asyncio.sleep(0.2)

        assert callback_called
        assert callback_data == test_data
        assert (
            len(service.pending_events) == 0
        )  # Event should be removed after execution
    finally:
        # Clean up any remaining tasks
        await service.flush()


@pytest.mark.asyncio
async def test_cancel_existing_event(service):
    """Test that adding an event with same ID cancels the existing one."""
    now = datetime.now()
    future = now + timedelta(seconds=1)

    # Add initial event
    await service.add_event("test1", {"data": "initial"}, future)
    assert len(service.pending_events) == 1

    # Add event with same ID
    await service.add_event("test1", {"data": "updated"}, future)
    assert len(service.pending_events) == 1
    assert any(task.id == "test1" for task in service.pending_events)


@pytest.mark.asyncio
async def test_immediate_execution_for_past_time(service):
    """Test that events with past times execute immediately."""
    callback_called = False

    async def test_callback(id: str, obj: Any, time: datetime):
        nonlocal callback_called
        callback_called = True

    service.callback = test_callback

    # Schedule event for a time in the past
    past_time = datetime.now() - timedelta(minutes=1)
    await service.add_event("test1", {"id": "test1", "data": "test"}, past_time)

    # Small delay to allow immediate execution
    await asyncio.sleep(0.1)

    assert callback_called
    assert len(service.pending_events) == 0


@pytest.mark.asyncio
async def test_multiple_events(service):
    """Test handling of multiple events with different IDs."""
    now = datetime.now()
    future = now + timedelta(seconds=0.5)

    # Add multiple events
    await service.add_event("test1", {"id": "test1", "data": "1"}, future)
    await service.add_event("test2", {"id": "test2", "data": "2"}, future)

    assert len(service.pending_events) == 2
    event_names = [task.id for task in service.pending_events]
    assert "test1" in event_names
    assert "test2" in event_names
