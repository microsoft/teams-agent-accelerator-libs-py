import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Awaitable, Callable, List, Optional

from pydantic import BaseModel, validator


class Event(BaseModel):
    """Represents a scheduled event with an ID, associated object, and execution time."""

    id: str
    object: Any
    time: datetime

    @validator("object")
    def validate_object_serializable(cls, v: Any) -> Any:
        """Ensure the object is JSON serializable.

        Handles:
        - Basic JSON types (dict, list, str, int, etc.)
        - Pydantic models (converts to dict)
        - Objects with to_json() method
        """
        try:
            if isinstance(v, BaseModel):
                # Handle Pydantic models
                return v.model_dump()
            elif hasattr(v, "to_json"):
                # Handle objects with to_json method
                return v.to_json()
            else:
                # Try regular JSON serialization
                json.dumps(v)
                return v
        except (TypeError, ValueError) as e:
            raise ValueError(f"Object must be JSON serializable: {e}")


class BaseScheduledEventsService(ABC):
    """Abstract base class for managing scheduled events."""

    @property
    @abstractmethod
    def callback(self) -> Optional[Callable[[str, Any, datetime], Awaitable[None]]]:
        """The callback to be executed when an event is triggered.

        Callback signature: async def callback(id: str, object: Any, time: datetime)
        """
        pass

    @callback.setter
    @abstractmethod
    def callback(self, value: Callable[[str, Any, datetime], Awaitable[None]]) -> None:
        pass

    @property
    @abstractmethod
    def pending_events(self) -> List[Event]:
        """Get list of pending events."""
        pass

    @abstractmethod
    async def add_event(self, id: str, object: Any, time: datetime) -> None:
        """Schedule a new event to be executed at the specified time.

        Args:
            id: Unique identifier for the event
            object: Object associated with the event
            time: When the event should be executed
        """
        pass

    @abstractmethod
    async def cancel_event(self, id: str) -> None:
        """Cancel a scheduled event.

        Args:
            id: Unique identifier of the event to cancel
        """
        pass
