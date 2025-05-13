class ProcessingError(Exception):
    """Error during processing of messages or memories."""

    pass


class MemoryNotFoundError(Exception):
    memory_id: str

    def __init__(self, memory_id: str):
        self.memory_id = memory_id
        super().__init__(f"Memory with id {memory_id} not found")


class InvalidUserError(Exception):
    user_id: str

    def __init__(self, user_id: str):
        self.user_id = user_id
        super().__init__(f"User with id {user_id} not found")


class MessageNotFoundError(Exception):
    """Error raised when a message is not found."""
    
    def __init__(self, message: str):
        super().__init__(message)
