"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import logging


def configure_logging(logging_level: int = logging.DEBUG) -> None:
    module_name = __name__.split(".")[0]  # == "teams_memory"
    logger = logging.getLogger(module_name)
    logger.setLevel(logging_level)

    handler_name = "teams-memory-console"
    handler = next(
        (handler for handler in logger.handlers if handler.name == handler_name), None
    )
    # If handler already added, update logging level
    if handler:
        handler.setLevel(logging_level)
        return

    handler = logging.StreamHandler()
    handler.name = handler_name
    handler.setLevel(logging_level)

    formatter = DefaultFormatter(
        f"%(asctime)s:{module_name.upper()}:%(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)


class DefaultFormatter(logging.Formatter):
    # ANSI escape codes for colors
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{log_color}{record.msg}{self.RESET}"
        return super().format(record)
