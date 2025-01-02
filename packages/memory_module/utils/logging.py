import logging


def configure_logging():
    module_name = __name__.split(".")[0]  # == "memory_module"
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = DefaultFormatter(f"%(asctime)s:{module_name.upper()}:%(levelname)s - %(message)s")
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

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{log_color}{record.msg}{self.RESET}"
        return super().format(record)
