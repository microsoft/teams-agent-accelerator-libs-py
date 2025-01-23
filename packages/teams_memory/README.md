# Memory Module

## Logging

You can enable logging when setting up the memory module in the config.

```py
config = MemoryModuleConfig()
config.enable_logging=True,
```

### How does it work?

The `teams_memory` library uses
Python's [logging](https://docs.python.org/3.12/library/logging.html) library to facilitate logging. The `teams_memory` logger is configured to log debug messages (and higher serverity) to the console.

To set up the logger in your Python file, use the following code:

```py
import logging

logger = logging.getLogger(__name__)
```


This will create a logger named `teams_memory.<sub_module>.<file_name>`, which is a descendant of the `teams_memory` logger. All logged messages will be passed up to the handler assigned to the `teams_memory` logger.


### How to customize the logging behavior of the library?

Instead of setting `MemoryModuleConfig.enable_logging` to True, directly access the `teams_memory` logger like this:

```py
import logging

logger = logging.getLogger("teams_memory")
```

You can apply customizations to it. All loggers used in the library will be a descendant of it and so logs will be propagated to it.