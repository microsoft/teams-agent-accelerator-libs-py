# Memory Server
FastAPI wrapper around the memory module.


## Setup Instructions

To set up and run the Memory Server, follow these steps:

1. Set up the dependencies by doing `uv sync` in the root folder.
1. Ensure the virtual environment activated in your terminal. See the root README.md file for details.
1. Configure the memory module in the `memory_service.py` file.
1. Navigate to this folder and run the following command:

```sh
uvicorn main:app
```

This will start the FastAPI server on port 8000.