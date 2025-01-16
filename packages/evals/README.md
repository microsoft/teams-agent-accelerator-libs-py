# Evals

The folder contains benchmarks for the different components

## Memory Module Benchmarking
How does the memory module work, and is it any good?


### Files

`benchmark_memory_module.py` - script to run the benchmark

`memory_module_dataset.json` - benchmark dataset

`helpers.py` - contains helper functions

`metrics.py` - evaluations metrics live here


### Dataset

This is the dataset. An individual test is an object in the `data` array.
```json
{
    "version": 1,
    "title": "Memory Module Benchmark",
    "description": "Each datapoint is session, i.e a conversation between the user and the assistant. And expected memories given the query.",
    "data": [
        {
            "category": "Customer Support Agent",
            "session": [
                { "role": "assistant", "content": "Hello! I'm your IT Support Assistant. How can I assist you today?" },
                { "role": "user", "content": "My computer is lagging really bad" },
                { "role": "assistant", "content": "To help troubleshoot your computer's lagging issue, I need a bit more information:\n\nWhat is the operating system of your device (e.g., Windows, macOS, Linux)?\nWhat type of device are you using (e.g., desktop, laptop)?\nWhat is the year of your device?\nCould you please provide these details?" },
                { "role": "user", "content": "I'm using a Dell laptop. Windows 11 is the operating system. And I got it back in 2019." }
            ],
            "query": "The user's facing computer lagging issues. I need to know the operating system of the device. What type of device. And also the year in which the device was used",
            "expected_strings_in_memories": ["Dell", "Windows 11", "2019"]
        }
    ]
}
```

Anatomy of a single test:

- `category` - simple description of what's being tested

- `session` - a conversation between the user and assistant that will be processed in batch by the semantic memory extraction component

- `query` - the query used to retrieve memories

- `expected_strings_in_memories` - each string in this list has to be present in at least one extracted memory


### Setup
0. Do `uv sync` in the root of the repository.

1. Install mlflow and start the mlflow server

```
mlflow server
```

2. Set the `EVAL_ENVIRONMENT` environment variable to `"local"`

3. Run the benchmark:

```
python packages/evals/benchmark_memory_module.py
```