import asyncio
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import click
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../packages"))

from memory_module.config import LLMConfig, MemoryModuleConfig
from memory_module.core.memory_module import MemoryModule, Message

from evals.helpers import Dataset, DatasetItem, load_dataset, setup_mlflow
from evals.metrics import string_check_metric

setup_mlflow(experiment_name="memory_module")


class MemoryModuleManager:
    def __init__(self, buffer_size=5):
        self._buffer_size = buffer_size
        self._memory_module: MemoryModule = None
        self._db_path = Path(__file__).parent / "data" / f"memory_{uuid.uuid4().hex}.db"

    def __enter__(self):
        # Create memory module
        llm = LLMConfig(
            model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        config = MemoryModuleConfig(db_path=self._db_path, buffer_size=self._buffer_size, llm=llm)

        self._memory_module = MemoryModule(config=config)
        return self._memory_module

    def __exit__(self, exc_type, exc_value, traceback):
        # Destroy memory module and database
        del self._memory_module
        os.remove(self._db_path)


async def add_messages(memory_module: MemoryModule, messages: List[dict]):
    def create_message(**kwargs):
        return Message(
            id=str(uuid.uuid4()),
            content=kwargs["content"],
            is_assistant_message=kwargs["is_assistant_message"],
            # author_id="user" if not kwargs["is_assistant_message"] else None,
            author_id="user",
            created_at=datetime.now(),
            conversation_ref="conversation_ref",
        )

    for message in messages:
        is_assistant_message = message["role"] == "assistant"
        msg = create_message(content=message["content"], is_assistant_message=is_assistant_message)
        await memory_module.add_message(msg)


def run_benchmark(
    name: str,
    dataset: Dataset,
    run_one: bool,
):
    if not name:
        name = "memory module benchmark"

    if run_one:
        dataset.data = dataset["data"][0:1]

    # prepare dataset
    inputs = dataset["data"]
    df = pd.DataFrame({"inputs": inputs})
    dataset_name = f"{dataset['title']} v{dataset['description']}"
    dataset = mlflow.data.pandas_dataset.from_pandas(df, name=dataset_name)

    # benchmark function
    async def benchmark_memory_module(input: DatasetItem):
        session = input["session"]
        query = input["query"]
        expected_strings_in_memories = input["expected_strings_in_memories"]

        # buffer size has to be the same as the session length to trigger sm processing
        with MemoryModuleManager(buffer_size=len(session)) as memory_module:
            await add_messages(memory_module, messages=session)
            memories = await memory_module.retrieve_memories(query, user_id=None, limit=None)

        return {
            "input": {
                "session": session,
                "query": query,
                "expected_strings_in_memories": expected_strings_in_memories,
            },
            "output": "No memories" if len(memories) == 0 else [memory.content for memory in memories],
        }

    # iterate over benchmark cases
    def iterate_benchmark_cases(inputs: pd.Series):
        results = []
        for row in tqdm(inputs.itertuples(), total=inputs.size):
            results.append(asyncio.run(benchmark_memory_module(row.inputs)))

        return pd.DataFrame(
            {
                "predictions": [{"memories": result["output"]} for result in results],
            }
        )

    # run benchmark
    mlflow_metric = string_check_metric()
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"dataset": dataset_name})
        mlflow.evaluate(iterate_benchmark_cases, dataset, extra_metrics=[mlflow_metric])


@click.command()
@click.option("--name", type=str, required=False, help="Name of the benchmark")
@click.option("--run_one", type=bool, default=False, help="Run only one benchmark case")
def main(name, run_one):
    dataset = load_dataset()
    run_benchmark(name, dataset, run_one)


if __name__ == "__main__":
    main()
