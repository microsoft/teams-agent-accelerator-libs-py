import json
import logging
import os
from pathlib import Path
from typing import List, TypedDict

import mlflow
from dotenv import load_dotenv
from mlflow.openai import autolog

load_dotenv()

def setup_mlflow(experiment_name: str):
    if not experiment_name.startswith("/"):
        experiment_name = f"/{experiment_name}"

    env = os.getenv("EVAL_ENVIRONMENT", "local")
    if env == "local":
        mlflow.set_tracking_uri("http://localhost:5000")
    else:
        logging.info("Using Remote Databricks")
        mlflow.set_tracking_uri("databricks://memorymodule-evals")
        mlflow.set_experiment(experiment_name)
    autolog()


class SessionMessage(TypedDict):
    content: str
    role: str


class DatasetItem(TypedDict):
    category: str
    session: List[SessionMessage]
    query: str
    expected_strings_in_memories: List[str]


class Dataset(TypedDict):
    version: int
    title: str
    description: str
    data: List[DatasetItem]


def load_dataset() -> Dataset:
    with open(Path(__file__) / ".." / "memory_module_dataset.json") as f:
        dataset = json.load(f)
        return dataset
