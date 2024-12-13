import json
from pydantic import BaseModel
from typing import List

class ExpectedMemory(BaseModel):
    query: str
    response: str

class SessionMessage(BaseModel):
    content: str
    role: str

class DatasetItem(BaseModel):
    category: str
    session: List[SessionMessage]
    query: str
    expected_strings_in_memories: List[str]

class Dataset(BaseModel):
    version: int
    title: str
    description: str
    data: List[DatasetItem]

def load_memory_module_dataset() -> Dataset:
    with open("evals/memory_module_dataset.json") as f:
        dataset = json.load(f)
        return Dataset.model_validate(dataset)