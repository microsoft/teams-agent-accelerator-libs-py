import os

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
        print("Using Remote Databricks")
        mlflow.set_tracking_uri("databricks://memorymodule-evals")
        mlflow.set_experiment(experiment_name)
    autolog()
