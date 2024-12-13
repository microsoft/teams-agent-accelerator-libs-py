from typing import List
import pandas as pd
from mlflow.metrics import MetricValue, make_metric
from mlflow.metrics.base import standard_aggregations

def check_string_in_retrieved_memories(expected_strings: List[str], retrieved_memories) -> bool:
    if retrieved_memories == "No memories":
        return False

    for s in expected_strings:
        for memory in retrieved_memories:
            if s.lower().strip() in memory.lower().strip():
                break
        else:
            return False

    return True

def build_ml_metric():
    def ml_metric(
        predictions: pd.Series,
        inputs: pd.Series,
        metrics: dict[str, MetricValue],
    ) -> MetricValue:
        scores: list[int] = []
        memories = predictions.apply(lambda x: x["memories"])
        for expected, actual in zip(inputs, memories):
            score = 1 if check_string_in_retrieved_memories(actual, expected) else 0
            scores.append(score)

        aggregated_results = standard_aggregations(scores)

        return MetricValue(
            scores=scores, aggregate_results=aggregated_results
        )

    return make_metric(eval_fn=ml_metric, greater_is_better=True, name="string check")
