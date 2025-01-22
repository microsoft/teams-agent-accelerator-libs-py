"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from typing import List, Optional

import pandas as pd
from mlflow.metrics import MetricValue, make_metric
from mlflow.metrics.base import standard_aggregations


def check_strings_in_retrieved_memories(
    expected_strings: List[str], actual_strings: Optional[List[str]]
) -> bool:
    """Check if all expected strings are present in the retrieved memories.

    For example:

    expected_strings = ["john", "world"]
    actual_strings = ["The user's name is John", "John thinks the world is a beautiful place"]

    The function should return True.

    Args:
        expected_strings (List[str]): The list of strings to check in the retrieved memories
        actual_strings (Optional[List[str]]): The strings to compare against the expected strings

    Returns:
        bool: True if all expected strings are present in atleast one of the actual strings, False otherwise
    """

    if not actual_strings:
        return False

    for s in expected_strings:
        for a in actual_strings:
            if s.lower().strip() in a.lower().strip():
                break
        else:
            return False

    return True


def string_check_metric():
    def ml_metric(
        predictions: pd.Series,
        inputs: pd.Series,
        metrics: dict[str, MetricValue],
    ) -> MetricValue:
        scores: list[int] = []
        memories = predictions.apply(lambda x: x["memories"])
        for expected, actual in zip(inputs, memories, strict=False):
            score = 1 if check_strings_in_retrieved_memories(expected, actual) else 0
            scores.append(score)

        aggregated_results = standard_aggregations(scores)

        return MetricValue(scores=scores, aggregate_results=aggregated_results)

    return make_metric(eval_fn=ml_metric, greater_is_better=True, name="string check")
