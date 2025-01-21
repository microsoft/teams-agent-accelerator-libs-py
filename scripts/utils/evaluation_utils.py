import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


class EvaluationResult:
    def __init__(
        self, success: bool, test_case: Dict, response: Any, failures: List[str]
    ):
        self.success = success
        self.test_case = test_case
        self.response = response
        self.failures = failures

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "test_case": self.test_case["title"],
            "response": self.response,
            "failures": self.failures,
        }


class BaseEvaluator:
    def __init__(self, test_cases: List[Dict]):
        self.test_cases = test_cases

    async def evaluate_single(self, test_case: Dict) -> EvaluationResult:
        raise NotImplementedError("Subclasses must implement evaluate_single")


async def run_evaluation(
    evaluator: BaseEvaluator,
    num_runs: int,
    output_file: str,
    description: str = "Processing test cases",
) -> None:
    all_results = []

    # Extract run name from output file
    run_name = Path(output_file).stem.replace("evaluation_results_", "")

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        run_results = []
        successes = 0
        failures = 0

        for test_case in tqdm(evaluator.test_cases, desc=description):
            result = await evaluator.evaluate_single(test_case)
            run_results.append(result.to_dict())

            if result.success:
                successes += 1
            else:
                failures += 1

        # Calculate statistics
        total = len(evaluator.test_cases)
        success_rate = (successes / total) * 100

        # Print summary for this run
        print_run_summary(run + 1, total, successes, failures)

        # Print failures if any
        if failures > 0:
            print_failures(run_results)

        all_results.append(
            {
                "run": run + 1,
                "name": run_name,
                "timestamp": datetime.now().isoformat(),
                "success_rate": success_rate,
                "results": run_results,
            }
        )

    save_results(all_results, output_file)
    print_final_report(all_results)


def print_run_summary(run_num: int, total: int, successes: int, failures: int) -> None:
    success_rate = (successes / total) * 100
    print(f"\n=== Run {run_num} Summary ===")
    print(f"Total test cases: {total}")
    print(f"Successes: {successes} ({success_rate:.1f}%)")
    print(f"Failures: {failures} ({100 - success_rate:.1f}%)")


def print_failures(results: List[Dict]) -> None:
    print("\n=== Failed Cases ===")
    for result in results:
        if not result["success"]:
            test_case = result["test_case"]
            print(f"\nTest Case: {test_case}")
            print(f"Failures: {result['failures']}")
            print(f"Response: {result['response']}")
            print("-" * 50)


def save_results(results: List[Dict], output_file: str) -> None:
    output_path = Path(output_file)
    with output_path.open("w") as f:
        json.dump(
            results, f, indent=2, default=lambda v: list(v) if isinstance(v, set) else v
        )
    print(f"\nResults saved to {output_path}")


def print_final_report(all_results: List[Dict]) -> None:
    """Print a final report summarizing results across all runs."""
    total_runs = len(all_results)
    total_test_cases = sum(len(run["results"]) for run in all_results)
    total_successes = sum(
        sum(1 for result in run["results"] if result["success"]) for run in all_results
    )
    total_failures = total_test_cases - total_successes

    # Get run name
    run_name = all_results[0]["name"]

    # Calculate average success rate
    avg_success_rate = sum(run["success_rate"] for run in all_results) / total_runs

    # Find best and worst runs
    best_run = max(all_results, key=lambda x: x["success_rate"])
    worst_run = min(all_results, key=lambda x: x["success_rate"])

    print("\n" + "=" * 50)
    print(f"FINAL EVALUATION REPORT - {run_name}")
    print("=" * 50)
    print(f"\nTotal Runs: {total_runs}")
    print(f"Total Test Cases: {total_test_cases}")
    print(f"Total Successes: {total_successes}")
    print(f"Total Failures: {total_failures}")
    print(f"Average Success Rate: {avg_success_rate:.1f}%")
    print(f"\nBest Run: Run {best_run['run']} ({best_run['success_rate']:.1f}%)")
    print(f"Worst Run: Run {worst_run['run']} ({worst_run['success_rate']:.1f}%)")

    # Most common failures if there are any failures
    if total_failures > 0:
        failure_counts = {}
        for run in all_results:
            for result in run["results"]:
                if not result["success"]:
                    test_case = result["test_case"]
                    failure_counts[test_case] = failure_counts.get(test_case, 0) + 1

        print("\nMost Common Failures:")
        for test_case, count in sorted(
            failure_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(
                f"- {test_case}: failed {count} times ({(count / total_runs) * 100:.1f}% of runs)"
            )

    print("\n" + "=" * 50)
