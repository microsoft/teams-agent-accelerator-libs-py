"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import click

sys.path.append(Path(__file__).parent.parent)
sys.path.append(Path(__file__).parent.parent / "packages")

from teams_memory import (
    MemoryModule,
    MemoryModuleConfig,
    StorageConfig,
    UserMessageInput,
)

from scripts.utils.evaluation_utils import (
    BaseEvaluator,
    EvaluationResult,
    run_evaluation,
)
from tests.teams_memory.utils import build_llm_config

TEST_CASES = [
    {
        "title": "General vs. Specific Detail",
        "input": {
            "old_messages": [
                "I love outdoor activities.",
                "I often visit national parks.",
            ],
            "incoming_message": "I enjoy hiking in Yellowstone National Park.",
            "expected_decision": "ignore",
        },
        "reason": "The old messages already cover the new message's context.",
    },
    {
        "title": "Specific Detail vs. General",
        "input": {
            "old_messages": [
                "I really enjoy hiking in Yellowstone National Park.",
                "I like exploring scenic trails.",
            ],
            "incoming_message": "I enjoy hiking in national parks.",
            "expected_decision": "ignore",
        },
        "reason": "The new message is broader and redundant to the old messages.",
    },
    {
        "title": "Repeated Behavior Over Time",
        "input": {
            "old_messages": [
                "I had coffee at 8 AM yesterday.",
                "I had coffee at 8 AM this morning.",
            ],
            "incoming_message": "I had coffee at 8 AM again today.",
            "expected_decision": "add",
        },
        "reason": "This reinforces a recurring pattern of behavior over time.",
    },
    {
        "title": "Updated Temporal Context",
        "input": {
            "old_messages": [
                "I'm planning a trip to Japan.",
                "I've been looking at flights to Japan.",
            ],
            "incoming_message": "I just canceled my trip to Japan.",
            "expected_decision": "add",
        },
        "reason": "The new message reflects a significant update to the old messages.",
    },
    {
        "title": "Irrelevant or Unnecessary Update",
        "input": {
            "old_messages": [
                "I prefer tea over coffee.",
                "I usually drink tea every day.",
            ],
            "incoming_message": "I like tea.",
            "expected_decision": "ignore",
        },
        "reason": "The new message does not add any unique or relevant information.",
    },
    {
        "title": "Redundant Memory with Different Wording",
        "input": {
            "old_messages": [
                "I have an iPhone 12.",
                "I bought an iPhone 12 back in 2022.",
            ],
            "incoming_message": "I own an iPhone 12.",
            "expected_decision": "ignore",
        },
        "reason": "The new message is a rephrased duplicate of the old messages.",
    },
    {
        "title": "Additional Specific Information",
        "input": {
            "old_messages": [
                "I like playing video games.",
                "I often play games on my console.",
            ],
            "incoming_message": "I love playing RPG video games like Final Fantasy.",
            "expected_decision": "add",
        },
        "reason": "The new message adds specific details about the type of games.",
    },
    {
        "title": "Contradictory Information",
        "input": {
            "old_messages": ["I like cats.", "I have a cat named Whiskers."],
            "incoming_message": "Actually, I don't like cats.",
            "expected_decision": "add",
        },
        "reason": "The new message reflects a contradiction or change in preference.",
    },
    {
        "title": "New Memory Completely Unrelated",
        "input": {
            "old_messages": [
                "I love reading mystery novels.",
                "I'm a big fan of Agatha Christie's books.",
            ],
            "incoming_message": "I really enjoy playing soccer.",
            "expected_decision": "add",
        },
        "reason": "The new message introduces entirely new information.",
    },
    {
        "title": "Multiple Old Messages with Partial Overlap",
        "input": {
            "old_messages": ["I have a car.", "My car is a Toyota Camry."],
            "incoming_message": "I own a blue Toyota Camry.",
            "expected_decision": "add",
        },
        "reason": "The new message adds a specific detail (color) not covered by the old messages.",
    },
]


class MemoryDecisionEvaluator(BaseEvaluator):
    def __init__(self, test_cases: List[Dict]):
        super().__init__(test_cases)
        llm_config = build_llm_config()
        if not llm_config.api_key:
            print("Error: OpenAI API key not provided")
            sys.exit(1)

        self.llm_config = llm_config

    async def evaluate_single(self, test_case: Dict) -> EvaluationResult:
        try:
            db_path = Path(__file__).parent / "data" / "evaluation" / "teams_memory.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Delete existing db if it exists
            if db_path.exists():
                db_path.unlink()

            config = MemoryModuleConfig(
                storage=StorageConfig(db_path=db_path),
                buffer_size=5,
                timeout_seconds=60,
                llm=self.llm_config,
            )

            memory_module = MemoryModule(config=config)
            conversation_id = str(uuid4())

            # Add old messages
            for message_content in test_case["input"]["old_messages"]:
                message = UserMessageInput(
                    id=str(uuid4()),
                    content=message_content,
                    author_id="user-123",
                    conversation_ref=conversation_id,
                    created_at=datetime.now() - timedelta(days=1),
                )
                await memory_module.add_message(message)

            await memory_module.process_messages(conversation_id)

            # Create incoming message
            new_message = [
                UserMessageInput(
                    id=str(uuid4()),
                    content=test_case["input"]["incoming_message"],
                    author_id="user-123",
                    conversation_ref=conversation_id,
                    created_at=datetime.now(),
                )
            ]

            # Get the decision
            extraction = (
                await memory_module.memory_core._extract_semantic_fact_from_messages(
                    new_message
                )
            )
            if not (extraction.action == "add" and extraction.facts):
                return EvaluationResult(
                    success=False,
                    test_case=test_case,
                    response="failed_extraction",
                    failures=["Failed to extract semantic facts"],
                )

            failures = []
            success = True

            for fact in extraction.facts:
                decision = (
                    await memory_module.memory_core._get_add_memory_processing_decision(
                        fact.text, "user-123"
                    )
                )
                if decision.decision != test_case["input"]["expected_decision"]:
                    success = False
                    failures.append(
                        f"Expected {test_case['input']['expected_decision']}, got {decision.decision}"
                    )

            # Cleanup
            await memory_module.message_queue.message_buffer.scheduler.cleanup()

            return EvaluationResult(
                success=success,
                test_case=test_case,
                response={
                    "decision": decision.decision,
                    "reason": decision.reason_for_decision,
                },
                failures=failures,
            )

        except Exception as e:
            return EvaluationResult(False, test_case, None, [f"Error: {str(e)}"])


@click.command()
@click.option("--runs", default=1, help="Number of evaluation runs")
@click.option("--name", default="default", help="Name for this evaluation run")
@click.option(
    "--output",
    default="memory_decisions_{name}.json",
    help="Output file for results. {name} will be replaced with the run name",
    type=click.Path(dir_okay=False),
)
def main(runs: int, name: str, output: str) -> None:
    """Evaluate memory processing decisions."""
    # Format the output filename with the run name
    output = output.format(name=name)

    evaluator = MemoryDecisionEvaluator(TEST_CASES)
    asyncio.run(
        run_evaluation(
            evaluator=evaluator,
            num_runs=runs,
            output_file=output,
            description=f"Evaluating memory decisions ({name})",
        )
    )


if __name__ == "__main__":
    main()
