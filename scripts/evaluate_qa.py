"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import click

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "packages/teams_memory"))

import logging

from teams_memory import (
    MemoryModuleConfig,
    StorageConfig,
    Topic,
)
from teams_memory.core.memory_core import MemoryCore
from teams_memory.interfaces.types import Memory, MemoryType
from teams_memory.services.llm_service import LLMService
from teams_memory.utils.logging import configure_logging

from scripts.utils.evaluation_utils import (
    BaseEvaluator,
    EvaluationResult,
    run_evaluation,
)
from tests.teams_memory.utils import build_llm_config

configure_logging(logging.DEBUG)

TEST_CASES: List[Dict[str, Any]] = [
    {
        "title": "Basic Question Answering",
        "setup": {
            "memories": [
                {
                    "content": "The user's favorite food is sushi, particularly salmon rolls",
                    "topics": ["Food Preferences"],
                },
                {
                    "content": "The user is allergic to shellfish and avoids all seafood except fish",
                    "topics": ["Health"],
                },
                {
                    "content": "The user enjoys cooking Japanese cuisine at home",
                    "topics": ["Hobbies"],
                },
                {
                    "content": "The user prefers spicy food and always adds extra hot sauce",
                    "topics": ["Food Preferences"],
                },
                {
                    "content": "The user has lactose intolerance and uses dairy alternatives",
                    "topics": ["Health"],
                },
            ],
            "topics": [
                Topic(
                    name="Food Preferences", description="Food preferences and likes"
                ),
                Topic(name="Health", description="Health-related information"),
                Topic(name="Hobbies", description="User's hobbies and interests"),
            ],
        },
        "questions": [
            {
                "question": "What is the user's favorite food?",
                "expected_answer_contains": ["sushi", "salmon rolls"],
                "topic": "Food Preferences",
                "relevant_memory_indices": [0],
            },
            {
                "question": "What are the user's food allergies and restrictions?",
                "expected_answer_contains": ["shellfish", "lactose intolerance"],
                "topic": "Health",
                "relevant_memory_indices": [1, 4],
            },
            {
                "question": "What kind of cuisine does the user cook?",
                "expected_answer_contains": ["Japanese"],
                "topic": "Hobbies",
                "relevant_memory_indices": [2],
            },
            {
                "question": "How does the user like their food prepared?",
                "expected_answer_contains": ["spicy", "hot sauce"],
                "topic": "Food Preferences",
                "relevant_memory_indices": [3],
            },
        ],
    },
    {
        "title": "Multi-Memory Answer Construction",
        "setup": {
            "memories": [
                {
                    "content": "The user works remotely as a software engineer",
                    "topics": ["Work"],
                },
                {
                    "content": "The user prefers to work from coffee shops in the morning",
                    "topics": ["Work Environment"],
                },
                {
                    "content": "The user uses a MacBook Pro for work",
                    "topics": ["Equipment"],
                },
                {
                    "content": "The user has a standing desk setup at home",
                    "topics": ["Equipment"],
                },
                {
                    "content": "The user takes regular breaks every 2 hours",
                    "topics": ["Work"],
                },
                {
                    "content": "The user has noise-cancelling headphones for focus",
                    "topics": ["Equipment"],
                },
            ],
            "topics": [
                Topic(name="Work", description="Work-related information"),
                Topic(
                    name="Work Environment", description="Where and how the user works"
                ),
                Topic(
                    name="Equipment", description="Tools and equipment the user uses"
                ),
            ],
        },
        "questions": [
            {
                "question": "Tell me about the user's work setup and environment",
                "expected_answer_contains": [
                    "software engineer",
                    "coffee shops",
                    "MacBook Pro",
                ],
                "topic": None,
                "relevant_memory_indices": [0, 1, 2],
            },
            {
                "question": "What equipment does the user use for work?",
                "expected_answer_contains": [
                    "MacBook Pro",
                    "standing desk",
                    "headphones",
                ],
                "topic": "Equipment",
                "relevant_memory_indices": [2, 3, 5],
            },
            {
                "question": "How does the user structure their work day?",
                "expected_answer_contains": [
                    "breaks every 2 hours",
                    "morning",
                    "coffee shops",
                ],
                "topic": None,
                "relevant_memory_indices": [1, 4],
            },
        ],
    },
    {
        "title": "Unknown Information Handling",
        "setup": {
            "memories": [
                {
                    "content": "The user has a dog named Max",
                    "topics": ["Pets"],
                },
                {
                    "content": "The user's dog is a golden retriever",
                    "topics": ["Pets"],
                },
            ],
            "topics": [
                Topic(name="Pets", description="Information about user's pets"),
                Topic(name="Family", description="Family-related information"),
                Topic(name="Education", description="Educational background"),
            ],
        },
        "questions": [
            {
                "question": "Does the user have any siblings?",
                "expected_answer_contains": None,
                "topic": "Family",
                "relevant_memory_indices": None,
            },
            {
                "question": "What kind of pets does the user have?",
                "expected_answer_contains": ["dog", "Max", "golden retriever"],
                "topic": "Pets",
                "relevant_memory_indices": [0, 1],
            },
            {
                "question": "Where did the user go to college?",
                "expected_answer_contains": None,
                "topic": "Education",
                "relevant_memory_indices": None,
            },
        ],
    },
    {
        "title": "Finding Relevant Memories from Noise",
        "setup": {
            "memories": [
                {
                    "content": "The user is learning to play piano",
                    "topics": ["Hobbies"],
                },
                {
                    "content": "The user drinks coffee black with no sugar",
                    "topics": ["Preferences"],
                },
                {
                    "content": "The user takes piano lessons every Tuesday",
                    "topics": ["Hobbies"],
                },
                {
                    "content": "The user prefers window seats on flights",
                    "topics": ["Preferences"],
                },
                {
                    "content": "The user has been playing piano for 6 months",
                    "topics": ["Hobbies"],
                },
                {
                    "content": "The user likes to read science fiction",
                    "topics": ["Hobbies"],
                },
                {
                    "content": "The user prefers texting over calling",
                    "topics": ["Preferences"],
                },
                {
                    "content": "The user enjoys hiking on weekends",
                    "topics": ["Hobbies"],
                },
                {
                    "content": "The user is allergic to cats",
                    "topics": ["Health"],
                },
                {
                    "content": "The user practices piano for 30 minutes daily",
                    "topics": ["Hobbies"],
                },
            ],
            "topics": [
                Topic(name="Hobbies", description="User's hobbies and interests"),
                Topic(name="Preferences", description="User's general preferences"),
                Topic(name="Health", description="Health-related information"),
            ],
        },
        "questions": [
            {
                "question": "Tell me about the user's piano playing",
                "expected_answer_contains": [
                    "6 months",
                    "lessons",
                    "Tuesday",
                    "30 minutes daily",
                ],
                "topic": "Hobbies",
                "relevant_memory_indices": [0, 2, 4, 9],
            },
            {
                "question": "How does the user take their coffee?",
                "expected_answer_contains": ["black", "no sugar"],
                "topic": "Preferences",
                "relevant_memory_indices": [1],
            },
            {
                "question": "What are the user's communication preferences?",
                "expected_answer_contains": ["texting", "calling"],
                "topic": "Preferences",
                "relevant_memory_indices": [6],
            },
        ],
    },
]


class QuestionAnsweringEvaluator(BaseEvaluator):
    def __init__(self, test_cases: List[dict[str, Any]]):
        super().__init__(test_cases)
        llm_config = build_llm_config()
        if not llm_config.api_key:
            print("Error: OpenAI API key not provided")
            sys.exit(1)

        self.llm_config = llm_config
        self.llm_service = LLMService(llm_config)

    async def evaluate_single(self, test_case: Dict) -> EvaluationResult:
        try:
            # Setup memory core with test configuration
            db_path = (
                Path(__file__).parent / "data" / "evaluation" / f"qa_test_{uuid4()}.db"
            )
            config = MemoryModuleConfig(
                storage=StorageConfig(db_path=db_path),
                buffer_size=5,
                timeout_seconds=60,
                llm=self.llm_config,
                topics=test_case["setup"]["topics"],
            )
            memory_core = MemoryCore(config, self.llm_service)

            # Store test memories
            user_id = f"test-user-{uuid4()}"
            memory_inputs = []
            for memory in test_case["setup"]["memories"]:
                memory_input = Memory(
                    id=str(uuid4()),
                    content=memory["content"],
                    created_at=datetime.now(),
                    user_id=user_id,
                    memory_type=MemoryType.SEMANTIC,
                    topics=memory["topics"],
                )
                memory_inputs.append(memory_input)

            # Test question answering for each query
            success = True
            failures = []
            all_answers = []

            for question_test in test_case["questions"]:
                answer_tuple = await memory_core._answer_question_from_memories(
                    memories=memory_inputs,
                    question=question_test["question"],
                )

                if question_test["expected_answer_contains"] is None:
                    # We expect no answer for this question
                    if answer_tuple is not None:
                        success = False
                        failures.append(
                            f"Question '{question_test['question']}' should have returned None but got an answer"
                        )
                else:
                    # We expect an answer containing specific phrases
                    if answer_tuple is None:
                        success = False
                        failures.append(
                            f"Question '{question_test['question']}' failed to get an answer"
                        )
                    else:
                        answer, returned_memories = answer_tuple
                        all_answers.append(
                            {
                                "question": question_test["question"],
                                "answer": answer,
                                "memories": [m.content for m in returned_memories],
                            }
                        )

                        # Check expected phrases in answer
                        for expected in question_test["expected_answer_contains"]:
                            if expected.lower() not in answer.lower():
                                success = False
                                failures.append(
                                    f"Answer to '{question_test['question']}' missing expected content '{expected}'"
                                )

                        # Check that returned memories match expected indices
                        expected_indices = question_test["relevant_memory_indices"]
                        if expected_indices is not None:
                            expected_memories_ids = set(
                                [memory_inputs[i].id for i in expected_indices]
                            )
                            returned_memory_set = set([m.id for m in returned_memories])
                            if expected_memories_ids != returned_memory_set:
                                success = False
                                missing_memories_ids = (
                                    expected_memories_ids - returned_memory_set
                                )
                                missing_memories = [
                                    next(
                                        (m.content for m in memory_inputs if m.id == id)
                                    )
                                    for id in missing_memories_ids
                                ]
                                extra_memories_ids = (
                                    returned_memory_set - expected_memories_ids
                                )
                                extra_memories = [
                                    next(
                                        (m.content for m in memory_inputs if m.id == id)
                                    )
                                    for id in extra_memories_ids
                                ]
                                failures.append(
                                    f"Answer to '{question_test['question']}' returned unexpected memories where the answer was '{answer}'. "  # noqa: E501
                                    f"Missing memories: {missing_memories}. "
                                    f"Extra memories: {extra_memories}. "
                                )

            return EvaluationResult(
                success=success,
                test_case=test_case,
                response={"answers": all_answers},
                failures=failures,
            )

        except Exception as e:
            logging.error(e)
            return EvaluationResult(False, test_case, None, [f"Error: {str(e)}"])


@click.command()
@click.option("--runs", default=1, help="Number of evaluation runs")
@click.option("--name", default="qa", help="Name for this evaluation run")
@click.option(
    "--output",
    default="evaluation_results_{name}.json",
    help="Output file for results. {name} will be replaced with the run name",
)
def main(runs: int, name: str, output: str) -> None:
    """Evaluate question answering capabilities."""
    output = output.format(name=name)
    evaluator = QuestionAnsweringEvaluator(TEST_CASES)
    asyncio.run(
        run_evaluation(
            evaluator=evaluator,
            num_runs=runs,
            output_file=output,
            description=f"Evaluating question answering ({name})",
        )
    )


if __name__ == "__main__":
    main()
