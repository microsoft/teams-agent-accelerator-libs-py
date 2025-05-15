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
    SQLiteStorageConfig,
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
                "required_memory_indices": [0],
                "irrelevant_memory_indices": [2, 3],
            },
            {
                "question": "What are the user's food allergies and restrictions?",
                "expected_answer_contains": ["shellfish", "lactose intolerance"],
                "topic": "Health",
                "required_memory_indices": [1, 4],
                "irrelevant_memory_indices": [0, 2, 3],
            },
            {
                "question": "What kind of cuisine does the user cook?",
                "expected_answer_contains": ["Japanese"],
                "topic": "Hobbies",
                "required_memory_indices": [2],
                "irrelevant_memory_indices": [0, 1, 4],
            },
            {
                "question": "How does the user like their food prepared?",
                "expected_answer_contains": ["spicy", "hot sauce"],
                "topic": "Food Preferences",
                "required_memory_indices": [3],
                "irrelevant_memory_indices": [1, 2],
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
                {
                    "content": "The user has a dairly allergy",
                    "topics": ["Health"],
                },
                {
                    "content": "The user has a young daughter",
                    "topics": ["Family"],
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
                "required_memory_indices": [0, 1, 2],
                "irrelevant_memory_indices": [6, 7],
            },
            {
                "question": "What equipment does the user use for during work?",
                "expected_answer_contains": [
                    "MacBook Pro",
                    "standing desk",
                    "headphones",
                ],
                "topic": "Equipment",
                "required_memory_indices": [2, 3, 5],
                "irrelevant_memory_indices": [0, 1],
            },
            {
                "question": "How does the user structure their work day?",
                "expected_answer_contains": [
                    "breaks every 2 hours",
                    "morning",
                    "coffee shops",
                ],
                "topic": None,
                "required_memory_indices": [1, 4],
                "irrelevant_memory_indices": [2, 3, 5],
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
                "required_memory_indices": None,
                "irrelevant_memory_indices": [0, 1],
            },
            {
                "question": "Provide some details about the user's pets",
                "expected_answer_contains": ["dog", "Max", "golden retriever"],
                "topic": "Pets",
                "required_memory_indices": [0, 1],
                "irrelevant_memory_indices": [],
            },
            {
                "question": "Where did the user go to college?",
                "expected_answer_contains": None,
                "topic": "Education",
                "required_memory_indices": None,
                "irrelevant_memory_indices": [0, 1],
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
                "required_memory_indices": [0, 2, 4, 9],
                "irrelevant_memory_indices": [1, 3, 6, 7, 8],
            },
            {
                "question": "How does the user take their coffee?",
                "expected_answer_contains": ["black", "no sugar"],
                "topic": "Preferences",
                "required_memory_indices": [1],
                "irrelevant_memory_indices": [0, 2, 4, 5, 6, 7, 8, 9],
            },
            {
                "question": "What are the user's communication preferences?",
                "expected_answer_contains": ["texting", "calling"],
                "topic": "Preferences",
                "required_memory_indices": [6],
                "irrelevant_memory_indices": [1, 3],
            },
        ],
    },
    {
        "title": "Chronological Memory Processing",
        "setup": {
            "memories": [
                {
                    "content": "The user works at Microsoft",
                    "topics": ["Work"],
                    "created_at": "2021-01-01T10:00:00Z",
                },
                {
                    "content": "The user works at Amazon",
                    "topics": ["Work"],
                    "created_at": "2022-01-02T10:00:00Z",
                },
                {
                    "content": "The user works at Google",
                    "topics": ["Work"],
                    "created_at": "2023-01-03T10:00:00Z",
                },
                {
                    "content": "The user enjoys programming",  # noise memory
                    "topics": ["Work"],
                    "created_at": "2024-03-02T10:00:00Z",
                },
            ],
            "topics": [
                Topic(name="Work", description="Work-related information"),
            ],
        },
        "questions": [
            {
                "question": "Where does the user currently work?",
                "expected_answer_contains": ["Google"],
                "topic": "Work",
                "required_memory_indices": [2],
                "irrelevant_memory_indices": [3],
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

    async def evaluate_single(self, test_case: Dict) -> List[EvaluationResult]:
        try:
            # Setup memory core with test configuration
            db_path = (
                Path(__file__).parent / "data" / "evaluation" / f"qa_test_{uuid4()}.db"
            )
            config = MemoryModuleConfig(
                storage=SQLiteStorageConfig(db_path=db_path),
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
                    created_at=(
                        datetime.strptime(memory["created_at"], "%Y-%m-%dT%H:%M:%SZ")
                        if "created_at" in memory
                        else datetime.now()
                    ),
                    user_id=user_id,
                    memory_type=MemoryType.SEMANTIC,
                    topics=memory["topics"],
                )
                memory_inputs.append(memory_input)

            # Test question answering for each query
            evaluation_results = []

            for question_test in test_case["questions"]:
                question_success = True
                question_failures = []
                answer_response = None

                answer_tuple = await memory_core._answer_question_from_memories(
                    memories=memory_inputs,
                    question=question_test["question"],
                )

                if question_test["expected_answer_contains"] is None:
                    # We expect no answer for this question
                    if answer_tuple is not None:
                        question_success = False
                        question_failures.append(
                            "Question should have returned None but got an answer"
                        )
                else:
                    # We expect an answer containing specific phrases
                    if answer_tuple is None:
                        question_success = False
                        question_failures.append("Failed to get an answer")
                    else:
                        answer, returned_memories = answer_tuple
                        answer_response = {
                            "answer": answer,
                            "memories": [m.content for m in returned_memories],
                        }

                        # Check expected phrases in answer
                        for expected in question_test["expected_answer_contains"]:
                            if expected.lower() not in answer.lower():
                                question_success = False
                                question_failures.append(
                                    f"Missing expected content '{expected}'. The answer was: '{answer}'"
                                )

                        # Check that returned memories include required and exclude irrelevant
                        required_indices = question_test.get("required_memory_indices")
                        irrelevant_indices = question_test.get(
                            "irrelevant_memory_indices", []
                        )

                        if required_indices is not None:
                            returned_memory_ids = set([m.id for m in returned_memories])
                            required_memory_ids = set(
                                [memory_inputs[i].id for i in required_indices]
                            )
                            irrelevant_memory_ids = set(
                                [memory_inputs[i].id for i in irrelevant_indices]
                            )

                            # Check for missing required memories
                            missing_memory_ids = (
                                required_memory_ids - returned_memory_ids
                            )
                            if missing_memory_ids:
                                question_success = False
                                missing_memories = [
                                    next(
                                        (m.content for m in memory_inputs if m.id == id)
                                    )
                                    for id in missing_memory_ids
                                ]
                                question_failures.append(
                                    f"Missing required memories: {missing_memories}"
                                )

                            # Check for incorrectly included irrelevant memories
                            included_irrelevant = (
                                returned_memory_ids & irrelevant_memory_ids
                            )
                            if included_irrelevant:
                                question_success = False
                                irrelevant_included = [
                                    next(
                                        (m.content for m in memory_inputs if m.id == id)
                                    )
                                    for id in included_irrelevant
                                ]
                                question_failures.append(
                                    f"Incorrectly included irrelevant memories: {irrelevant_included}"
                                )

                # Create a test case specific to this question
                question_test_case = {
                    "title": f"{test_case['title']} - {question_test['question']}",
                    "setup": test_case["setup"],
                    "questions": [question_test],
                }

                evaluation_results.append(
                    EvaluationResult(
                        success=question_success,
                        test_case=question_test_case,
                        response=answer_response,
                        failures=question_failures,
                    )
                )

            return evaluation_results

        except Exception as e:
            logging.error(e)
            # Return a failed result for each question in case of error
            return [
                EvaluationResult(
                    False,
                    {
                        "title": f"{test_case['title']} - {q['question']}",
                        "setup": test_case["setup"],
                        "questions": [q],
                    },
                    None,
                    [f"Error: {str(e)}"],
                )
                for q in test_case["questions"]
            ]


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
