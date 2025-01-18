import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import click

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "packages/memory_module"))

from memory_module import MemoryModuleConfig, RetrievalConfig, Topic
from memory_module.core.memory_core import MemoryCore
from memory_module.interfaces.types import BaseMemoryInput, MemoryType
from memory_module.services.llm_service import LLMService

from scripts.utils.evaluation_utils import (
    BaseEvaluator,
    EvaluationResult,
    run_evaluation,
)
from tests.memory_module.utils import build_llm_config

TEST_CASES = [
    {
        "title": "Basic Retrieval Test",
        "setup": {
            "memories": [
                {
                    "content": "The user enjoys hiking in Yosemite National Park every summer",
                    "topics": ["Outdoor Activity"],
                },
                {
                    "content": "The user prefers working from coffee shops in the morning",
                    "topics": ["Work Environment"],
                },
                {
                    "content": "The user is allergic to peanuts and avoids all nut products",
                    "topics": ["Health"],
                },
            ],
            "topics": [
                Topic(
                    name="Outdoor Activity",
                    description="Outdoor activities the user enjoys",
                ),
                Topic(
                    name="Work Environment",
                    description="Where the user prefers to work",
                ),
                Topic(name="Health", description="Health-related information"),
            ],
        },
        "queries": [
            {
                "query": "What are the user's outdoor activities?",
                "expected_memories": ["hiking in Yosemite"],
                "topic": "Outdoor Activity",
            },
            {
                "query": "Where does the user like to work?",
                "expected_memories": ["coffee shops"],
                "topic": "Work Environment",
            },
            {
                "query": "Tell me about allergies",
                "expected_memories": ["allergic to peanuts"],
                "topic": "Health",
            },
        ],
    },
    {
        "title": "Semantic Similarity Retrieval",
        "setup": {
            "memories": [
                {
                    "content": "The user practices yoga at sunrise in their home studio",
                    "topics": ["Exercise Routine"],
                },
                {
                    "content": "The user reads fantasy novels before bedtime",
                    "topics": ["Reading Preferences"],
                },
                {
                    "content": "The user is learning Spanish through online courses",
                    "topics": ["Education"],
                },
            ],
            "topics": [
                Topic(name="Exercise Routine", description="User's exercise habits"),
                Topic(
                    name="Reading Preferences",
                    description="What the user likes to read",
                ),
                Topic(
                    name="Education", description="Learning activities and preferences"
                ),
            ],
        },
        "queries": [
            {
                "query": "What kind of physical activities does the user do?",
                "expected_memories": ["yoga"],
                "topic": "Exercise Routine",
            },
            {
                "query": "What books does the user enjoy?",
                "expected_memories": ["fantasy novels"],
                "topic": "Reading Preferences",
            },
            {
                "query": "What languages is the user studying?",
                "expected_memories": ["Spanish"],
                "topic": "Education",
            },
        ],
    },
    {
        "title": "Operating System Topic with Semantic Search",
        "setup": {
            "memories": [
                {
                    "content": "I use Windows 11 on my gaming PC",
                    "topics": ["Operating System"],
                },
                {
                    "content": "I have a MacBook Pro from 2023",
                    "topics": ["Device Type"],
                },
                {
                    "content": "My MacBook runs macOS Sonoma",
                    "topics": ["Operating System"],
                },
            ],
            "topics": [
                Topic(
                    name="Operating System", description="The user's operating system"
                ),
                Topic(
                    name="Device Type", description="The type of device the user has"
                ),
                Topic(name="Device year", description="The year of the user's device"),
            ],
        },
        "queries": [
            {
                "query": "MacBook",
                "expected_memories": ["sonoma"],
                "topic": "Operating System",
            },
            {
                "query": "What operating system does the user use for their Windows PC?",
                "expected_memories": ["windows"],
                "topic": "Operating System",
            },
            {
                "query": "What kind of computer does the user have?",
                "expected_memories": ["MacBook Pro"],
                "topic": "Device Type",
            },
        ],
    },
    {
        "title": "Keyword-Based Retrieval",
        "setup": {
            "memories": [
                {
                    "content": "The user enjoys playing basketball every weekend at the local gym",
                    "topics": ["Sports"],
                },
                {
                    "content": "The user has a golden retriever named Max",
                    "topics": ["Pets"],
                },
                {
                    "content": "The user plays piano and guitar in a local band",
                    "topics": ["Hobbies"],
                },
                {
                    "content": "The user's golden retriever goes to dog training every Tuesday",
                    "topics": ["Pets"],
                },
            ],
            "topics": [
                Topic(name="Sports", description="Sports and physical activities"),
                Topic(name="Pets", description="Information about user's pets"),
                Topic(name="Hobbies", description="User's hobbies and interests"),
            ],
        },
        "queries": [
            {
                "query": "basketball",
                "expected_memories": ["basketball"],
                "topic": "Sports",
            },
            {
                "query": "golden retriever",
                "expected_memories": ["golden retriever named Max", "dog training"],
                "topic": "Pets",
            },
            {
                "query": "piano",
                "expected_memories": ["piano"],
                "topic": "Hobbies",
            },
            {
                "query": "Max",
                "expected_memories": ["golden retriever named Max"],
                "topic": "Pets",
            },
        ],
    },
    {
        "title": "Question-Based Retrieval",
        "setup": {
            "memories": [
                {
                    "content": "The user's favorite color is blue, particularly navy blue",
                    "topics": ["Preferences"],
                },
                {
                    "content": "The user visits their grandmother every Sunday for dinner",
                    "topics": ["Family Routine"],
                },
                {
                    "content": "The user takes their coffee with oat milk and no sugar",
                    "topics": ["Food Preferences"],
                },
            ],
            "topics": [
                Topic(name="Preferences", description="User's general preferences"),
                Topic(name="Family Routine", description="Regular family activities"),
                Topic(
                    name="Food Preferences", description="Food and drink preferences"
                ),
            ],
        },
        "queries": [
            {
                "query": "What is the user's favorite color?",
                "expected_memories": ["favorite color is blue"],
                "topic": "Preferences",
            },
            {
                "query": "How does the user take their coffee?",
                "expected_memories": ["oat milk and no sugar"],
                "topic": "Food Preferences",
            },
            {
                "query": "When does the user see their grandmother?",
                "expected_memories": ["visits their grandmother every Sunday"],
                "topic": "Family Routine",
            },
        ],
    },
    {
        "title": "Statement-Based Retrieval",
        "setup": {
            "memories": [
                {
                    "content": "The user has been working as a software engineer for 5 years",
                    "topics": ["Career"],
                },
                {
                    "content": "The user completed their master's degree in Computer Science in 2020",
                    "topics": ["Education"],
                },
                {
                    "content": "The user volunteers at the local animal shelter on weekends",
                    "topics": ["Volunteering"],
                },
            ],
            "topics": [
                Topic(
                    name="Career",
                    description="Professional experience and work history",
                ),
                Topic(name="Education", description="Educational background"),
                Topic(name="Volunteering", description="Volunteer activities"),
            ],
        },
        "queries": [
            {
                "query": "Tell me about their work experience",
                "expected_memories": ["software engineer for 5 years"],
                "topic": "Career",
            },
            {
                "query": "Information about their education",
                "expected_memories": ["master's degree in Computer Science"],
                "topic": "Education",
            },
            {
                "query": "The user's volunteer activities",
                "expected_memories": ["animal shelter"],
                "topic": "Volunteering",
            },
        ],
    },
]


class RetrievalEvaluator(BaseEvaluator):
    def __init__(self, test_cases: List[Dict]):
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
                Path(__file__).parent
                / "data"
                / "evaluation"
                / f"retrieval_test_{uuid4()}.db"
            )
            config = MemoryModuleConfig(
                db_path=db_path,
                buffer_size=5,
                timeout_seconds=60,
                llm=self.llm_config,
                topics=test_case["setup"]["topics"],
            )
            memory_core = MemoryCore(config, self.llm_service)

            # Store test memories
            user_id = f"test-user-{uuid4()}"
            for memory in test_case["setup"]["memories"]:
                memory_input = BaseMemoryInput(
                    content=memory["content"],
                    created_at=datetime.now(),
                    user_id=user_id,
                    memory_type=MemoryType.SEMANTIC,
                    topics=memory["topics"],
                )
                metadata = await memory_core._extract_metadata_from_fact(
                    memory["content"],
                    [
                        topic
                        for topic in config.topics
                        if topic.name in memory["topics"]
                    ],
                )
                embed_vectors = await memory_core._get_semantic_fact_embeddings(
                    memory["content"], metadata
                )
                await memory_core.storage.store_memory(
                    memory_input, embedding_vectors=embed_vectors
                )

            # Test retrieval for each query
            success = True
            failures = []

            for query_test in test_case["queries"]:
                topic = next(
                    (t for t in config.topics if t.name == query_test["topic"]), None
                )
                retrieved_memories = await memory_core.retrieve_memories(
                    user_id=user_id,
                    config=RetrievalConfig(
                        query=query_test["query"],
                        topic=topic,
                        limit=5,
                    ),
                )

                # Check if expected memories are found in retrieved results
                for expected in query_test["expected_memories"]:
                    if not any(
                        expected.lower() in memory.content.lower()
                        for memory in retrieved_memories
                    ):
                        success = False
                        failures.append(
                            f"Query '{query_test['query']}' failed to retrieve memory containing '{expected}'"  # noqa E501
                        )

            # test queries without passing in topic
            for query_test in test_case["queries"]:
                retrieved_memories = await memory_core.retrieve_memories(
                    user_id=user_id,
                    config=RetrievalConfig(
                        query=query_test["query"],
                        topic=None,
                        limit=5,
                    ),
                )

                # Check if expected memories are found in retrieved results
                for expected in query_test["expected_memories"]:
                    if not any(
                        expected.lower() in memory.content.lower()
                        for memory in retrieved_memories
                    ):
                        success = False
                        failures.append(
                            f"Query '{query_test['query']}' (without topic filtering) failed to retrieve memory containing '{expected}'"  # noqa E501
                        )

            return EvaluationResult(
                success=success,
                test_case=test_case,
                response={
                    "retrieved_memories": [m.content for m in retrieved_memories]
                },
                failures=failures,
            )

        except Exception as e:
            return EvaluationResult(False, test_case, None, [f"Error: {str(e)}"])


@click.command()
@click.option("--runs", default=1, help="Number of evaluation runs")
@click.option("--name", default="retrieval", help="Name for this evaluation run")
@click.option(
    "--output",
    default="evaluation_results_{name}.json",
    help="Output file for results. {name} will be replaced with the run name",
)
def main(runs: int, name: str, output: str) -> None:
    """Evaluate memory retrieval capabilities."""
    output = output.format(name=name)
    evaluator = RetrievalEvaluator(TEST_CASES)
    asyncio.run(
        run_evaluation(
            evaluator=evaluator,
            num_runs=runs,
            output_file=output,
            description=f"Evaluating memory retrieval ({name})",
        )
    )


if __name__ == "__main__":
    main()
