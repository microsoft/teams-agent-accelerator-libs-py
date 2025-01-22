import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import click

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "packages/memory_module"))
from memory_module import (
    AssistantMessageInput,
    MemoryModuleConfig,
    StorageConfig,
    Topic,
    UserMessageInput,
)
from memory_module.core.memory_core import MemoryCore
from memory_module.services.llm_service import LLMService

from scripts.utils.evaluation_utils import (
    BaseEvaluator,
    EvaluationResult,
    run_evaluation,
)
from tests.memory_module.utils import build_llm_config

TEST_CASES = [
    {
        "title": "Device Information",
        "input": {
            "topics": [
                Topic(
                    name="Device Type", description="The type of device the user has"
                ),
                Topic(
                    name="Operating System", description="The user's operating system"
                ),
                Topic(name="Device Year", description="The year of the user's device"),
            ],
            "messages": [
                UserMessageInput(
                    id=str(uuid4()),
                    content="I need help with my device...",
                    author_id="user-123",
                    conversation_ref="conversation-123",
                    created_at=datetime.now() - timedelta(minutes=10),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="I'm sorry to hear that. What device do you have?",
                    author_id="assistant",
                    conversation_ref="conversation-123",
                    created_at=datetime.now() - timedelta(minutes=9),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="I have a gaming PC and a MacBook Pro",
                    author_id="user-123",
                    conversation_ref="conversation-123",
                    created_at=datetime.now() - timedelta(minutes=8),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="Which one are you having issues with? And what operating systems are they running?",
                    author_id="assistant",
                    conversation_ref="conversation-123",
                    created_at=datetime.now() - timedelta(minutes=7),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="The PC with Windows 11 is having problems",
                    author_id="user-123",
                    conversation_ref="conversation-123",
                    created_at=datetime.now() - timedelta(minutes=6),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="My MacBook is from 2023 and runs macOS Sonoma, but it's working fine",
                    author_id="user-123",
                    conversation_ref="conversation-123",
                    created_at=datetime.now() - timedelta(minutes=5),
                ),
            ],
        },
        "criteria": {
            "must_contain": ["macbook", "windows", "2023"],
            "must_not_contain": [],
            "topics_match": [
                {"phrase": "macbook", "topic": "Device Type"},
                {"phrase": "windows 11", "topic": "Operating System"},
                {"phrase": "macOS Sonoma", "topic": "Operating System"},
                {"phrase": "2023", "topic": "Device Year"},
            ],
        },
    },
    {
        "title": "Travel Preferences",
        "input": {
            "topics": [
                Topic(
                    name="Preferred Mode of Travel",
                    description="How the user likes to travel",
                ),
                Topic(
                    name="Destination Type",
                    description="The type of destination the user prefers",
                ),
                Topic(
                    name="Frequent Travel Companions",
                    description="People the user often travels with",
                ),
            ],
            "messages": [
                UserMessageInput(
                    id=str(uuid4()),
                    content="I'm planning my next vacation and need some advice.",
                    author_id="user-456",
                    conversation_ref="conversation-456",
                    created_at=datetime.now() - timedelta(minutes=15),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="I'd be happy to help! What kind of vacation are you looking for?",
                    author_id="assistant",
                    conversation_ref="conversation-456",
                    created_at=datetime.now() - timedelta(minutes=14),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="Well, I really enjoy beach destinations. Last time I went with my siblings and we had a blast!",  # noqa E501
                    author_id="user-456",
                    conversation_ref="conversation-456",
                    created_at=datetime.now() - timedelta(minutes=13),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="That sounds fun! How do you usually like to travel to your destinations?",
                    author_id="assistant",
                    conversation_ref="conversation-456",
                    created_at=datetime.now() - timedelta(minutes=12),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="When I'm in Europe, I love taking the train. It's so scenic and relaxing.",
                    author_id="user-456",
                    conversation_ref="conversation-456",
                    created_at=datetime.now() - timedelta(minutes=11),
                ),
            ],
        },
        "criteria": {
            "must_contain": ["train", "beach", "siblings"],
            "must_not_contain": [],
            "topics_match": [
                {"phrase": "train", "topic": "Preferred Mode of Travel"},
                {"phrase": "beach", "topic": "Destination Type"},
                {"phrase": "siblings", "topic": "Frequent Travel Companions"},
            ],
        },
    },
    {
        "title": "Health Habits",
        "input": {
            "topics": [
                Topic(
                    name="Exercise Routine",
                    description="The user's regular exercise habits",
                ),
                Topic(
                    name="Dietary Preferences",
                    description="The type of diet the user follows",
                ),
                Topic(
                    name="Health Goals",
                    description="Goals related to health and fitness",
                ),
            ],
            "messages": [
                UserMessageInput(
                    id=str(uuid4()),
                    content="I need advice on my fitness journey",
                    author_id="user-789",
                    conversation_ref="conversation-789",
                    created_at=datetime.now() - timedelta(minutes=20),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="I'd be happy to help! What are your current exercise habits?",
                    author_id="assistant",
                    conversation_ref="conversation-789",
                    created_at=datetime.now() - timedelta(minutes=19),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="I run 5 miles every morning, but I feel like I need to do more",
                    author_id="user-789",
                    conversation_ref="conversation-789",
                    created_at=datetime.now() - timedelta(minutes=18),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="That's already impressive! What's your ultimate fitness goal?",
                    author_id="assistant",
                    conversation_ref="conversation-789",
                    created_at=datetime.now() - timedelta(minutes=17),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="I'm training for a marathon! Also trying to clean up my diet - I'm following a vegetarian diet and avoiding dairy",  # noqa E501
                    author_id="user-789",
                    conversation_ref="conversation-789",
                    created_at=datetime.now() - timedelta(minutes=16),
                ),
            ],
        },
        "criteria": {
            "must_contain": ["run", "vegetarian", "marathon"],
            "must_not_contain": [],
            "topics_match": [
                {"phrase": "run", "topic": "Exercise Routine"},
                {"phrase": "vegetarian", "topic": "Dietary Preferences"},
                {"phrase": "marathon", "topic": "Health Goals"},
            ],
        },
    },
    {
        "title": "Work Preferences",
        "input": {
            "topics": [
                Topic(
                    name="Work Environment",
                    description="The user's preferred work environment",
                ),
                Topic(
                    name="Primary Work Tool",
                    description="The tool the user uses most for work",
                ),
                Topic(name="Work Hours", description="The user's working schedule"),
            ],
            "messages": [
                UserMessageInput(
                    id=str(uuid4()),
                    content="I'm struggling with productivity in my new remote work setup",
                    author_id="user-234",
                    conversation_ref="conversation-234",
                    created_at=datetime.now() - timedelta(minutes=25),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="Let's figure this out. Where do you usually work from?",
                    author_id="assistant",
                    conversation_ref="conversation-234",
                    created_at=datetime.now() - timedelta(minutes=24),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="I prefer quiet places like libraries - the silence helps me focus",
                    author_id="user-234",
                    conversation_ref="conversation-234",
                    created_at=datetime.now() - timedelta(minutes=23),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="What tools do you use most frequently for your work?",
                    author_id="assistant",
                    conversation_ref="conversation-234",
                    created_at=datetime.now() - timedelta(minutes=22),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="I do all my writing in Google Docs. I work standard hours, 9 AM to 5 PM on weekdays",
                    author_id="user-234",
                    conversation_ref="conversation-234",
                    created_at=datetime.now() - timedelta(minutes=21),
                ),
            ],
        },
        "criteria": {
            "must_contain": ["quiet", "Google Docs", "9 AM to 5 PM"],
            "must_not_contain": [],
            "topics_match": [
                {"phrase": "quiet", "topic": "Work Environment"},
                {"phrase": "Google Docs", "topic": "Primary Work Tool"},
                {"phrase": "9 AM to 5 PM", "topic": "Work Hours"},
            ],
        },
    },
    {
        "title": "Hobbies and Interests",
        "input": {
            "topics": [
                Topic(
                    name="Artistic Hobby",
                    description="Art-related activities the user enjoys",
                ),
                Topic(
                    name="Outdoor Activity",
                    description="The user's favorite outdoor activities",
                ),
                Topic(
                    name="Reading Preference",
                    description="The genres of books the user likes",
                ),
            ],
            "messages": [
                UserMessageInput(
                    id=str(uuid4()),
                    content="I'm looking for new hobbies to try during weekends",
                    author_id="user-567",
                    conversation_ref="conversation-567",
                    created_at=datetime.now() - timedelta(minutes=30),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="That's great! What kind of activities do you currently enjoy?",
                    author_id="assistant",
                    conversation_ref="conversation-567",
                    created_at=datetime.now() - timedelta(minutes=29),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="Well, I love painting landscapes when I have free time",
                    author_id="user-567",
                    conversation_ref="conversation-567",
                    created_at=datetime.now() - timedelta(minutes=28),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="Do you enjoy any outdoor activities or reading as well?",
                    author_id="assistant",
                    conversation_ref="conversation-567",
                    created_at=datetime.now() - timedelta(minutes=27),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="Yes! I go hiking in the mountains whenever I can. And I'm currently reading a great historical fiction novel",  # noqa E501
                    author_id="user-567",
                    conversation_ref="conversation-567",
                    created_at=datetime.now() - timedelta(minutes=26),
                ),
            ],
        },
        "criteria": {
            "must_contain": ["painting", "hiking", "historical fiction"],
            "must_not_contain": [],
            "topics_match": [
                {"phrase": "painting", "topic": "Artistic Hobby"},
                {"phrase": "hiking", "topic": "Outdoor Activity"},
                {"phrase": "historical fiction", "topic": "Reading Preference"},
            ],
        },
    },
    {
        "title": "Food Preferences",
        "input": {
            "topics": [
                Topic(
                    name="Favorite Cuisine",
                    description="The type of cuisine the user likes",
                ),
                Topic(
                    name="Diet Restrictions",
                    description="Dietary restrictions the user follows",
                ),
                Topic(
                    name="Frequent Snacks",
                    description="The snacks the user enjoys most often",
                ),
            ],
            "messages": [
                UserMessageInput(
                    id=str(uuid4()),
                    content="I need recommendations for a new restaurant in town",
                    author_id="user-890",
                    conversation_ref="conversation-890",
                    created_at=datetime.now() - timedelta(minutes=35),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="I'd love to help! What kind of cuisine do you prefer?",
                    author_id="assistant",
                    conversation_ref="conversation-890",
                    created_at=datetime.now() - timedelta(minutes=34),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="Italian food is my absolute favorite, especially pasta dishes",
                    author_id="user-890",
                    conversation_ref="conversation-890",
                    created_at=datetime.now() - timedelta(minutes=33),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="Great choice! Are there any dietary restrictions I should keep in mind?",
                    author_id="assistant",
                    conversation_ref="conversation-890",
                    created_at=datetime.now() - timedelta(minutes=32),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="Yes, I'm lactose intolerant, so I have to be careful with dairy",
                    author_id="user-890",
                    conversation_ref="conversation-890",
                    created_at=datetime.now() - timedelta(minutes=31),
                ),
                AssistantMessageInput(
                    id=str(uuid4()),
                    content="What do you usually snack on between meals?",
                    author_id="assistant",
                    conversation_ref="conversation-890",
                    created_at=datetime.now() - timedelta(minutes=30),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="I always keep chips and guacamole around - they're my go-to snacks",
                    author_id="user-890",
                    conversation_ref="conversation-890",
                    created_at=datetime.now() - timedelta(minutes=29),
                ),
            ],
        },
        "criteria": {
            "must_contain": ["Italian", "lactose intolerant", "Chips and guacamole"],
            "must_not_contain": [],
            "topics_match": [
                {"phrase": "Italian", "topic": "Favorite Cuisine"},
                {"phrase": "lactose intolerant", "topic": "Diet Restrictions"},
                {"phrase": "Chips and guacamole", "topic": "Frequent Snacks"},
            ],
        },
    },
    {
        "title": "Device Information with Topic Retrieval",
        "input": {
            "topics": [
                Topic(
                    name="Device Type", description="The type of device the user has"
                ),
                Topic(
                    name="Operating System", description="The user's operating system"
                ),
                Topic(name="Device Year", description="The year of the user's device"),
            ],
            "messages": [
                UserMessageInput(
                    id=str(uuid4()),
                    content="I use Windows 11 on my PC",
                    author_id="user-123",
                    conversation_ref="conversation-123",
                    created_at=datetime.now() - timedelta(minutes=5),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="I have a MacBook Pro from 2023",
                    author_id="user-123",
                    conversation_ref="conversation-123",
                    created_at=datetime.now() - timedelta(minutes=3),
                ),
                UserMessageInput(
                    id=str(uuid4()),
                    content="My MacBook runs macOS Sonoma",
                    author_id="user-123",
                    conversation_ref="conversation-123",
                    created_at=datetime.now() - timedelta(minutes=1),
                ),
            ],
        },
        "criteria": {
            "must_contain": ["Windows 11", "MacBook Pro", "macOS Sonoma", "2023"],
            "must_not_contain": [],
            "topics_match": [
                {"phrase": "Windows 11", "topic": "Operating System"},
                {"phrase": "macOS Sonoma", "topic": "Operating System"},
                {"phrase": "MacBook Pro", "topic": "Device Type"},
                {"phrase": "2023", "topic": "Device Year"},
            ],
        },
    },
]


class SystemPromptEvaluator(BaseEvaluator):
    def __init__(self, test_cases: List[Dict]):
        super().__init__(test_cases)
        llm_config = build_llm_config()
        if not llm_config.api_key:
            print("Error: OpenAI API key not provided")
            sys.exit(1)

        self.llm_config = llm_config
        self.llm_service = LLMService(llm_config)

        print(f"LLM Config: {self.llm_config}")

    async def evaluate_single(self, test_case: Dict) -> EvaluationResult:
        try:
            db_path = Path(__file__).parent / "data" / "evaluation" / "memory_module.db"
            config = MemoryModuleConfig(
                storage=StorageConfig(db_path=db_path),
                buffer_size=5,
                timeout_seconds=60,
                llm=self.llm_config,
                topics=test_case["input"]["topics"],
                enable_logging=True,
            )
            memory_core = MemoryCore(config, self.llm_service)
            response = await memory_core._extract_semantic_fact_from_messages(
                messages=test_case["input"]["messages"]
            )
            print(f"Response: {response}")

            criteria = test_case["criteria"]
            success = True
            failures = []

            # Check must_contain criteria
            for phrase in criteria.get("must_contain", []):
                # check if the phrase is in any of the extracted facts
                if not any(
                    phrase.lower() in fact.text.lower() for fact in response.facts
                ):
                    success = False
                    failures.append(f"Missing required phrase: {phrase}")

            # Check must_not_contain criteria
            for phrase in criteria.get("must_not_contain", []):
                if any(phrase.lower() in fact.text.lower() for fact in response.facts):
                    success = False
                    failures.append(f"Contains forbidden phrase: {phrase}")

            for topic_match in criteria.get("topics_match", []):
                facts_with_phrase = [
                    fact
                    for fact in response.facts
                    if topic_match["phrase"].lower() in fact.text.lower()
                ]
                if not facts_with_phrase:
                    success = False
                    failures.append(f"Missing topic: {topic_match['topic']}")

                # topics containing phrase
                topics_for_facts = [
                    topic for fact in facts_with_phrase for topic in fact.topics
                ]
                if topic_match["topic"] not in topics_for_facts:
                    success = False
                    failures.append(f"Missing topic: {topic_match['topic']}")

            return EvaluationResult(success, test_case, response.model_dump(), failures)

        except Exception as e:
            return EvaluationResult(False, test_case, None, [f"Error: {str(e)}"])


@click.command()
@click.option("--runs", default=1, help="Number of evaluation runs")
@click.option("--name", default="default", help="Name for this evaluation run")
@click.option(
    "--output",
    default="evaluation_results_{name}.json",
    help="Output file for results. {name} will be replaced with the run name",
    type=click.Path(dir_okay=False),
)
def main(runs: int, name: str, output: str) -> None:
    """Evaluate system prompt responses."""
    # Format the output filename with the run name
    output = output.format(name=name)

    evaluator = SystemPromptEvaluator(TEST_CASES)
    asyncio.run(
        run_evaluation(
            evaluator=evaluator,
            num_runs=runs,
            output_file=output,
            description=f"Evaluating system prompts ({name})",
        )
    )


if __name__ == "__main__":
    main()
