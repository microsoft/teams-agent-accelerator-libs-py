import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import uuid4

from memory_module import MemoryModule, MemoryModuleConfig, UserMessageInput
from memory_module.core.scheduler import Scheduler
from tqdm import tqdm

from tests.memory_module.utils import build_llm_config

# Test cases from before
TEST_CASES = [
    {
        "title": "General vs. Specific Detail",
        "old_messages": ["I love outdoor activities.", "I often visit national parks."],
        "incoming_message": "I enjoy hiking in Yellowstone National Park.",
        "expected_decision": "ignore",
        "reason": "The old messages already cover the new message’s context.",
    },
    {
        "title": "Specific Detail vs. General",
        "old_messages": ["I really enjoy hiking in Yellowstone National Park.", "I like exploring scenic trails."],
        "incoming_message": "I enjoy hiking in national parks.",
        "expected_decision": "ignore",
        "reason": "The new message is broader and redundant to the old messages.",
    },
    {
        "title": "Repeated Behavior Over Time",
        "old_messages": ["I had coffee at 8 AM yesterday.", "I had coffee at 8 AM this morning."],
        "incoming_message": "I had coffee at 8 AM again today.",
        "expected_decision": "add",
        "reason": "This reinforces a recurring pattern of behavior over time.",
    },
    {
        "title": "Updated Temporal Context",
        "old_messages": ["I’m planning a trip to Japan.", "I’ve been looking at flights to Japan."],
        "incoming_message": "I just canceled my trip to Japan.",
        "expected_decision": "add",
        "reason": "The new message reflects a significant update to the old messages.",
    },
    {
        "title": "Irrelevant or Unnecessary Update",
        "old_messages": ["I prefer tea over coffee.", "I usually drink tea every day."],
        "incoming_message": "I like tea.",
        "expected_decision": "ignore",
        "reason": "The new message does not add any unique or relevant information.",
    },
    {
        "title": "Redundant Memory with Different Wording",
        "old_messages": ["I have an iPhone 12.", "I bought an iPhone 12 back in 2022."],
        "incoming_message": "I own an iPhone 12.",
        "expected_decision": "ignore",
        "reason": "The new message is a rephrased duplicate of the old messages.",
    },
    {
        "title": "Additional Specific Information",
        "old_messages": ["I like playing video games.", "I often play games on my console."],
        "incoming_message": "I love playing RPG video games like Final Fantasy.",
        "expected_decision": "add",
        "reason": "The new message adds specific details about the type of games.",
    },
    {
        "title": "Contradictory Information",
        "old_messages": ["I like cats.", "I have a cat named Whiskers."],
        "incoming_message": "Actually, I don’t like cats.",
        "expected_decision": "add",
        "reason": "The new message reflects a contradiction or change in preference.",
    },
    {
        "title": "New Memory Completely Unrelated",
        "old_messages": ["I love reading mystery novels.", "I’m a big fan of Agatha Christie’s books."],
        "incoming_message": "I really enjoy playing soccer.",
        "expected_decision": "add",
        "reason": "The new message introduces entirely new information.",
    },
    {
        "title": "Multiple Old Messages with Partial Overlap",
        "old_messages": ["I have a car.", "My car is a Toyota Camry."],
        "incoming_message": "I own a blue Toyota Camry.",
        "expected_decision": "add",
        "reason": "The new message adds a specific detail (color) not covered by the old messages.",
    },
]


async def evaluate_decision(memory_module, test_case):
    """Evaluate a single decision test case."""
    conversation_id = str(uuid4())

    # Add old messages
    for message_content in test_case["old_messages"]:
        message = UserMessageInput(
            id=str(uuid4()),
            content=message_content,
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now() - timedelta(days=1),
        )
        await memory_module.add_message(message)

    await memory_module.message_queue.message_buffer.scheduler.flush()

    # Create incoming message
    new_message = [
        UserMessageInput(
            id=str(uuid4()),
            content=test_case["incoming_message"],
            author_id="user-123",
            conversation_ref=conversation_id,
            created_at=datetime.now(),
        )
    ]

    # Get the decision
    extraction = await memory_module.memory_core._extract_semantic_fact_from_messages(new_message)
    if not (extraction.action == "add" and extraction.facts):
        return {
            "success": False,
            "error": "Failed to extract semantic facts",
            "test_case": test_case,
            "expected": test_case["expected_decision"],
            "got": "failed_extraction",
            "reason": "Failed to extract semantic facts",
        }

    for fact in extraction.facts:
        decision = await memory_module.memory_core._get_add_memory_processing_decision(fact.text, "user-123")
        return {
            "success": decision.decision == test_case["expected_decision"],
            "expected": test_case["expected_decision"],
            "got": decision.decision,
            "reason": decision.reason_for_decision,
            "test_case": test_case,
        }


async def main():
    # Initialize config and memory module
    llm_config = build_llm_config()
    if not llm_config.api_key:
        print("Error: OpenAI API key not provided")
        sys.exit(1)

    db_path = Path(__file__).parent / "data" / "evaluation" / "memory_module.db"
    # Create db directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = MemoryModuleConfig(
        db_path=db_path,
        buffer_size=5,
        timeout_seconds=60,
        llm=llm_config,
    )

    # Delete existing db if it exists
    if db_path.exists():
        db_path.unlink()

    memory_module = MemoryModule(config=config)

    results = []
    successes = 0
    failures = 0

    # Run evaluations with progress bar
    print("\nEvaluating memory processing decisions...")
    for test_case in tqdm(TEST_CASES, desc="Processing test cases"):
        result = await evaluate_decision(memory_module, test_case)
        results.append(result)
        if result["success"]:
            successes += 1
        else:
            failures += 1

    # Calculate statistics
    total = len(TEST_CASES)
    success_rate = (successes / total) * 100

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Total test cases: {total}")
    print(f"Successes: {successes} ({success_rate:.1f}%)")
    print(f"Failures: {failures} ({100 - success_rate:.1f}%)")

    # Print detailed failures if any
    if failures > 0:
        print("\n=== Failed Cases ===")
        for result in results:
            if not result["success"]:
                test_case = result["test_case"]
                print(f"\nTest Case: {test_case['title']}")
                print(f"Reason: {test_case['reason']}")
                print(f"Actual result: {result['reason']}")
                print(f"Expected: {result['expected']}")
                print(f"Got: {result['got']}")
                print("Old messages:")
                for msg in test_case["old_messages"]:
                    print(f"  - {msg}")
                print(f"New message: {test_case['incoming_message']}")
                print("-" * 50)

    # Cleanup
    await cast(Scheduler, memory_module.message_queue.message_buffer.scheduler).cleanup()


if __name__ == "__main__":
    asyncio.run(main())
