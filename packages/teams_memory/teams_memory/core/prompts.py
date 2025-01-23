# pylint: disable=E501

MEMORY_PROCESSING_DECISION_PROMPT = """You are a semantic memory management agent. Your task is to decide whether the new memory should be added to the memory system or ignored as a duplicate.

Considerations:
1.	Context Overlap:
If the new memory conveys information that is substantially covered by an existing memory, it should be ignored.
If the new memory adds unique or specific information not present in any old memory, it should be added.
2.	Granularity of Detail:
Broader or more general memories should not replace specific ones. However, a specific detail can replace a general statement if it conveys the same underlying idea.
For example:
Old memory: "The user enjoys hiking in national parks."
New memory: "The user enjoys hiking in Yellowstone National Park."
Result: Ignore (The older memory already encompasses the specific case).
3.	Repeated Patterns:
If the new memory reinforces a pattern of behavior over time (e.g., multiple mentions of a recurring habit, preference, or routine), it should be added to reflect this trend.
4.	Temporal Relevance:
If the new memory reflects a significant change or update to the old memory, it should be added.
For example:
Old memory: "The user is planning a trip to Japan."
New memory: "The user has canceled their trip to Japan."
Result: Add (The new memory reflects a change).

Process:
	1.	Compare the specificity, unique details, and time relevance of the new memory against old memories.
	2.	Decide whether to add or ignore based on the considerations above.
	3.	Provide a clear and concise justification for your decision.

Here are the old memories:
{old_memory_content}

Here is the new memory:
{new_memory} created at {created_at}"""  # noqa: E501

METADATA_EXTRACTION_PROMPT = """Your role is to rephrase the text in your own words and provide a summary of the text.
{topics_context}"""  # noqa: E501

SEMANTIC_FACT_EXTRACTION_PROMPT = """You are a semantic memory management agent. Your goal is to extract meaningful, facts and preferences from user messages. Focus on recognizing general patterns and interests that will remain relevant over time, even if the user is mentioning short-term plans or events.

<TOPICS>
{topics_str}
</TOPICS>

<EXISTING_MEMORIES>
{existing_memories_str}
</EXISTING_MEMORIES>"""  # noqa: E501

SEMANTIC_FACT_USER_PROMPT = """
<TRANSCRIPT>
{messages_str}
</TRANSCRIPT>

<INSTRUCTIONS>
Extract new FACTS from the user messages in the TRANSCRIPT.
FACTS are patterns and interests that will remain relevant over time, even if the user is mentioning short-term plans or events.
Treat FACTS independently, even if multiple facts relate to the same topic.
Ignore FACTS found in EXISTING_MEMORIES.
"""  # noqa: E501
