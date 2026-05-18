"""
utils/state.py
--------------
This file defines the "shared memory" of the entire pipeline.

Think of AgentState as a baton in a relay race.
Each agent (Researcher → Analyst → Writer) receives this baton,
reads what the previous agent wrote, adds its own output, and passes it forward.

Why TypedDict?
    - It's just a Python dictionary, but with type hints.
    - LangGraph requires this format to track state across nodes.
    - It gives us autocomplete and makes debugging much easier.
"""

from typing import TypedDict


class AgentState(TypedDict):
    """
    The single shared state object that flows through all agents.

    Fields:
        query        : The original question/topic from the user.
        research     : Raw information gathered by the Researcher agent.
        analysis     : Structured insights extracted by the Analyst agent.
        final_report : The polished, publication-ready report from the Writer agent.
    """

    query: str
    research: str
    analysis: str
    final_report: str
