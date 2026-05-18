"""
agents/analyst.py
-----------------
The ANALYST is the second agent in the pipeline.

Its job:
    - Receive the raw research from the Researcher.
    - Extract key insights, spot patterns, and structure the findings.
    - Produce a clean, organized analytical summary.

The Analyst does NOT re-research. It works ONLY with what the Researcher provided.
This is the "inter-agent handoff" — the Analyst's entire input is the Researcher's output.

Why have a separate Analyst instead of just asking the Researcher to analyze?
    - Separation of concerns: each agent is laser-focused on ONE job.
    - Easier to debug: if analysis is wrong, you know exactly which agent to fix.
    - Mirrors real team structures (research team → analysis team → writing team).

Interview tip:
    The key design choice here is that state["research"] is passed as context.
    The Analyst has FULL visibility into what the Researcher found — this is
    what the resume bullet calls "full context sharing."
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState


# --- Prompt Template ---
# Notice we inject BOTH the original query AND the research.
# This keeps the analyst anchored to the user's intent, not just the raw research.

ANALYST_PROMPT = ChatPromptTemplate.from_template(
    """You are a Senior Data Analyst. Your job is to extract meaning from raw research.

Original Question: {query}

Raw Research Provided:
{research}

Instructions:
- Identify the 3-4 most important insights from the research above.
- Group related information into clear themes or categories.
- Highlight any notable patterns, contradictions, or gaps in the research.
- End with a one-paragraph "Key Takeaway" that captures the core finding.
- Do NOT write a final report yet — just structure and analyze.

Analytical Summary:"""
)


def analyst_agent(state: AgentState) -> AgentState:
    """
    The Analyst agent node.

    Args:
        state (AgentState): Contains 'query' (original question) and
                            'research' (output from the Researcher).

    Returns:
        AgentState: Same state with 'analysis' now filled in.

    How it works:
        1. Pull query + research from state.
        2. Inject both into the analyst prompt.
        3. LLM produces a structured analytical summary.
        4. Store in state["analysis"] and return.
    """

    print("\n[Analyst] Analyzing research findings...")

    # temperature=0.2 → very structured, analytical output (lower = more deterministic)
    llm = ChatOpenAI(api_key=os.getenv('KEY'),
                     base_url="https://api.deepseek.com/v1",
                     model="deepseek-chat",
                     temperature=0.2)

    formatted_prompt = ANALYST_PROMPT.format_messages(
        query=state["query"],
        research=state["research"],  # ← This is the handoff from the Researcher
    )

    response = llm.invoke(formatted_prompt)

    print("[Analyst] Analysis complete.")

    return {
        **state,
        "analysis": response.content,
    }
