"""
agents/writer.py
----------------
The WRITER is the final agent in the pipeline.

Its job:
    - Receive BOTH the raw research AND the structured analysis.
    - Synthesize everything into a polished, publication-ready report.
    - Write clearly, with proper structure (intro, body sections, conclusion).

The Writer sees the full picture:
    - query      → the user's original intent
    - research   → raw facts from the Researcher
    - analysis   → structured insights from the Analyst

This is what makes it better than a single-agent approach:
by the time the Writer runs, all the hard thinking is done.
The Writer only needs to focus on COMMUNICATION, not comprehension.

Interview tip:
    A single LLM asked to "research, analyze, and write about X" in one prompt
    often produces shallow output. By splitting roles, each agent can go deep
    in its specific task. This mirrors how real teams work.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState


# --- Prompt Template ---
# The Writer has access to ALL previous outputs — this is full context sharing.

WRITER_PROMPT = ChatPromptTemplate.from_template(
    """You are a Professional Report Writer. Your job is to write a clear, engaging report.

Original Question: {query}

Research Gathered:
{research}

Analytical Insights:
{analysis}

Instructions:
- Write a structured report with the following sections:
    1. Executive Summary (2-3 sentences, the TL;DR)
    2. Background & Key Facts (drawn from the research)
    3. Core Insights & Analysis (drawn from the analysis)
    4. Conclusion & Recommendations (your synthesis)
- Use professional but accessible language.
- The report should be self-contained — a reader who hasn't seen the research
  or analysis should be able to fully understand it.
- Aim for ~400-500 words.

Final Report:"""
)


def writer_agent(state: AgentState) -> AgentState:
    """
    The Writer agent node.

    Args:
        state (AgentState): Contains query, research, and analysis — the full pipeline output so far.

    Returns:
        AgentState: Same state with 'final_report' filled in. This is the pipeline's final output.

    How it works:
        1. Pull all three fields from state.
        2. Inject them into the writer prompt.
        3. LLM writes a structured, publication-ready report.
        4. Store in state["final_report"] and return.
    """

    print("\n[Writer] Writing final report...")

    # temperature=0.5 → more creative for writing quality (still grounded in the research)
    llm = ChatOpenAI(api_key=os.getenv('KEY'),
                     base_url="https://api.deepseek.com/v1",
                     model="deepseek-chat",
                     temperature=0.5)

    formatted_prompt = WRITER_PROMPT.format_messages(
        query=state["query"],
        research=state["research"],   # ← from Researcher
        analysis=state["analysis"],   # ← from Analyst
    )

    response = llm.invoke(formatted_prompt)

    print("[Writer] Report complete.")

    return {
        **state,
        "final_report": response.content,
    }
