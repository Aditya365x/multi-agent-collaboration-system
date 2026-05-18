"""
agents/researcher.py
--------------------
The RESEARCHER is the first agent in the pipeline.

Its job:
    - Receive the user's query.
    - Simulate (or actually perform) research on that topic.
    - Return factual bullet points and key information.

In a real system, this agent could:
    - Call a web search tool (e.g., Tavily, Serper API)
    - Query a vector database (RAG)
    - Read uploaded documents

For now, it uses an LLM with a carefully written prompt to simulate research.

Interview tip:
    This agent is stateless — it doesn't remember anything between runs.
    All context it needs comes from the AgentState passed into it.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState


# --- Prompt Template ---
# This tells the LLM exactly what role it is playing and what to produce.
# Using a template (instead of a hardcoded string) lets us inject dynamic values
# like the user's query at runtime.

RESEARCHER_PROMPT = ChatPromptTemplate.from_template(
    """You are a Research Specialist. Your job is to gather thorough, factual information.

Topic to research: {query}

Instructions:
- Provide 5-7 key facts or findings about this topic.
- Include relevant statistics, definitions, and background context.
- Be factual and detailed. Do NOT draw conclusions yet — just gather information.
- Format your response as clear bullet points.

Research Output:"""
)


def researcher_agent(state: AgentState) -> AgentState:
    """
    The Researcher agent node.

    Args:
        state (AgentState): The current pipeline state. We read 'query' from it.

    Returns:
        AgentState: The same state, but with 'research' now filled in.

    How it works:
        1. Pull the user's query from state.
        2. Format the prompt by injecting the query.
        3. Send the prompt to the LLM and get a response.
        4. Store the response in state["research"] and return updated state.
    """

    print("\n[Researcher] Starting research...")

    # Initialize the LLM
    # temperature=0.3 → slightly creative but mostly factual (0 = deterministic, 1 = very creative)
    llm = ChatOpenAI(api_key=os.getenv('KEY'),
                     base_url="https://api.deepseek.com/v1",
                     model="deepseek-chat",
                     temperature=0.3)

    # Format the prompt with the actual query from state
    formatted_prompt = RESEARCHER_PROMPT.format_messages(query=state["query"])

    # Call the LLM
    response = llm.invoke(formatted_prompt)

    print("[Researcher] Research complete.")

    # Return updated state — always return the FULL state, just with new fields filled
    return {
        **state,  # spread existing state so we don't lose any fields
        "research": response.content,
    }
