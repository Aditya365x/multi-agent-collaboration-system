"""
graph/pipeline.py
-----------------
This file is the BRAIN of the system — it wires all agents together into a pipeline
using LangGraph's StateGraph.

What is a StateGraph?
    Think of it like a flowchart builder.
    - Nodes = agents (each one does a job)
    - Edges = the arrows between agents (defines execution order)
    - State  = the data that flows through every arrow

What is LangGraph?
    LangGraph is a library built on top of LangChain that lets you build
    multi-step, stateful workflows with LLMs. Instead of chaining prompts
    linearly (LangChain's default), LangGraph gives you a full graph —
    meaning you can have loops, branches, and conditional routing.

This file exposes ONE thing: `build_pipeline()`
    Call it to get a compiled, ready-to-run pipeline object.

Interview tip:
    The compile() call is what "locks in" the graph structure.
    After compilation, LangGraph validates the graph (no orphan nodes,
    proper entry/finish points) and optimizes it for execution.
"""

from langgraph.graph import StateGraph, END
from state import AgentState
from researcher import researcher_agent
from analyst import analyst_agent
from writer import writer_agent


def build_pipeline():
    """
    Builds and compiles the multi-agent LangGraph pipeline.

    The flow is:
        START → researcher → analyst → writer → END

    Each agent is a "node" in the graph. The edges define the order.

    Returns:
        A compiled LangGraph pipeline ready to be invoked with .invoke(state)
    """

    # --- Step 1: Create the graph with our shared state type ---
    # AgentState tells LangGraph what fields to expect and pass between nodes.
    graph = StateGraph(AgentState)

    # --- Step 2: Register each agent as a node ---
    # Syntax: graph.add_node("node_name", function_to_call)
    # The function must accept AgentState and return AgentState.
    graph.add_node("researcher", researcher_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("writer", writer_agent)

    # --- Step 3: Set the entry point ---
    # This tells LangGraph which node to run FIRST when .invoke() is called.
    graph.set_entry_point("researcher")

    # --- Step 4: Add edges (define execution order) ---
    # add_edge("from_node", "to_node") means:
    #   "After 'from_node' finishes, automatically run 'to_node' next."
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")

    # --- Step 5: Mark the finish point ---
    # END is a special LangGraph constant that signals the pipeline is done.
    graph.add_edge("writer", END)

    # --- Step 6: Compile and return ---
    # compile() validates the graph and prepares it for execution.
    compiled_pipeline = graph.compile()

    print("[Pipeline] Multi-agent pipeline built successfully.")
    return compiled_pipeline


# --- BONUS: Conditional routing example (not used in main flow, but shown for learning) ---
#
# In real systems, you might want to re-run the researcher if the results are thin.
# Here's how you'd add conditional routing:
#
# def should_retry_research(state: AgentState) -> str:
#     """Decide whether to retry research or move to analysis."""
#     if len(state["research"]) < 200:   # research is too short
#         return "researcher"             # loop back
#     return "analyst"                    # move forward
#
# graph.add_conditional_edges(
#     "researcher",            # from this node...
#     should_retry_research,   # ...run this function to decide where to go
#     {
#         "researcher": "researcher",  # if it returns "researcher", loop back
#         "analyst": "analyst",        # if it returns "analyst", move forward
#     }
# )
