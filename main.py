"""
main.py
-------
A simple command-line entry point to test the pipeline WITHOUT starting the API server.

Use this when:
    - You want to quickly test if the pipeline works end-to-end
    - You're debugging a specific agent's output
    - You don't need the HTTP layer

How to run:
    python main.py

What it does:
    1. Builds the multi-agent pipeline
    2. Runs it with a hardcoded test query
    3. Prints each agent's output to the terminal

This is useful during development — you can see exactly what each agent
produces before worrying about the API layer.
"""

from dotenv import load_dotenv

from pipeline import build_pipeline


def run_pipeline(query: str):
    """
    Runs the full pipeline and prints all outputs.

    Args:
        query (str): The research topic to run through the pipeline.
    """

    print("=" * 60)
    print("MULTI-AGENT RESEARCH REPORT SYSTEM")
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    # Build the compiled pipeline
    pipeline = build_pipeline()

    # Set the initial state — only 'query' is filled; others start empty
    initial_state = {
        "query": query,
        "research": "",
        "analysis": "",
        "final_report": "",
    }

    # Run the full pipeline
    # LangGraph automatically runs: researcher → analyst → writer
    result = pipeline.invoke(initial_state)

    # --- Print each agent's output ---
    print("\n" + "=" * 60)
    print("RESEARCHER OUTPUT")
    print("=" * 60)
    print(result["research"])

    print("\n" + "=" * 60)
    print("ANALYST OUTPUT")
    print("=" * 60)
    print(result["analysis"])

    print("\n" + "=" * 60)
    print("FINAL REPORT (Writer Output)")
    print("=" * 60)
    print(result["final_report"])

    return result


if __name__ == "__main__":
    load_dotenv()
    # Change this query to test with different topics
    test_query = "What are the main applications of Large Language Models in healthcare?"
    run_pipeline(test_query)
