"""
api/app.py
----------
This file exposes the multi-agent pipeline as a REST API using FastAPI.

What is FastAPI?
    FastAPI is a modern Python web framework for building APIs.
    It's fast, easy to use, and auto-generates documentation at /docs.

What does this file do?
    - Defines a POST endpoint: /generate-report
    - Accepts a JSON body with a "query" field
    - Runs the full multi-agent pipeline (Researcher → Analyst → Writer)
    - Returns the final report as a JSON response

Why wrap it in an API?
    Without an API, your pipeline only runs from the terminal.
    With an API, any frontend, mobile app, or service can call it.
    This is what makes your ML project a real "system."

How to run:
    uvicorn app:app --reload
    Then visit: http://localhost:8000/docs

Interview tip:
    FastAPI uses Pydantic models (like QueryRequest below) to automatically
    validate incoming JSON. If the user sends missing or wrong-type fields,
    FastAPI rejects it with a clear error — no manual validation needed.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pipeline import build_pipeline


# --- App Initialization ---
# This creates the FastAPI application instance.
# The metadata here shows up in the auto-generated /docs page.
load_dotenv()
app = FastAPI(
    title="Multi-Agent Research Report System",
    description=(
        "A LangGraph-powered pipeline with three AI agents "
        "(Researcher, Analyst, Writer) that generates structured research reports."
    ),
    version="1.0.0",
)

# Allow requests from any origin (frontend dev server, file://, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Build the pipeline ONCE at startup ---
# We build it here (outside the endpoint) so it's not rebuilt on every request.
# This is more efficient — the graph structure doesn't change between requests.
pipeline = build_pipeline()


# --- Request Model ---
# Pydantic validates that incoming JSON has a "query" field that is a string.
# If it's missing or wrong type, FastAPI automatically returns a 422 error.
class QueryRequest(BaseModel):
    query: str  # The research topic or question from the user


# --- Response Model ---
# Defines the shape of the JSON we return.
# Pydantic ensures we always return consistent, well-typed responses.
class ReportResponse(BaseModel):
    query: str          # Echo back the original query
    research: str       # What the Researcher found
    analysis: str       # What the Analyst extracted
    final_report: str   # The Writer's final report


# --- Health Check Endpoint ---
# A simple endpoint to confirm the API is running.
# Good practice for production systems and deployment monitoring.
@app.get("/health")
def health_check():
    """Returns a simple status message to confirm the API is up."""
    return {"status": "ok", "message": "Multi-Agent System is running."}


# --- Serve Frontend ---
# Serve the built frontend at the root so the whole app lives on one port.
HERE = Path(__file__).parent


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


# --- Main Endpoint ---
@app.post("/generate-report", response_model=ReportResponse)
def generate_report(request: QueryRequest):
    """
    The main pipeline endpoint.

    Accepts a research query, runs it through all three agents, and returns
    the full pipeline output including research, analysis, and final report.

    Args:
        request (QueryRequest): JSON body with a "query" field.

    Returns:
        ReportResponse: JSON with all four pipeline fields.

    Raises:
        HTTPException 500: If the pipeline fails for any reason.
    """

    print(f"\n[API] Received query: '{request.query}'")

    try:
        # --- Invoke the pipeline ---
        # We pass in the INITIAL state. query is set, everything else is empty.
        # LangGraph will fill in research, analysis, and final_report as it runs.
        initial_state = {
            "query": request.query,
            "research": "",       # Will be filled by Researcher
            "analysis": "",       # Will be filled by Analyst
            "final_report": "",   # Will be filled by Writer
        }

        result = pipeline.invoke(initial_state)

        print(f"[API] Pipeline completed for query: '{request.query}'")

        # Return the full result as a structured JSON response
        return ReportResponse(
            query=result["query"],
            research=result["research"],
            analysis=result["analysis"],
            final_report=result["final_report"],
        )

    except Exception as e:
        # If anything goes wrong (API key missing, LLM error, etc.),
        # return a clear 500 error instead of crashing silently.
        print(f"[API] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
