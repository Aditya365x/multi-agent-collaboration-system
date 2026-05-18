# Multi-Agent Collaboration System

> A production-style AI pipeline where three specialized LLM agents — **Researcher**, **Analyst**, and **Writer** — collaborate to generate structured, publication-ready research reports.

Built with **LangGraph · LangChain · FastAPI · OpenAI**

---

## What This Project Does

Most LLM apps send a single prompt and get a single response. This project takes a different approach — it **splits the work across three specialized AI agents**, each with a distinct role, and passes their outputs forward through a shared state.

**The result:** given any research query, the system produces a polished, multi-section report — complete with background research, structured analysis, and a written summary — without any human re-prompting between steps.

---

## Why Multi-Agent?

| Approach | Single Agent | Multi-Agent (This Project) |
|---|---|---|
| How it works | One LLM does everything in one prompt | Three LLMs, each focused on one task |
| Output quality | Shallow — the model spreads attention | Deep — each agent goes all-in on its job |
| Debuggability | Hard — you can't isolate what went wrong | Easy — check each agent's output individually |
| Scalability | Hard to extend without breaking things | Add a new agent node without touching others |

This mirrors how real teams work: a researcher gathers data, an analyst structures it, a writer communicates it. The pipeline just automates the handoffs.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              LangGraph StateGraph            │
│                                             │
│  ┌────────────┐   state   ┌─────────────┐  │
│  │ Researcher │ ────────► │   Analyst   │  │
│  │   Agent    │           │   Agent     │  │
│  └────────────┘           └──────┬──────┘  │
│                                  │ state    │
│                           ┌──────▼──────┐  │
│                           │   Writer    │  │
│                           │   Agent     │  │
│                           └──────┬──────┘  │
└──────────────────────────────────┼─────────┘
                                   │
                                   ▼
                            Final Report
                          (via FastAPI JSON)
```

The **shared state** (`AgentState`) is the baton passed between agents. Every agent reads from it and writes back to it — no agent calls another directly.

---

## Project Structure

```
multi_agent_system/
│
├── agents/                     # One file per agent
│   ├── __init__.py
│   ├── researcher.py           # Agent 1: Gathers raw information
│   ├── analyst.py              # Agent 2: Extracts insights and structure
│   └── writer.py               # Agent 3: Writes the final report
│
├── graph/                      # LangGraph wiring
│   ├── __init__.py
│   └── pipeline.py             # Connects agents into a StateGraph pipeline
│
├── api/                        # FastAPI HTTP layer
│   ├── __init__.py
│   └── app.py                  # POST /generate-report endpoint
│
├── utils/                      # Shared utilities
│   ├── __init__.py
│   └── state.py                # AgentState TypedDict (shared memory)
│
├── main.py                     # CLI entry point (run without API server)
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment variables
├── .gitignore                  # Files excluded from Git
└── README.md                   # This file
```

---

## Component Breakdown

### `utils/state.py` — Shared Memory

Defines `AgentState`, a Python TypedDict with four fields:

```python
class AgentState(TypedDict):
    query: str          # Original user question
    research: str       # Researcher's output
    analysis: str       # Analyst's output
    final_report: str   # Writer's output
```

**Why this matters:** LangGraph passes this object between every agent. Each agent reads what it needs, adds its own output, and returns the updated state. No agent needs to directly communicate with another — the state handles everything.

---

### `agents/researcher.py` — Researcher Agent

**Role:** First agent in the pipeline. Receives the raw user query and produces detailed, factual bullet points.

**Key design choices:**
- `temperature=0.3` — low creativity, high factuality
- Prompt instructs the model to *gather facts only*, not draw conclusions
- Output stored in `state["research"]`

**In a real system, this agent could:** call a web search tool (Tavily, Serper), query a vector database, or read uploaded documents.

---

### `agents/analyst.py` — Analyst Agent

**Role:** Second agent. Receives the Researcher's raw findings and extracts structured insights.

**Key design choices:**
- `temperature=0.2` — the most deterministic agent (analysis should be consistent)
- Has access to BOTH the original query AND the research — keeps it anchored to user intent
- Produces themed groupings + a "Key Takeaway" paragraph
- Output stored in `state["analysis"]`

**Interview talking point:** This agent never re-reads the source material. It works *only* from the Researcher's output. This simulates real inter-team handoffs where one team passes a deliverable to the next.

---

### `agents/writer.py` — Writer Agent

**Role:** Final agent. Has full context — query, research, and analysis — and synthesizes everything into a structured report.

**Key design choices:**
- `temperature=0.5` — slightly more creative for better writing quality
- Produces a fixed 4-section format: Executive Summary, Background, Insights, Conclusion
- Targets ~400-500 words — long enough to be useful, short enough to be readable
- Output stored in `state["final_report"]`

---

### `graph/pipeline.py` — LangGraph Pipeline

**Role:** Wires all three agents into a directed graph and compiles it.

```python
graph.set_entry_point("researcher")
graph.add_edge("researcher", "analyst")
graph.add_edge("analyst", "writer")
graph.add_edge("writer", END)
pipeline = graph.compile()
```

**What `compile()` does:**
- Validates the graph (no orphan nodes, proper start/end)
- Optimizes it for execution
- Returns a runnable object you can call with `.invoke(state)`

The file also includes a **commented-out conditional routing example** showing how to add loops (e.g., retry research if output is too short). This is the "conditional routing" mentioned in the architecture.

---

### `api/app.py` — FastAPI Endpoint

**Role:** Wraps the pipeline in an HTTP API so any client can call it.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Check if the API is running |
| `POST` | `/generate-report` | Run the full pipeline |

**Request body:**
```json
{
  "query": "What are the main applications of LLMs in healthcare?"
}
```

**Response:**
```json
{
  "query": "...",
  "research": "...",
  "analysis": "...",
  "final_report": "..."
}
```

FastAPI auto-generates interactive documentation at `http://localhost:8000/docs` — you can test the API directly from your browser.

---

### `main.py` — CLI Entry Point

A simple script to test the pipeline locally without running a server. Useful for debugging individual agents.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/multi-agent-system.git
cd multi-agent-system
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
# Open .env and add your OpenAI API key
```

### 5a. Run from the terminal (no server)

```bash
python main.py
```

### 5b. Run as an API server

```bash
uvicorn api.app:app --reload
```

Then open `http://localhost:8000/docs` to test it interactively.

---

## Example Output

**Query:** *"What are the main applications of LLMs in healthcare?"*

The pipeline produces:
1. **Research:** 6-7 bullet points on LLM use cases (clinical notes, diagnosis, drug discovery, etc.)
2. **Analysis:** 3-4 themed insights + a Key Takeaway paragraph
3. **Final Report:** A 400-500 word structured report with Executive Summary, Background, Insights, and Conclusion

---

## Key Concepts (Interview Reference)

| Concept | What It Means Here |
|---|---|
| **StateGraph** | LangGraph's graph builder — defines nodes (agents) and edges (flow) |
| **AgentState** | Shared TypedDict passed between all agents — the "baton" |
| **Nodes** | Individual agent functions registered in the graph |
| **Edges** | Directed connections that define execution order |
| **Conditional Edges** | Edges that use a function to decide the next node dynamically |
| **`compile()`** | Validates and finalizes the graph before execution |
| **Full Context Sharing** | Every agent has access to ALL previous agents' outputs via state |
| **Inter-Agent Handoff** | One agent's output becomes the next agent's input — no human re-prompting |

---

## Possible Extensions

- **Web Search Tool** — Give the Researcher access to real-time search (Tavily API)
- **RAG** — Let the Researcher query a vector database of documents
- **Streaming** — Use FastAPI `StreamingResponse` to stream the report as it's written
- **LangSmith Tracing** — One env variable gives you full agent-level trace logs
- **Retry Logic** — Add conditional routing to re-run the Researcher if output is too thin
- **Async Agents** — Run subtasks in parallel within an agent using `asyncio`

---

## Tech Stack

| Tool | Purpose |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Multi-agent state machine orchestration |
| [LangChain](https://github.com/langchain-ai/langchain) | LLM wrappers and prompt templates |
| [OpenAI GPT-4o-mini](https://platform.openai.com/) | Underlying language model |
| [FastAPI](https://fastapi.tiangolo.com/) | HTTP API framework |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server for FastAPI |

---

## License

MIT License — free to use, modify, and distribute.
