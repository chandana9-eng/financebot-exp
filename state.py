"""
state.py — The shared memory of the entire agent system.

HUMAN ANALOGY:
Imagine a hospital patient file. Every doctor (agent) who sees the patient
reads from and writes to the SAME file. The file is always up to date.
No doctor has to call another doctor to ask "what happened before?"
They just read the file.

In LangGraph, State is that patient file. It flows through every node.
Every node can read everything and update what it handles.
"""

from typing import TypedDict, List, Optional, Annotated
import operator


class AgentState(TypedDict):
    # ── What the user said ────────────────────────────────────────────────────
    user_input: str                    # Original user message
    user_id: str                       # Who is talking (for memory)
    session_id: str                    # This conversation session

    # ── Routing decisions ─────────────────────────────────────────────────────
    intent: str                        # "finance", "entertainment", "orders", "unknown"
    subtasks: List[str]                # Supervisor breaks task into these

    # ── Agent results ─────────────────────────────────────────────────────────
    finance_result: str                # Output from finance agent
    entertainment_result: str          # Output from entertainment agent
    orders_result: str                 # Output from orders agent
    tool_calls_made: List[str]         # Log of every tool called (for debugging)

    # ── Memory ────────────────────────────────────────────────────────────────
    past_memories: List[str]           # Loaded from episodic memory at session start
    retrieved_context: str             # Retrieved from vector DB (RAG)

    # ── Conversation history ──────────────────────────────────────────────────
    messages: List[dict]               # Full message history for Claude calls

    # ── Safety ────────────────────────────────────────────────────────────────
    safety_passed: bool                # Did input guardrail pass?
    safety_reason: str                 # Why it failed (if it did)

    # ── Final output ──────────────────────────────────────────────────────────
    final_response: str                # What the user sees
    error: str                         # If something went wrong, what happened

    # ── Metadata for debugging + evals ───────────────────────────────────────
    iteration_count: int               # How many LangGraph iterations ran
    tokens_used: int                   # Total tokens across all Claude calls
    cost_usd: float                    # Total cost of this request


def initial_state(user_input: str, user_id: str = "user_001") -> AgentState:
    """
    Creates a fresh state for a new user request.
    Like opening a new blank patient file.
    """
    import uuid
    return {
        "user_input": user_input,
        "user_id": user_id,
        "session_id": str(uuid.uuid4())[:8],
        "intent": "",
        "subtasks": [],
        "finance_result": "",
        "entertainment_result": "",
        "orders_result": "",
        "tool_calls_made": [],
        "past_memories": [],
        "retrieved_context": "",
        "messages": [{"role": "user", "content": user_input}],
        "safety_passed": False,
        "safety_reason": "",
        "final_response": "",
        "error": "",
        "iteration_count": 0,
        "tokens_used": 0,
        "cost_usd": 0.0,
    }
