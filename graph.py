"""
graph.py — Assembles all nodes into the LangGraph workflow.

HUMAN ANALOGY — THE COMPLETE PICTURE:
You are looking at the wiring diagram of FinanceBot Pro.
Every box is a room. Every arrow is a hallway.
The user enters at Reception (supervisor).
Reception sends them to the right room.
They get their answer in that room.
The summary writer (synthesizer) writes their discharge note.
They exit with their answer.

THE FLOW:
User Input
    → [safety_gate] — checks if input is safe
        → BLOCKED: return error immediately
        → SAFE: → [memory_loader] — loads past memories
            → [supervisor] — decides which agents to call
                → [finance_agent] — if finance question
                → [entertainment_agent] — if entertainment question
                → [orders_agent] — if order question
                → [synthesizer] — always last — composes final response
    → Final Response to User
"""

from langgraph.graph import StateGraph, END
from state import AgentState
from agents.agents import (
    supervisor_node, route_from_supervisor,
    finance_agent_node, entertainment_agent_node,
    orders_agent_node, synthesizer_node
)
from guardrails.guardrails import check_input, check_output, get_safe_fallback
from memory.memory_store import load_episodic_memories, seed_knowledge_base


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER NODES (guardrails + memory as graph nodes)
# ─────────────────────────────────────────────────────────────────────────────

def safety_gate_node(state: AgentState) -> dict:
    """
    Input guardrail as a LangGraph node.
    If it fails, sets safety_passed=False and a reason.
    The conditional edge after this node routes to END if failed.
    """
    print(f"\n[Safety Gate] Checking input")
    is_safe, reason = check_input(state["user_input"])

    if not is_safe:
        print(f"[Safety Gate] BLOCKED: {reason}")
        return {
            "safety_passed": False,
            "safety_reason": reason,
            "final_response": "I'm unable to process that request. Please rephrase your question."
        }

    print(f"[Safety Gate] PASSED")
    return {"safety_passed": True, "safety_reason": "passed"}


def memory_loader_node(state: AgentState) -> dict:
    """
    Load episodic memories at the start of every session.
    Runs after safety gate, before supervisor.

    WHY SEPARATE NODE:
    Memory loading is IO-bound (file/DB read). Separating it means
    you can parallelize it with other setup steps in future.
    Also makes it easy to disable for testing.
    """
    print(f"\n[Memory Loader] Loading memories for user: {state['user_id']}")
    memories = load_episodic_memories(state["user_id"])
    if memories:
        print(f"[Memory Loader] Found {len(memories)} memories")
    return {"past_memories": memories}


def output_validator_node(state: AgentState) -> dict:
    """
    Output guardrail as a LangGraph node.
    Runs after synthesizer, before returning to user.
    """
    print(f"\n[Output Validator] Checking response")
    final = state.get("final_response", "")

    is_safe, reason = check_output(final)
    if not is_safe:
        print(f"[Output Validator] BLOCKED output: {reason}")
        safe_response = get_safe_fallback(reason)
        return {"final_response": safe_response}

    print(f"[Output Validator] PASSED")
    return {}  # No changes needed


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def route_after_safety(state: AgentState) -> str:
    """After safety gate: pass or end."""
    return "pass" if state["safety_passed"] else "end"


def route_after_finance(state: AgentState) -> str:
    """
    After finance agent: does entertainment also need to run?
    For multi-intent queries (e.g. finance + entertainment).
    """
    intent = state.get("intent", "finance")
    if intent == "multi" and not state.get("entertainment_result"):
        return "entertainment"
    return "synthesizer"


def route_after_entertainment(state: AgentState) -> str:
    """After entertainment agent: does orders also need to run?"""
    intent = state.get("intent", "orders")
    if intent == "multi" and not state.get("orders_result"):
        return "orders"
    return "synthesizer"


# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    Assembles all nodes and edges into a compiled LangGraph.

    Think of this as the architect's blueprint.
    add_node() = place a room on the map
    add_edge() = draw a hallway between rooms
    add_conditional_edges() = draw a hallway with a sign: "go left if X, right if Y"
    set_entry_point() = mark the front door
    compile() = hand the blueprint to the construction crew
    """
    builder = StateGraph(AgentState)

    # ── Add all nodes ──────────────────────────────────────────────────────
    builder.add_node("safety_gate",         safety_gate_node)
    builder.add_node("memory_loader",       memory_loader_node)
    builder.add_node("supervisor",          supervisor_node)
    builder.add_node("finance_agent",       finance_agent_node)
    builder.add_node("entertainment_agent", entertainment_agent_node)
    builder.add_node("orders_agent",        orders_agent_node)
    builder.add_node("synthesizer",         synthesizer_node)
    builder.add_node("output_validator",    output_validator_node)

    # ── Entry point ────────────────────────────────────────────────────────
    builder.set_entry_point("safety_gate")

    # ── Edges (the hallways) ───────────────────────────────────────────────
    # After safety: either proceed or end
    builder.add_conditional_edges(
        "safety_gate",
        route_after_safety,
        {"pass": "memory_loader", "end": END}
    )

    # Memory → Supervisor (always)
    builder.add_edge("memory_loader", "supervisor")

    # Supervisor → right specialist
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "finance_agent":       "finance_agent",
            "entertainment_agent": "entertainment_agent",
            "orders_agent":        "orders_agent",
            "synthesizer":         "synthesizer",
        }
    )

    # Finance → entertainment (multi) or synthesizer
    builder.add_conditional_edges(
        "finance_agent",
        route_after_finance,
        {"entertainment": "entertainment_agent", "synthesizer": "synthesizer"}
    )

    # Entertainment → orders (multi) or synthesizer
    builder.add_conditional_edges(
        "entertainment_agent",
        route_after_entertainment,
        {"orders": "orders_agent", "synthesizer": "synthesizer"}
    )

    # Orders → synthesizer (always — orders is always last specialist)
    builder.add_edge("orders_agent", "synthesizer")

    # Synthesizer → output validator → END
    builder.add_edge("synthesizer", "output_validator")
    builder.add_edge("output_validator", END)

    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZE
# ─────────────────────────────────────────────────────────────────────────────

print("[Startup] Seeding knowledge base...")
seed_knowledge_base()
print("[Startup] Building graph...")
app = build_graph()
print("[Startup] FinanceBot Pro ready!\n")
