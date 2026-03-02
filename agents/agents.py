"""
agents/agents.py — All agent node functions for the LangGraph workflow.

HUMAN ANALOGY — THE WHOLE SYSTEM:
Think of this like a hospital with specialist departments.

Reception (Supervisor): Patient walks in, receptionist assesses symptoms,
sends to the right department. Does NOT treat the patient themselves.

Cardiology (Finance Agent): Specialist for heart issues (financial questions).
Has their own tools (ECG machine = stock price tool, calculator).
Uses Chain-of-Thought to reason through complex cases before diagnosing.

Entertainment (Entertainment Agent): Like a leisure therapy department.
Recommends activities (shows/movies) based on patient preferences.
Checks records (memory) for what the patient liked before.

Orders (Orders Agent): Like patient administration.
Tracks appointments (orders), handles paperwork (returns/cancellations).

Synthesizer: Like the discharge summary doctor. Takes all specialist reports
and writes one clear summary letter for the patient to take home.
"""

import json
from state import AgentState
from config import tracked_call, MODELS
from tools.tools import TOOL_DEFINITIONS, execute_tool

# Give each agent only its relevant tools (Groq/Llama works better with fewer tools)
FINANCE_TOOLS = [t for t in TOOL_DEFINITIONS if t["name"] in ["calculate", "get_stock_price", "calculate_emi"]]
ENTERTAINMENT_TOOLS = [t for t in TOOL_DEFINITIONS if t["name"] in ["get_entertainment_info"]]
ORDERS_TOOLS = [t for t in TOOL_DEFINITIONS if t["name"] in ["track_order", "calculate"]]
from memory.memory_store import retrieve_from_vector_db, load_episodic_memories
from prompts.prompts import (
    SUPERVISOR_PROMPT, FINANCE_AGENT_PROMPT,
    ENTERTAINMENT_AGENT_PROMPT, ORDERS_AGENT_PROMPT,
    SYNTHESIZER_PROMPT
)


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR AGENT
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> dict:
    """
    Routes the user request to the right specialist agent(s).

    CHAIN OF THOUGHT PATTERN:
    We force the supervisor to respond in JSON — this is a structured output
    pattern using prompt constraints rather than tool_choice.
    For routing decisions, JSON is more reliable than asking for prose.

    FAILURE SCENARIO — INVALID JSON:
    Claude sometimes returns JSON wrapped in markdown (```json ... ```).
    We strip those markers before parsing.
    If parsing still fails, we default to finance_agent rather than crashing.
    """
    print(f"\n[Supervisor] Analyzing: '{state['user_input'][:50]}...'")

    response, cost = tracked_call(
        model=MODELS["classify"],
        system=SUPERVISOR_PROMPT,
        messages=[{"role": "user", "content": state["user_input"]}],
        max_tokens=256,
        use_cache=True  # Cache the supervisor system prompt — called every request
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        routing = json.loads(raw)
        intent = routing.get("intent", "finance")
        subtasks = [r["subtask"] for r in routing.get("routing", [])]
        agents_to_call = [r["agent"] for r in routing.get("routing", [])]
        print(f"[Supervisor] Intent: {intent} | Agents: {agents_to_call}")
    except (json.JSONDecodeError, KeyError):
        # FALLBACK: If JSON parsing fails, default behavior
        # This is a graceful degradation — system keeps working
        print(f"[Supervisor] JSON parse failed, defaulting to finance_agent")
        intent = "finance"
        subtasks = [state["user_input"]]
        agents_to_call = ["finance_agent"]

    return {
        "intent": intent,
        "subtasks": subtasks,
        "cost_usd": state["cost_usd"] + cost,
        "iteration_count": state["iteration_count"] + 1,
    }


def route_from_supervisor(state: AgentState) -> str:
    """
    LangGraph conditional edge: decides which node to go to after supervisor.

    HUMAN ANALOGY:
    The receptionist's decision: "You need cardiology" → walks you to cardiology.
    "You need both cardiology and radiology" → takes you to cardiology first.
    """
    intent = state.get("intent", "finance")

    if intent == "entertainment":
        return "entertainment_agent"
    elif intent == "orders":
        return "orders_agent"
    elif intent == "general":
        return "synthesizer"
    else:
        return "finance_agent"  # Default for finance + multi


# ─────────────────────────────────────────────────────────────────────────────
# REACT AGENT RUNNER (shared by all specialist agents)
# ─────────────────────────────────────────────────────────────────────────────

def run_react_loop(system_prompt, user_message,
                   context="", model=None, tools=None,
                   max_iterations=6):
    """
    The ReAct loop: Reason → Act (tool) → Observe → Repeat.

    HUMAN ANALOGY:
    Like a chef following a recipe:
    THINK: "I need to make a sauce. I need tomatoes."
    ACT: Opens fridge (tool call). Gets tomatoes.
    OBSERVE: "I have tomatoes. Now I need to chop them."
    THINK: "I need a knife."
    ACT: Gets knife from drawer (another tool call).
    OBSERVE: "Tomatoes chopped. Now I can cook."
    FINAL: Makes the sauce and serves it.

    Each loop = one Think+Act cycle.
    Loop stops when Claude gives a final answer (no more tool calls needed).

    FAILURE SCENARIOS:
    1. Tool keeps failing → max_iterations prevents infinite loop
    2. Claude ignores tool results → adding "Use the tool result above" to messages
    3. Claude makes up tool results → we ALWAYS execute tools, never let Claude fake it
    """
    if model is None:
        model = MODELS["reason"]

    # Build the initial message with any retrieved context
    full_message = user_message
    if context:
        full_message = f"<retrieved_context>\n{context}\n</retrieved_context>\n\n{user_message}"

    messages = [{"role": "user", "content": full_message}]
    total_cost = 0.0
    tool_calls_log = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"  [ReAct loop iteration {iteration}]")

        response, cost = tracked_call(
            model=model,
            system=system_prompt,
            messages=messages,
            max_tokens=1024,
            tools=tools or TOOL_DEFINITIONS,
            use_cache=True
        )
        total_cost += cost

        # Case 1: Claude is done — gave a final text answer
        if response.stop_reason == "end_turn":
            final_text = next(
                (b.text for b in response.content if hasattr(b, "text") and b.text),
                ""
            )
            print(f"  [ReAct done after {iteration} iterations]")
            return final_text, total_cost, tool_calls_log

        # Case 2: Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Convert ContentBlock objects to plain dicts for JSON serialization
            # Groq needs: {"role": "assistant", "tool_calls": [...]} format
            tool_calls_for_history = []
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_calls_for_history.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                        }
                    })

            # Add assistant message with tool calls to history
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls_for_history
            })

            # Execute every tool and add results
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    print(f"  [Tool call] {tool_name}({json.dumps(tool_input)[:60]}...)")
                    result = execute_tool(tool_name, tool_input)
                    print(f"  [Tool result] {result[:80]}...")

                    tool_calls_log.append(f"{tool_name}: {str(tool_input)[:50]}")

                    # Each tool result is its own message in Groq format
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.id,
                        "content": str(result)
                    })

        else:
            # Unexpected stop reason — break to avoid hanging
            print(f"  [ReAct] Unexpected stop_reason: {response.stop_reason}")
            break

    # Safety: return whatever we have if max iterations hit
    last_text = ""
    for msg in reversed(messages):
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if hasattr(block, "text") and block.text:
                    last_text = block.text
                    break
        elif isinstance(msg.get("content"), str) and msg.get("role") == "assistant":
            last_text = msg["content"]
            break

    return last_text or "I reached my processing limit. Please try a simpler question.", total_cost, tool_calls_log


# ─────────────────────────────────────────────────────────────────────────────
# FINANCE AGENT
# ─────────────────────────────────────────────────────────────────────────────

def finance_agent_node(state: AgentState) -> dict:
    """
    Handles all finance-related queries using ReAct pattern with CoT.

    CHAIN OF THOUGHT in finance:
    Before calculating investment returns, Claude thinks:
    "User wants to know if SIP is better than FD.
    I need: SIP return rate, FD rate, investment period.
    Let me calculate both using the calculator tool.
    Then compare risk profiles."

    This step-by-step reasoning catches errors before the final answer.
    """
    print("\n[Finance Agent] Starting")

    # RAG: Retrieve relevant financial knowledge
    retrieved = retrieve_from_vector_db(state["user_input"], top_k=3)
    context = "\n".join(f"- {text}" for text, score in retrieved) if retrieved else ""

    # Load past memories for personalization
    memories = load_episodic_memories(state["user_id"])
    memory_context = ""
    if memories:
        memory_context = "\n<user_history>\n" + "\n".join(f"- {m}" for m in memories) + "\n</user_history>"

    # The subtask from supervisor (more specific than raw user input)
    task = state["subtasks"][0] if state["subtasks"] else state["user_input"]
    full_task = f"{task}{memory_context}"

    result, cost, tools_used = run_react_loop(
        system_prompt=FINANCE_AGENT_PROMPT,
        user_message=full_task,
        context=context,
        model=MODELS["reason"],
        tools=FINANCE_TOOLS
    )

    return {
        "finance_result": result,
        "retrieved_context": context,
        "tool_calls_made": state["tool_calls_made"] + tools_used,
        "cost_usd": state["cost_usd"] + cost,
        "iteration_count": state["iteration_count"] + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTERTAINMENT AGENT
# ─────────────────────────────────────────────────────────────────────────────

def entertainment_agent_node(state: AgentState) -> dict:
    """
    Handles movie/show recommendations using RAG from the entertainment knowledge base.

    KEY PATTERN — MEMORY + RAG TOGETHER:
    1. Episodic memory: "User previously liked Mirzapur (crime thriller)"
    2. RAG: Finds similar crime thrillers in the knowledge base
    3. Tool: Gets actual ratings for recommended titles
    Combined, this gives a deeply personalized recommendation.
    """
    print("\n[Entertainment Agent] Starting")

    retrieved = retrieve_from_vector_db(state["user_input"], top_k=3)
    context = "\n".join(f"- {text}" for text, score in retrieved) if retrieved else ""

    memories = load_episodic_memories(state["user_id"])
    memory_context = ""
    if memories:
        entertainment_memories = [m for m in memories if any(
            word in m.lower() for word in ["movie", "show", "watch", "netflix", "ott", "series"]
        )]
        if entertainment_memories:
            memory_context = "\n<user_preferences>\n" + "\n".join(f"- {m}" for m in entertainment_memories) + "\n</user_preferences>"

    task = state["subtasks"][-1] if state["subtasks"] else state["user_input"]

    result, cost, tools_used = run_react_loop(
        system_prompt=ENTERTAINMENT_AGENT_PROMPT,
        user_message=f"{task}{memory_context}",
        context=context,
        model=MODELS["reason"],
        tools=ENTERTAINMENT_TOOLS
    )

    return {
        "entertainment_result": result,
        "tool_calls_made": state["tool_calls_made"] + tools_used,
        "cost_usd": state["cost_usd"] + cost,
        "iteration_count": state["iteration_count"] + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ORDERS AGENT
# ─────────────────────────────────────────────────────────────────────────────

def orders_agent_node(state: AgentState) -> dict:
    """
    Handles order tracking and customer service queries.

    KEY PATTERN — TOOL-FIRST REASONING:
    Orders agent MUST call track_order tool before answering anything
    about a specific order. We enforce this in the system prompt.
    Without this constraint, Claude might answer from outdated memory.
    """
    print("\n[Orders Agent] Starting")

    retrieved = retrieve_from_vector_db(state["user_input"], top_k=2)
    context = "\n".join(f"- {text}" for text, score in retrieved) if retrieved else ""

    task = state["subtasks"][-1] if state["subtasks"] else state["user_input"]

    result, cost, tools_used = run_react_loop(
        system_prompt=ORDERS_AGENT_PROMPT,
        user_message=task,
        context=context,
        model=MODELS["reason"],
        tools=ORDERS_TOOLS
    )

    return {
        "orders_result": result,
        "tool_calls_made": state["tool_calls_made"] + tools_used,
        "cost_usd": state["cost_usd"] + cost,
        "iteration_count": state["iteration_count"] + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE SYNTHESIZER
# ─────────────────────────────────────────────────────────────────────────────

def synthesizer_node(state: AgentState) -> dict:
    """
    Combines outputs from all specialist agents into one coherent response.

    WHEN THIS IS CALLED:
    - After a single specialist (wraps their output with proper formatting)
    - After multiple specialists (merges outputs seamlessly)
    - For general conversation (no specialist needed)

    USES OPUS: This is the final customer-facing response. Quality matters most here.
    All the cheap Haiku calls earlier saved money so we can use Opus for what matters.
    """
    print("\n[Synthesizer] Composing final response")

    # Gather all specialist outputs
    specialist_outputs = []
    if state.get("finance_result"):
        specialist_outputs.append(f"<finance_output>\n{state['finance_result']}\n</finance_output>")
    if state.get("entertainment_result"):
        specialist_outputs.append(f"<entertainment_output>\n{state['entertainment_result']}\n</entertainment_output>")
    if state.get("orders_result"):
        specialist_outputs.append(f"<orders_output>\n{state['orders_result']}\n</orders_output>")

    if not specialist_outputs:
        # No specialist was called — handle as general conversation
        response, cost = tracked_call(
            model=MODELS["synthesize"],
            system="You are FinanceBot Pro, a helpful assistant. Answer concisely.",
            messages=[{"role": "user", "content": state["user_input"]}],
            max_tokens=256
        )
        return {
            "final_response": response.content[0].text,
            "cost_usd": state["cost_usd"] + cost,
        }

    synthesis_prompt = f"""Original user question: {state['user_input']}

Specialist agent outputs:
{''.join(specialist_outputs)}

Compose a single, coherent final response for the user."""

    response, cost = tracked_call(
        model=MODELS["synthesize"],
        system=SYNTHESIZER_PROMPT,
        messages=[{"role": "user", "content": synthesis_prompt}],
        max_tokens=512,
        use_cache=True
    )

    return {
        "final_response": response.content[0].text,
        "cost_usd": state["cost_usd"] + cost,
        "iteration_count": state["iteration_count"] + 1,
    }
