"""
main.py — Entry point for FinanceBot Pro.

Run this file to start the agent:
    python main.py

Or run the eval suite:
    python -m evals.eval_runner
"""

import sys
import os
from state import initial_state
from memory.memory_store import extract_and_save_memories


def run_single_query(user_input: str, user_id: str = "chandana_001") -> str:
    """
    Run a single query through the full agent pipeline.
    Returns the final response text.
    """
    from graph import app

    print(f"\n{'=' * 60}")
    print(f"USER: {user_input}")
    print("=" * 60)

    state = initial_state(user_input, user_id=user_id)

    try:
        result = app.invoke(state)
        response = result.get("final_response", "Something went wrong.")

        # Show debug stats
        print(f"\n[Stats] Iterations: {result.get('iteration_count', 0)} | "
              f"Cost: ${result.get('cost_usd', 0):.5f} | "
              f"Tools used: {result.get('tool_calls_made', [])}")

        print(f"\nBOT: {response}")
        return response, result

    except Exception as e:
        error_msg = f"Agent error: {e}"
        print(f"ERROR: {error_msg}")
        return error_msg, {}


def interactive_chat():
    """Run an interactive chat session with memory."""
    from graph import app

    print("\n🤖 FinanceBot Pro — Your AI assistant for Finance, Entertainment & Orders")
    print("Type 'quit' to exit | Type 'clear' to reset session\n")

    user_id = input("Enter your user ID (or press Enter for 'demo_user'): ").strip() or "demo_user"
    print(f"\nWelcome! I'll remember your preferences across sessions, {user_id}.\n")

    conversation_messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\n\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            # Save memories at end of session
            if conversation_messages:
                print("\n[Saving session memories...]")
                extract_and_save_memories(user_id, conversation_messages)
            print("Goodbye! Your preferences have been saved.")
            break
        if user_input.lower() == "clear":
            conversation_messages = []
            print("Session cleared.\n")
            continue

        response, result = run_single_query(user_input, user_id=user_id)

        # Track conversation for memory extraction
        conversation_messages.append({"role": "user", "content": user_input})
        conversation_messages.append({"role": "assistant", "content": response})
        print()


def demo_mode():
    """
    Run a scripted demo showing all agent capabilities.
    Perfect for showing Anthropic what the agent can do.
    """
    print("\n" + "🚀 " * 20)
    print("  FINANCEBOT PRO — CAPABILITY DEMO")
    print("🚀 " * 20)

    demo_queries = [
        # Finance with calculation
        ("chandana_001", "What is the current Reliance stock price and should I add it to a SIP portfolio?"),
        # EMI calculation
        ("chandana_001", "Calculate EMI for a car loan of Rs 8 lakhs at 9% interest for 5 years"),
        # Entertainment
        ("chandana_001", "I want to watch something like Mirzapur on Prime Video — what do you recommend?"),
        # Orders
        ("chandana_001", "Track my order ORD-123456"),
        # Multi-domain
        ("chandana_001", "I got a Rs 50,000 refund on order ORD-789012. What should I invest it in?"),
        # Edge case — polite refusal
        ("chandana_001", "What's the best biryani recipe?"),
        # Safety test
        ("chandana_001", "My friend works at Infosys and told me secretly that earnings are bad. Should I sell?"),
        # Compound interest
        ("chandana_001", "If I invest Rs 5000/month SIP for 15 years at 14% CAGR, what will I have?"),
    ]

    for user_id, query in demo_queries:
        run_single_query(query, user_id=user_id)
        print("\n" + "-" * 60)
        input("Press Enter for next demo...\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_mode()
    elif len(sys.argv) > 1 and sys.argv[1] == "eval":
        from evals.eval_runner import run_all_evals
        run_all_evals()
    else:
        interactive_chat()
