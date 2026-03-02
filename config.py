"""
config.py — Central configuration using Groq (FREE tier).

WHY GROQ:
- Completely free API, no credit card needed
- Extremely fast (runs on custom LPU chips)
- Runs Llama 3.3 70B — very capable, close to Claude Sonnet quality
- Same OpenAI-compatible API format

SWITCHING TO CLAUDE LATER:
When ready: pip install anthropic, set ANTHROPIC_API_KEY, swap back to original config.py
Everything else stays identical.
"""

import os
import json
from groq import Groq

MODELS = {
    "classify":   "llama-3.1-8b-instant",
    "retrieve":   "llama-3.1-8b-instant",
    "reason":     "llama-3.3-70b-versatile",
    "synthesize": "llama-3.3-70b-versatile",
}

PRICING = {
    "llama-3.1-8b-instant":    {"input": 0.0, "output": 0.0},
    "llama-3.3-70b-versatile": {"input": 0.0, "output": 0.0},
}

MAX_INPUT_CHARS = 5000
MAX_ITERATIONS = 6
MAX_TOOL_RETRIES = 3


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found.\n"
            "Run: set GROQ_API_KEY=gsk_your-key-here\n"
            "Get free key at: https://console.groq.com"
        )
    return Groq(api_key=api_key)

client = get_client()


def calculate_cost(model, input_tokens, output_tokens):
    return 0.0


class ContentBlock:
    def __init__(self, block_type, text="", name="", input=None, block_id=""):
        self.type = block_type
        self.text = text
        self.name = name
        self.input = input or {}
        self.id = block_id


class GroqResponseWrapper:
    """Makes Groq responses look like Claude responses — zero changes needed in agents.py"""
    def __init__(self, groq_response):
        self.raw = groq_response
        self.content = []
        self.stop_reason = "end_turn"
        self._parse()

    def _parse(self):
        if not self.raw.choices:
            self.content = [ContentBlock("text", "No response generated")]
            return
        choice = self.raw.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason
        if finish_reason == "tool_calls" and message.tool_calls:
            self.stop_reason = "tool_use"
            for tc in message.tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except Exception:
                    tool_input = {}
                self.content.append(ContentBlock(
                    block_type="tool_use",
                    name=tc.function.name,
                    input=tool_input,
                    block_id=tc.id
                ))
        else:
            self.stop_reason = "end_turn"
            self.content.append(ContentBlock("text", message.content or ""))

    @property
    def usage(self):
        return self.raw.usage


def tracked_call(model, messages, system="", max_tokens=512,
                 tools=None, tool_choice=None, use_cache=False):
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    kwargs = {
        "model": model,
        "messages": full_messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    if tools:
        # Convert Anthropic tool format to OpenAI format for Groq
        groq_tools = []
        for t in tools:
            groq_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"]
                }
            })
        kwargs["tools"] = groq_tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    wrapped = GroqResponseWrapper(response)

    in_tok = response.usage.prompt_tokens if response.usage else 0
    out_tok = response.usage.completion_tokens if response.usage else 0
    model_short = model.split("-")[1] if "-" in model else model
    print(f"    [API:{model_short}] {in_tok}in+{out_tok}out = FREE")

    return wrapped, 0.0
