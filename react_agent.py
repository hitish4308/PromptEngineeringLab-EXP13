"""
Experiment 13 - LLM Agent with LangGraph (ReAct)

Builds a ReAct agent that can:
  - Retrieve weather for a city (mock data)
  - Evaluate mathematical expressions

The agent decides which tools to call, in what order, and how to
combine results. Uses NVIDIA NIM for the LLM.
"""

import os
import ast
import operator
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.prebuilt import create_react_agent

# ---------- Setup ----------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise SystemExit("NVIDIA_API_KEY not found. Create a .env file.")

LLM_MODEL = "openai/gpt-oss-20b"


# ---------- Mock Weather Data ----------
WEATHER_DATA = {
    "london": 18,
    "paris": 22,
    "new york": 25,
    "tokyo": 28,
    "sydney": 20,
    "delhi": 33,
    "mumbai": 31,
}


# ---------- Tools ----------
@tool
def get_weather(city: str) -> str:
    """Get the current temperature for a city. Returns Celsius."""
    key = city.strip().lower()
    if key in WEATHER_DATA:
        temp_c = WEATHER_DATA[key]
        return f"The temperature in {city.title()} is {temp_c}°C."
    return f"Weather data for '{city}' not available."


# Safe math evaluator - only allows basic operations
SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
        return SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
        return SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic math expression like '320 * 15 / 100' or '18 * 9/5 + 32'.
    Only supports + - * / ** and parentheses."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"[calc error] {type(e).__name__}: {e}"


TOOLS = [get_weather, calculate]


# ---------- Query ----------
QUERY = (
    "What is the current temperature in London in Fahrenheit? "
    "Also calculate 15% of 320 and add it to the temperature."
)


# ---------- Main ----------
if __name__ == "__main__":
    print("=" * 72)
    print("  EXPERIMENT 13 - REACT AGENT (LangGraph)")
    print("=" * 72)
    print(f"LLM  : {LLM_MODEL}")
    print(f"Tools: {[t.name for t in TOOLS]}")
    print(f"\nQuery: {QUERY}\n")

    # LLM - temperature 0 for deterministic tool selection
    llm = ChatNVIDIA(model=LLM_MODEL, temperature=0.0, max_tokens=500)

    # Create ReAct agent
    agent = create_react_agent(llm, TOOLS)

    print("=" * 72)
    print("  AGENT EXECUTION TRACE")
    print("=" * 72)
    print()

    # Invoke the agent
    try:
        result = agent.invoke(
            {"messages": [("user", QUERY)]},
            config={"recursion_limit": 25},
        )
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        raise SystemExit(1)

    # Print the full message trace
    for i, msg in enumerate(result["messages"], start=1):
        role = getattr(msg, "type", msg.__class__.__name__)
        content = getattr(msg, "content", "")

        print(f"--- Message {i} ---")
        print(f"  Role: {role}")

        if role == "human":
            print(f"  Content: {content}")
        elif role == "ai":
            # AI message can contain tool calls
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    print(f"  Tool call: {tc.get('name')}")
                    print(f"    Args   : {tc.get('args')}")
            if content:
                preview = content if len(content) < 500 else content[:500] + "..."
                print(f"  Content: {preview}")
        elif role == "tool":
            print(f"  Tool result: {content}")

        print()

    # Final answer
    print("=" * 72)
    print("  FINAL ANSWER")
    print("=" * 72)
    print()
    final = result["messages"][-1].content
    print(final)
    print()

    # ---------- Summary ----------
    print("=" * 72)
    print("  SUMMARY - REACT AGENT CONCEPTS")
    print("=" * 72)
    print("""
The ReAct loop (Reason + Act):

  1. THOUGHT  - the model decides what to do next
  2. ACTION   - it selects a tool and provides arguments
  3. OBSERVATION - the tool returns a result
  4. Repeat until the model has enough information
  5. FINAL ANSWER - the model produces a natural-language reply

Key components:

  - Tools: @tool-decorated functions with clear docstrings.
    The docstring is the model's ONLY description of what the tool does,
    so it must be precise.

  - Agent: create_react_agent(llm, tools). This wires the LLM into a
    graph that alternates between reasoning and tool-calling.

  - Recursion limit: caps the number of iterations to prevent infinite
    loops. 25 is generous for simple tasks.

  - Temperature 0: deterministic tool selection. Higher temperatures
    make the agent's choices less predictable.

Design principles:

  - Tool descriptions matter more than code.
  - Each tool should do ONE thing well.
  - Tools should return structured, easy-to-parse output.
  - The agent relies on observations, not its own memory, for facts.
""")
