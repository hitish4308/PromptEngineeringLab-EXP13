# Prompt Engineering Lab - Experiment 13

## LLM Agent with LangGraph (ReAct)

Builds a ReAct agent with two tools: get_weather and calculate. The
agent decides which tools to call, in what order, and combines results
into a final answer.

## The Task

"What is the current temperature in London in Fahrenheit? Also
calculate 15% of 320 and add it to the temperature."

The agent must:

1. Call get_weather("London") -> 18°C
2. Convert Celsius to Fahrenheit using calculate -> 64.4°F
3. Compute 15% of 320 using calculate -> 48.0
4. Add the two using calculate -> 112.4
5. Produce a final answer

## Tools

| Tool | Description |
|------|-------------|
| get_weather(city) | Returns the current temperature for a city in Celsius (mock data) |
| calculate(expression) | Evaluates basic math expressions (+, -, *, /, **) |

The docstring of each tool is its only description to the model. Precise
docstrings are essential for correct tool selection.

## The ReAct Loop

    THOUGHT  ->  ACTION  ->  OBSERVATION  ->  REPEAT  ->  FINAL ANSWER

The agent repeats this loop until it has enough information to answer.

## Model and Parameters

- LLM: openai/gpt-oss-20b (via ChatNVIDIA)
- Temperature: 0.0 (deterministic tool selection)
- Recursion limit: 25 (caps iterations)

## Setup

Reuse the environment from Experiment 1:

    Copy-Item ..\EXP_1\.env .
    python -m pip install -r requirements.txt

## Run

    python react_agent.py

## Expected Output

- Agent execution trace (every human, AI, and tool message)
- Final answer
- A summary of ReAct agent concepts

## Key Findings

1. **Tool docstrings are the API** - the model picks tools based on the
   docstring alone. Ambiguous descriptions cause wrong tool calls.

2. **The agent plans** - it breaks the multi-step query into individual
   calls without being told to. This is the reasoning half of ReAct.

3. **Observations drive the next step** - the agent uses tool outputs
   rather than its own assumptions. This grounds the answer.

4. **Temperature 0 matters** - with higher temperatures the agent may
   pick the wrong tool or produce unstable reasoning.

5. **Recursion limits prevent runaway loops** - always set a cap when
   the agent has multiple tools.

## Troubleshooting

### Error: 410 Gone on the LLM model

NVIDIA retired the model. Swap LLM_MODEL to a live one:

    LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b"
    LLM_MODEL = "meta/llama-3.3-70b-instruct"

### Agent loops forever

Reduce the recursion limit or tighten the tool docstrings. Ensure
each tool has an unambiguous purpose.

### Tool is never called

The docstring may not match the query vocabulary. Try rewriting it
to include the exact words the user is likely to use.

### ModuleNotFoundError: langgraph

    python -m pip install langgraph

## Security

- Never commit .env
- Never paste API keys in chat, logs, or screenshots

## License

For educational / lab use only.
