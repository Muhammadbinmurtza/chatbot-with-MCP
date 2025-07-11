# mcp_agent_sdk.py
"""
Re‑implementation of the original MCP orchestration script using the **OpenAI Agent SDK**.
The script spins up an Agent that is able to:
  • communicate with an MCP documentation server ("doc_client")
  • spawn any extra server scripts passed on the command line
  • relay conversation between your terminal and those backend services — all through
    the SDK's `Runner` loop, rather than a hand‑rolled CLI loop.

Usage (same as before):
    python mcp_agent_sdk.py [optional_extra_server.py ...]

Env vars expected (see .env):
    OPENAI_API_KEY        – required
    OPENAI_MODEL          – e.g. "gpt-4o-mini" (default)
    USE_UV                – "1" to launch the doc server with uvicorn instead of python

Notes
-----
* The Anthropic‑specific `Claude` helper has been removed.
* We keep the existing `mcp_client.MCPClient` because that library is backend‑agnostic.
* All LLM calls now go through `AsyncOpenAI`, which the Agent SDK wraps for you.
* The old `CliChat`/`CliApp` are collapsed into a **single Tool** (`terminal_chat`) that
  seamlessly reads from `stdin` and writes to `stdout` when invoked by the agent.
* If you prefer to hook a GUI later, simply swap out the tool implementation – the
  agent logic stays unchanged.
"""

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from typing import Dict, Any
import asyncio, sys
from dotenv import load_dotenv

# --- OpenAI Agent SDK imports -------------------------------------------------
from agents import (
    Agent,
    Runner,
    function_tool,
    AsyncOpenAI,
    enable_verbose_stdout_logging,
    OpenAIChatCompletionsModel
)

from mcp_client import MCPClient  # unchanged 3rd‑party library

# -----------------------------------------------------------------------------
load_dotenv()

Gemini_API_KEY = os.getenv("Gemini_API")
client = AsyncOpenAI(
    api_key=Gemini_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

model1 = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client)
enable_verbose_stdout_logging()

# ----------------------------- TOOLS -----------------------------------------

@function_tool(name_override="terminal_chat", description_override="Read a line from the user and echo it back.")
async def terminal_chat() -> str:  # noqa: D401 – concise docstring ok here
    """Prompt the user in the terminal, then return their input as a string."""

    # (Because this is async and stdin/stdout are blocking, we call to_thread.)
    import asyncio, sys  # local import to keep tool hermetic

    def read_input() -> str:
        return input("👤 › ")

    user_input = await asyncio.to_thread(read_input)  # blocking read
    return user_input


@function_tool(
    name_override="send_to_client",
    description_override="Send a message to an MCP client and return its response.",
)
async def send_to_client(client_id: str, message: str, **kwargs: Any) -> str:
    """Forward *message* to the specified `MCPClient` and return its reply."""

    client = CLIENT_REGISTRY.get(client_id)
    if client is None:
        return f"⚠️ Unknown client_id '{client_id}'"

    # MCPClient has an async `chat` helper that takes str -> str
    response: str = await client.chat(message)
    return response


# The tool needs a global registry; we fill it in `main()` once the clients exist.
CLIENT_REGISTRY: Dict[str, MCPClient] = {}

# ------------------------- AGENT DEFINITION ----------------------------------

SYSTEM_PROMPT = (
    "You are an orchestration agent that routes user queries to the appropriate "
    "backend service (the doc server or one of the extra clients). "
    "For each question, decide which client can answer best, call the tool, and "
    "then present a concise reply."
)

agent = Agent(
    llm=AsyncOpenAI(model=model1),
    system_message=SYSTEM_PROMPT,
    tools=[terminal_chat, send_to_client],
)

# --------------------------- MAIN -------------------------------------------

async def main() -> None:
    global CLIENT_REGISTRY  # mutates the module‑level mapping

    server_scripts = sys.argv[1:]
    clients: Dict[str, MCPClient] = {}

    # Decide how to start the MCP documentation server
    command, args = (
        ("uv", ["run", "mcp_server.py"]) if os.getenv("USE_UV", "0") == "1" else ("python", ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(MCPClient(command=command, args=args))
        clients["doc_client"] = doc_client

        # Spawn any additional server scripts passed on the CLI
        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            # Always launch with uvicorn to keep latency low
            client = await stack.enter_async_context(MCPClient(command="uv", args=["run", server_script]))
            clients[client_id] = client

        # Publish mapping so the tool can route calls
        CLIENT_REGISTRY = clients

        # ---------------- Run the agent loop ----------------
        async with Runner(agent) as runner:
            print("🤖 Agent ready! Type your questions below. Ctrl‑C to quit.\n")
            while True:
                # Ask the agent to call `terminal_chat`, which will in turn read stdin
                result = await runner.run(tools=["terminal_chat"], allow_llm_responses=False)

                # `result` is whatever the agent returns – echo it out
                print(f"🤖 › {result}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Exiting…")
