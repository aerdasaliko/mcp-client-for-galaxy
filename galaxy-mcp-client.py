import asyncio
import json
import re
from typing import Any
import logging

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from langchain.tools import Tool
from langchain_community.chat_models import ChatDeepInfra
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.memory import ConversationBufferMemory

# logging for tests and debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("galaxy_mcp_client")
tool_logger = logging.getLogger("galaxy_mcp_client.tools")
logging.getLogger("mcp").setLevel(logging.DEBUG)
logging.getLogger("langchain").setLevel(logging.DEBUG) 

load_dotenv()

server_params = StdioServerParameters(
    command="uv",
    args=["run", "<path-to-mcp-server>"]
)

llm = ChatDeepInfra(
    model="openai/gpt-oss-120b",
    max_retries=5,
    max_tokens=16384
)

minillm = ChatDeepInfra(model="google/gemma-3-4b-it")

def stringify_content(result: Any) -> str:
    """Flatten MCP responses into plain text."""
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, TextContent):
                parts.append(item.text)
            else:
                parts.append(f"[{item.type}] {item!r}")
        return "\n".join(parts)
    if isinstance(result, TextContent):
        return result.get("output", "")
    return str(result)

def wrap_for_react(structured_tool, loop: asyncio.AbstractEventLoop):
    """
    Expose a structured MCP tool as a string-based tool for ReAct agents.
    Wrap an MCP structured tool (which is async and expects JSON)
    into a LangChain Tool that:
      - accepts string input (ReAct requirement)
      - exposes both sync and async interfaces
    """
    schema_dict = None
    schema = getattr(structured_tool, "input_schema", None)

    # normalize schema into a plain dict
    if schema:
        if isinstance(schema, dict):
            schema_dict = schema
        elif hasattr(schema, "model_json_schema"):
            schema_dict = schema.model_json_schema()
        elif hasattr(schema, "dict"):
            schema_dict = schema.dict()
        else:
            schema_dict = {"description": str(schema)}

    # parse input JSON
    def _prep_input(input_str: str):
        stripped = input_str.strip()
        # Handle empty input
        if not stripped: return {}

        # tolerate ```json ... ```
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        
        # tolerate inline backticks
        if stripped.startswith("`") and stripped.endswith("`"):
            stripped = stripped[1:-1].strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            llm_fix = minillm.invoke(
                f"Create a valid JSON using the schema {schema} and the information from {input_str}. "
                f"Return only JSON, no code fences."
            )
            return llm_fix

    # agent is running synchronously, but tools are async
    def _run(input_str: str, _tool=structured_tool):
        payload = _prep_input(input_str)
        tool_logger.debug("Calling tool %s sync with payload=%s", _tool.name, payload)
        # submit the async tool call to the event loop
        future = asyncio.run_coroutine_threadsafe(_tool.ainvoke(payload), loop)
        result = future.result() # block thread until complete
        tool_logger.debug("Tool %s returned: %s", _tool.name, result)
        return stringify_content(result)

    # agent is running async  
    async def _arun(input_str: str, _tool=structured_tool):
        payload = _prep_input(input_str)
        tool_logger.debug("Calling tool %s async with payload=%s", _tool.name, payload)
        # await tool call with safety timeout
        result = await _tool.ainvoke(payload)
        tool_logger.debug("Tool %s returned: %s", _tool.name, result)
        return stringify_content(result)

    description = structured_tool.description or ""
    if schema_dict:
        description += (
            "\nArguments must be provided as JSON matching this schema:\n"
            + json.dumps(schema_dict, indent=2)
        )

    # return wrapped tool
    return Tool(
        name=structured_tool.name,
        description=description.strip(),
        func=_run,  #sync
        coroutine=_arun,  #async
    )

# Adding line-editing for ease of use
session = PromptSession("You: ")
async def get_input():
    with patch_stdout():
        return await session.prompt_async()

async def run():
    # start mcp server and session
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # initialize the connection
            await session.initialize()
            loop = asyncio.get_running_loop()

            # raw tools are (1) mcp structured tools, (2) async, 
            # (3) expect json input, (4) return structured content
            # -> wrap tools so they accept strings, expose both sync
            # and async interfaces 
            raw_tools = await load_mcp_tools(session)
            tools = [wrap_for_react(t, loop) for t in raw_tools]

            prompt = hub.pull("hwchase17/react-chat").partial(
                instructions=(
                    "You are an assistant. Respond as follows:\n"
                    "- You can output observations, lists, or free text.\n"
                    "- If you need to use a tool, include 'Action:' and 'Action Input:' always with a valid JSON.\n"
                    "- If no tool is needed, you may directly write 'Final Answer:'\n"
                    "- Always produce a complete response without truncating.\n"
                    "- Do not call a tool if the information is in your memory."
                ),
                input_variables=["input", "chat_history"]
            )

            # react agents pass string inputs, call tools sync or async
            agent = create_react_agent(llm, tools, prompt)
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                memory=memory,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=None,
            )

            console = Console()
            while True:
                try:
                    user_msg = await get_input()
                    if user_msg.lower() in {"exit", "quit"}:
                        console.print("Goodbye!")
                        break
                    with console.status("[green]Thinking...[/green]"):
                        result = await asyncio.to_thread(agent_executor.invoke, {"input": user_msg})
                    output = result.get("output", "")
                    console.print(
                        Panel.fit(
                            Text(output, style="white"),
                            title="[bold green]Assistant[/bold green]",
                            border_style="green"
                        )
                    )

                except (KeyboardInterrupt, asyncio.CancelledError):
                    console.print("Goodbye!")
                    break
                except Exception as exc:
                    # Handle all other errors without stopping the loop
                    error_type = type(exc).__name__
                    error_msg = str(exc)
                    console.print(
                        Panel.fit(
                                f"\nAssistant: Sorry, something went wrong ({error_type}: {error_msg}).",
                                title="[bold red]Error[/bold red]",
                                border_style="red"
                            )
                    )
                    logger.exception("Agent execution failed")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
