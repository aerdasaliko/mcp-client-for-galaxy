# MCP Client for Galaxy
An interactive AI assistant client for the Galaxy platform using MCP. Provides a ReAct-style AI interface for executing Galaxy tasks.

## Features

- **Interactive command-line interface** with line-editing and prompt support.
- **Structured tool integration**: MCP tools are wrapped for ReAct agents and accept JSON inputs.
- **Large Language Model support** via DeepInfra.
- **Conversation memory** to maintain context using LangChain's `ConversationBufferMemory`.
- **Flexible tool invocation**: handles both sync and async tools automatically.

## Requirements
- Python 3.11+

## Setup

1. Clone this repository
```bash
git clone https://github.com/aerdasaliko/mcp-client-for-galaxy.git
cd galaxy-mcp-client
```
2. Create a .env file with your environment variables
```
GALAXY_API_KEY=<your-galaxy-api-key>
GALAXY_URL="https://usegalaxy.eu/"
DEEPINFRA_API_KEY=<your-deepinfra-api-key>
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Set the path to your MCP server in `galaxy-mcp-client.py`
```python
server_params = StdioServerParameters(
    command="uv",
    args=["run", "<path-to-mcp-server>"]
)
``` 
5. Run the assistant
```bash
uv run galaxy-mcp-client.py
```
