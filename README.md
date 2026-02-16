# MCP Client for Galaxy
An interactive AI assistant client for the Galaxy platform using MCP. Provides a ReAct-style AI interface for executing Galaxy tasks.

## Features

- **Interactive command-line interface** with line-editing and prompt support.
- **Structured tool integration**: MCP tools are wrapped for ReAct agents and accept JSON inputs.
- **Large Language Model support** via DeepInfra.
- **Conversation memory** to maintain context using LangChain's `ConversationBufferMemory`.
- **Flexible tool invocation**: handles both sync and async tools automatically.

## Setup

1. Clone this repository
```
git clone https://github.com/aerdasaliko/mcp-client-for-galaxy.git
cd galaxy-mcp-client
```
2. Create a .env file with your environment variables:
```
GALAXY_API_KEY=<your-galaxy-api-key>
DEEPINFRA_API_KEY=<your-deepinfra-api-key>
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Run the assistant
```
uv run galaxy-mcp-client.py
```
