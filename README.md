# Gemma MCP Client

A Python package that combines Google's Gemma language model with MCP (Model Content Protocol) server integration, enabling powerful function calling capabilities across both local functions and remote MCP tools.

## Features

- Seamless integration with Google's Gemma language model
- Support for both local Python functions and remote MCP tools
- Automatic tool discovery and registration from MCP servers
- Python-style function calling syntax
- Proper resource management with async context managers
- Support for multiple MCP servers
- Easy testing through test server support

## Installation

```bash
uv add gemma-mcp # or pip install gemma-mcp if you love the old way
```

## Requirements

- Python 3.10+
- `google-genai`: Google Generative AI Python SDK
- `FastMCP` MCP utilities

## Usage

### Basic Usage

```python
from gemma_mcp import GemmaMCPClient

# a standard MCP configuration
mcp_config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather-api.example.com/mcp"
        },
        "assistant": {
            "command": "python",
            "args": ["./assistant_server.py"]
        }
    }
}

# Initialize client with MCP support
async with GemmaMCPClient(mcp_config=mcp_config).managed() as client:
    # Chat with automatic function execution
    response = await client.chat(
        "What's the weather like in London?",
        execute_functions=True
    )
    print(response)
```

### Adding Local Functions

You can add local functions in three ways:

1. Using a callable:

```python
async def my_function(param1: str, param2: int = 0):
    """Function description."""
    return {"result": param1 + str(param2)}

client.add_function(my_function)
```

2. Using a dictionary:

```python
function_def = {
    "name": "my_function",
    "description": "Function description",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
            "param2": {"type": "integer", "default": 0}
        },
        "required": ["param1"]
    }
}
client.add_function(function_def)
```

3. Using a FunctionDefinition object:

```python
from gemma_mcp import FunctionDefinition

function_def = FunctionDefinition(
    name="my_function",
    description="Function description",
    parameters={
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
            "param2": {"type": "integer", "default": 0}
        },
        "required": ["param1"]
    },
    required=["param1"]
)
client.add_function(function_def)
```

### MCP Server Configuration

The MCP configuration supports multiple server types:

1. servers with SSE transport:

```python
mcp_config = {
    "mcpServers": {
        "server_name": {
            "url": "https://server-url/mcp"
        }
    }
}
```

2. servers with STDIO transport:

```python
mcp_config = {
    "mcpServers": {
        "server_name": {
            "command": "python",
            "args": ["./server.py"]
        }
    }
}
```

### Testing

The package includes support for testing with in-memory MCP servers:

```python
from fastmcp import FastMCP
from gemma_mcp import GemmaMCPClient

# Create test server
mcp = FastMCP("Test Server")

# Initialize client with test server
client = GemmaMCPClient()
client.mcp_client.add_test_server(mcp)

# Use the client as normal
async with client.managed():
    response = await client.chat("Test message", execute_functions=True)
```

## API Reference

### GemmaMCPClient

The main client class that handles both Gemma model interactions and MCP tool integration.

#### Parameters

- `api_key` (str, optional): Gemini API key. If not provided, will look for GEMINI_API_KEY env var
- `model` (str): Model to use, defaults to "gemma-3-27b-it"
- `temperature` (float): Generation temperature, defaults to 0.7
- `system_prompt` (str, optional): Custom system prompt
- `mcp_config` (dict, optional): MCP configuration dictionary

#### Methods

- `add_function(function)`: Add a function definition
- `chat(message, execute_functions=False)`: Send a message and get response
- `initialize()`: Initialize the client and all components
- `cleanup()`: Clean up all resources

### FunctionDefinition

A dataclass for representing function definitions.

#### Parameters

- `name` (str): Function name
- `description` (str): Function description
- `parameters` (dict): Function parameters schema
- `required` (list): List of required parameters
- `callable` (callable, optional): The actual callable function

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
