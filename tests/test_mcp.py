import asyncio
from typing import Dict, Any
from gemma_mcp import GemmaMCPClient


# Example usage
async def example():
    # Example function with type hints and docstring
    async def get_weather(location: str, units: str = "celsius") -> Dict[str, Any]:
        """Get weather information for a location.

        Args:
            location: The city or location to get weather for
            units: Temperature units (celsius or fahrenheit)
        """
        # Dummy implementation
        return {"temp": 20, "location": location, "units": units}

    # MCP configuration example
    mcp_config = {
        "mcpServers": {
            "users": {"command": "python", "args": ["./test_server.py"]},
        }
    }

    # Initialize client with MCP support
    async with GemmaMCPClient(mcp_config=mcp_config).managed() as client:
        # Add local function
        client.add_function(get_weather)

        # Chat and get response with automatic function execution
        # This will now be able to use both local functions and MCP tools
        response = await client.chat(
            "What's the weather like in London?", execute_functions=True
        )
        print(response)  # Will print the actual weather data

        # Example of using an MCP tool
        response = await client.chat("What is my name?", execute_functions=True)
        print(response)  # Will use the assistant MCP tool if available


def test_mcp():
    asyncio.run(example())
