import ast
import inspect
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union, Callable, Type, get_type_hints

from fastmcp import Client, FastMCP
from google import genai
from google.genai import types


@dataclass
class FunctionDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]
    callable: Optional[Callable] = None  # Store the actual callable if provided

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionDefinition":
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data["parameters"],
            required=data["parameters"].get("required", []),
        )

    @classmethod
    def from_callable(cls, func: Callable) -> "FunctionDefinition":
        """Create a FunctionDefinition from a callable object."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        parameters = {}
        required_params = []

        # Type mapping from Python types to JSON schema types
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            datetime: "string",
            date: "string",
            # Add more type mappings as needed
        }

        def get_type_info(param_type: Type) -> Dict[str, Any]:
            """Get JSON schema type info for a Python type."""
            # Handle Optional types
            if (
                getattr(param_type, "__origin__", None) is Union
                and type(None) in param_type.__args__
            ):
                actual_type = next(
                    t for t in param_type.__args__ if t is not type(None)
                )
                return get_type_info(actual_type)

            # Handle List types
            if getattr(param_type, "__origin__", None) is list:
                item_type = param_type.__args__[0]
                return {
                    "type": "array",
                    "items": {"type": type_map.get(item_type, "string")},
                }

            # Handle Dict types
            if getattr(param_type, "__origin__", None) is dict:
                return {"type": "object"}

            # Handle basic types
            base_type = type_map.get(param_type, "string")
            type_info = {"type": base_type}

            # Add format for special string types
            if param_type in (datetime, date):
                type_info["format"] = "date-time" if param_type is datetime else "date"

            return type_info

        # Process parameters
        for name, param in sig.parameters.items():
            param_type = type_hints.get(name, str)
            type_info = get_type_info(param_type)

            parameters[name] = type_info

            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                required_params.append(name)
            else:
                # Add default value to schema if available
                parameters[name]["default"] = param.default

        # Create the function definition
        return cls(
            name=func.__name__,
            description=func.__doc__ or "No description provided.",
            parameters={
                "type": "object",
                "properties": parameters,
                "required": required_params,
            },
            required=required_params,
            callable=func,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class GemmaMCPClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemma-3-27b-it",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        mcp_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Gemma Function Client.

        Args:
            api_key: Gemini API key. If not provided, will look for GEMINI_API_KEY env var
            model: Model to use, defaults to gemma-3-27b-it
            temperature: Generation temperature
            system_prompt: Custom system prompt. If None, a default will be used.
            mcp_config: MCP configuration dictionary with server definitions
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either directly or via GEMINI_API_KEY environment variable"
            )

        self.client = genai.Client(vertexai=False, api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.functions: List[FunctionDefinition] = []

        # Initialize MCP client if config is provided
        self.mcp_client = MCPFunctionClient(mcp_config) if mcp_config else None

        # Default system prompt for Python-style function calling
        self._system_prompt = (
            system_prompt
            or """You are a helpful AI assistant.

You have access to functions. If you decide to invoke any of the function(s),
you MUST put it in the format of:

```tool_code
[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
```

You SHOULD NOT include any other text in the response if you call a function.
If you don't need to call any functions, respond normally.
"""
        )

    def start_chat(self, history: list = []):
        return self.client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(temperature=self.temperature),
        )

    def add_function(
        self, function: Union[Dict[str, Any], FunctionDefinition, Callable]
    ) -> None:
        """
        Add a function definition to the client.

        Args:
            function: Can be one of:
                - A dictionary containing the function definition
                - A FunctionDefinition object
                - A callable object (function or method)
        """
        if isinstance(function, dict):
            function = FunctionDefinition.from_dict(function)
        elif callable(function):
            function = FunctionDefinition.from_callable(function)
        elif not isinstance(function, FunctionDefinition):
            raise ValueError("Function must be a dict, FunctionDefinition, or callable")

        self.functions.append(function)

    def _build_prompt(self) -> str:
        """Build the complete system prompt including function definitions."""
        functions_json = json.dumps([f.to_dict() for f in self.functions], indent=2)
        return f"{self._system_prompt}\n\n{functions_json}"

    def _parse_function_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse the function calls from the model's response."""
        # If response doesn't look like a function call, return empty list
        if not response.strip().startswith(
            "```tool_code"
        ) and not response.strip().startswith("["):
            return []

        try:
            # Clean up the response to make it valid Python
            # strip potential function call markers
            cleaned_response = (
                response.strip().lstrip("```tool_code").rstrip("```").strip()
            )

            # Parse as Python expression
            parsed = ast.parse(cleaned_response, mode="eval")
            if not isinstance(parsed.body, ast.List):
                return []

            function_calls = []

            # Process each function call in the list
            for node in parsed.body.elts:
                if not isinstance(node, ast.Call):
                    continue

                func_name = node.func.id
                args_dict = {}

                # Handle keyword arguments
                for kw in node.keywords:
                    # Try to safely evaluate the value
                    try:
                        value = ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        # If we can't evaluate it, use the source text
                        value = ast.unparse(kw.value)
                    args_dict[kw.arg] = value

                # Handle positional arguments if any (convert to named args using function definition)
                if node.args:
                    matching_func = next(
                        (f for f in self.functions if f.name == func_name), None
                    )
                    if matching_func:
                        required_params = matching_func.required
                        for i, arg in enumerate(node.args):
                            if i < len(required_params):
                                try:
                                    value = ast.literal_eval(arg)
                                except (ValueError, SyntaxError):
                                    value = ast.unparse(arg)
                                args_dict[required_params[i]] = value

                function_calls.append({"name": func_name, "arguments": args_dict})

            return function_calls

        except (SyntaxError, AttributeError, ValueError) as e:
            # If parsing fails, try the old regex method as fallback
            pattern = r"(\w+)\((.*?)\)"
            matches = re.findall(pattern, response)

            function_calls = []
            for func_name, args_str in matches:
                # Parse arguments
                args_dict = {}
                if args_str.strip():
                    current_key = None
                    current_value = []
                    in_string = False
                    string_char = None

                    for char in args_str + ",":  # Add comma to handle last argument
                        if char in "\"'":
                            if not in_string:
                                in_string = True
                                string_char = char
                            elif char == string_char:
                                in_string = False
                        elif char == "," and not in_string:
                            if current_key and current_value:
                                args_dict[current_key.strip()] = "".join(
                                    current_value
                                ).strip()
                            current_key = None
                            current_value = []
                        elif char == "=" and not in_string and not current_key:
                            current_key = "".join(current_value)
                            current_value = []
                        else:
                            current_value.append(char)

                function_calls.append({"name": func_name, "arguments": args_dict})

            return function_calls

    async def initialize(self) -> None:
        """Initialize the client and all its components."""
        if self.mcp_client:
            await self.mcp_client.initialize()
            # Register MCP tools as functions
            await self._register_mcp_tools()

    async def cleanup(self) -> None:
        """Clean up all resources."""
        if self.mcp_client:
            await self.mcp_client.cleanup()

    async def _register_mcp_tools(self) -> None:
        """Register all discovered MCP tools as functions."""
        if not self.mcp_client:
            return

        for tool_name, tool_info in self.mcp_client.tools.items():
            tool = tool_info["tool"]

            # Convert MCP tool schema to FunctionDefinition format
            function_def = FunctionDefinition(
                name=tool_name,
                description=tool.description or "No description provided.",
                parameters={
                    "type": "object",
                    "properties": tool.inputSchema.get("properties", {}),
                    "required": tool.inputSchema.get("required", []),
                },
                required=tool.inputSchema.get("required", []),
            )

            self.functions.append(function_def)

    async def execute_function(self, func_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a registered function if it has a callable.

        Args:
            func_name: Name of the function to execute
            arguments: Arguments to pass to the function

        Returns:
            The result of the function execution
        """
        # Check if it's an MCP tool
        if self.mcp_client and func_name in self.mcp_client.tools:
            return await self.mcp_client.execute_tool(func_name, arguments)

        # Handle local functions
        func_def = next((f for f in self.functions if f.name == func_name), None)
        if not func_def or not func_def.callable:
            raise ValueError(f"No callable found for function {func_name}")

        return (
            await func_def.callable(**arguments)
            if inspect.iscoroutinefunction(func_def.callable)
            else func_def.callable(**arguments)
        )

    async def chat(
        self, message: str, execute_functions: bool = False, chat_session=None
    ) -> Union[str, List[Dict[str, Any]], Any]:
        """
        Send a message and get either a normal response or function calls.

        Args:
            message: The user message
            execute_functions: If True, automatically execute function calls and return their results

        Returns:
            Either:
            - A string response
            - A list of function calls
            - Results of function execution (if execute_functions=True)
        """

        if not chat_session:
            chat_session = self.start_chat()
            system_prompt = self._build_prompt()
            message = f"{system_prompt}\n\n{message}"
        response = chat_session.send_message(message)

        text = response.text

        # Check if the response is a function call
        function_calls = self._parse_function_calls(text)
        if function_calls:
            if execute_functions:
                results = []
                for func_call in function_calls:
                    result = await self.execute_function(
                        func_call["name"], func_call["arguments"]
                    )
                    results.append({"name": func_call["name"], "result": result})
                # return results[0] if len(results) == 1 else results
                return await self.chat(
                    json.dumps(results, indent=2),
                    execute_functions=execute_functions,
                    chat_session=chat_session,
                )
            else:
                return function_calls

        return text

    @asynccontextmanager
    async def managed(self):
        """Context manager for client lifecycle."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()


class MCPFunctionClient:
    """Client for managing MCP server connections and tool discovery."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MCP Function Client.

        Args:
            config: MCP configuration dictionary with server definitions
        """
        self.config = config or {}
        self.clients: Dict[str, Client] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connections to all configured MCP servers."""
        if self._initialized:
            return

        for server_name, server_config in self.config.get("mcpServers", {}).items():
            try:
                temp_server_config = {"mcpServers": {server_name: server_config}}
                client = Client(temp_server_config)

                async with client:
                    # Discover tools for this server
                    tools = await client.list_tools()
                self.clients[server_name] = client

                for tool in tools:
                    tool_name = f"{server_name}_{tool.name}"
                    self.tools[tool_name] = {"server": server_name, "tool": tool}
            except Exception as e:
                print(f"Failed to initialize MCP server {server_name}: {e}")

        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up all MCP server connections."""
        for client in self.clients.values():
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error cleaning up MCP client: {e}")
        self.clients.clear()
        self.tools.clear()
        self._initialized = False

    def get_tool_definition(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get the definition of an MCP tool."""
        return self.tools.get(tool_name)

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute an MCP tool.

        Args:
            tool_name: Name of the tool to execute (format: server_name_tool_name)
            arguments: Arguments to pass to the tool

        Returns:
            The result of the tool execution
        """
        if not self._initialized:
            await self.initialize()

        tool_info = self.tools.get(tool_name)
        if not tool_info:
            raise ValueError(f"Unknown MCP tool: {tool_name}")

        server_name = tool_info["server"]
        client = self.clients.get(server_name)
        if not client:
            raise ValueError(f"No client found for server: {server_name}")

        try:
            async with client:
                result = await client.call_tool(tool_info["tool"].name, arguments)
                # print(result, type(result))
                if result:
                    return result[0].text
                else:
                    return ""

        except Exception as e:
            raise RuntimeError(f"Failed to execute MCP tool {tool_name}: {e}")

    @asynccontextmanager
    async def managed(self):
        """Context manager for MCP client lifecycle."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()

    def add_test_server(self, server: FastMCP) -> None:
        """
        Add a test server for in-memory testing.

        Args:
            server: FastMCP server instance
        """
        client = Client(server)
        self.clients["test"] = client
        self._initialized = False  # Force re-initialization to discover tools
