from typing import Any, List
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport



class MCPToolManager:
    """
    Manages MCP Server Tools and provides access to Nodes
    """

    def __init__(self, mcp_url: str = "http://localhost:8000/mcp"):
        self.transport = StreamableHttpTransport(mcp_url)
        self.client = Client(self.transport)
        self.tools_cache = None

    async def initialize(self):
        """
        Initializes the MCP Client
        """
        await self.client.__aenter__()
        return self

    async def close_client(self):
        """
        Closes the MCP Client
        """
        await self.client.__aexit__(None, None, None)

    async def get_available_tools(self) -> List[str]:
        """
        Get List of available tools
        """
        if self.tools_cache is None:
            tools = await self.client.list_tools()
            self.tools_cache = [tool.name for tool in tools]
        return self.tools_cache

    async def call_tool(self, tool_name: str, arguments: Any): 
        """
        Calling MCP Server Tools
        """
        result = await self.client.call_tool(
            name=tool_name,
            arguments=arguments
        )
        return result


    
