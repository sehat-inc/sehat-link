from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from core.logging import get_logger

logger = get_logger("TOOL MANAGER")


class MCPToolManager:
    """
    Manages MCP Server Tools and provides access to Nodes
    """

    def __init__(self, client: MultiServerMCPClient):
        self.client = client
        self.tools_by_name = {}

    async def initialize(self):
        """
        Initializes the MCP Client
        """

        tools = await self.client.get_tools()
        self.tools_by_name = {tool.name: tool for tool in tools}
        logger.info(f"Initialized with {len(self.tools_by_name)} tools")
        return tools

    async def execute_tool_calls(self, tool_calls: list) -> list[ToolMessage]:
        """
        Execute multiple tool calls and return ToolMessages.

        Args:
            tool_calls: List of tool calls dicts from AIMessage

        Returns:
            List of ToolMessage objects with results
        """
        tool_messages = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            try:
                # Get Tool
                tool = self.tools_by_name.get(tool_name)

                if not tool:
                    error_msg = f"Tool '{tool_name}' not found. Available tools: {list(self.tools_by_name.keys())}"
                    logger.error(error_msg)
                    tool_messages.append(
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id,
                            name=tool_name,
                            status="error"
                        )
                    )
                    continue

                # Execute Tool
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                result = await tool.ainvoke(tool_args)
                
                # Create success ToolMessage
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                )
                logger.info(f"Tool {tool_name} completed successfully")

            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                logger.error(error_msg)
                tool_messages.append(
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="error"
                    )
                )
        
        return tool_messages

