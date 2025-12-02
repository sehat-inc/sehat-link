from typing import List, Optional, Dict, Any
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient, asyncio
from langchain_mcp_adapters.tools import MCPTool
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



class MCPClientPool:
    """
    Singleton Pattern Implemented Instead of the Previous constant reconnections with 
    the MCP Server

    """

    _instance: Optional['MCPClientPool'] = None
    _lock = asyncio.Lock()
    _initialized = False

    def __init__(self):
        """
        Create a Private constructor - use get_instance() Instead
        """

        if not hasattr(self, '_initialized'):
            self._initialized = False
            self.client: Optional[MultiServerMCPClient] = None
            self.tools_by_name: Dict[str, MCPTool] = {}
            self._all_tools: List[MCPTool] = []
            self._connection_healthy = False

    
    @classmethod
    async def get_instance(cls) -> 'MCPClientPool':
        """
        Get or create the singleton instance (thread-safe).
        
        Returns:
            Initialized MCPClientPool instance
        """
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:  # Double-check locking
                    # Create instance using object.__new__ to bypass __init__
                    instance = object.__new__(cls)
                    
                    # Manually set attributes
                    instance._initialized = False
                    instance.client = None
                    instance.tools_by_name = {}
                    instance._all_tools = []
                    instance._connection_healthy = False
                    
                    cls._instance = instance
                    
                    # Initialize the MCP client
                    await cls._instance._initialize_client()
        
        return cls._instance
    
    async def _initialize_client(self):
        """Initialize MCP client and load tools"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing MCP Client Pool...")
            
            # Create MCP client
            self.client = MultiServerMCPClient({
                "sehat-link": {
                    "transport": "streamable_http",
                    "url": "http://localhost:8000/mcp",
                }
            })
            
            # Load all available tools
            self._all_tools = await self.client.get_tools()
            self.tools_by_name = {tool.name: tool for tool in self._all_tools}
            
            self._connection_healthy = True
            self._initialized = True
            
            logger.info(f"MCP Client Pool initialized with {len(self.tools_by_name)} tools: "
                       f"{list(self.tools_by_name.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP Client Pool: {e}")
            self._connection_healthy = False
            raise
    
    async def get_tools(self, allowed_tools: Optional[List[str]] = None) -> List[MCPTool]:
        """
        Get tools, optionally filtered by allowed list.
        
        Args:
            allowed_tools: List of tool names to filter, or None for all tools
            
        Returns:
            List of MCPTool objects
        """
        if not self._initialized:
            await self._initialize_client()
        
        if allowed_tools is None:
            return self._all_tools
        
        # Filter tools efficiently using cached dict
        filtered = [
            self.tools_by_name[name] 
            for name in allowed_tools 
            if name in self.tools_by_name
        ]
        
        # Warn about missing tools
        missing = set(allowed_tools) - set(self.tools_by_name.keys())
        if missing:
            logger.warning(f"Requested tools not found: {missing}")
        
        return filtered
    
    async def get_tool_by_name(self, tool_name: str) -> Optional[MCPTool]:
        """
        Get a single tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            MCPTool if found, None otherwise
        """
        if not self._initialized:
            await self._initialize_client()
        
        return self.tools_by_name.get(tool_name)
    
    async def execute_tool_calls(self, tool_calls: list) -> List[ToolMessage]:
        """
        Execute multiple tool calls and return ToolMessages.
        
        Args:
            tool_calls: List of tool call dicts from AIMessage
            
        Returns:
            List of ToolMessage objects with results
        """
        if not self._initialized:
            await self._initialize_client()
        
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            
            try:
                # Get tool from cache (O(1) lookup)
                tool = self.tools_by_name.get(tool_name)
                
                if not tool:
                    error_msg = (f"Tool '{tool_name}' not found. "
                               f"Available tools: {list(self.tools_by_name.keys())}")
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
                
                # Execute tool
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
    
    async def health_check(self) -> bool:
        """
        Check if MCP client connection is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self._initialized:
            return False
        
        try:
            # Attempt to get tools as health check
            await self.client.get_tools()
            self._connection_healthy = True
            return True
        except Exception as e:
            logger.error(f"MCP health check failed: {e}")
            self._connection_healthy = False
            return False
    
    async def refresh_tools(self):
        """Refresh tool cache (useful if MCP server tools change)"""
        try:
            logger.info("Refreshing MCP tools cache...")
            self._all_tools = await self.client.get_tools()
            self.tools_by_name = {tool.name: tool for tool in self._all_tools}
            logger.info(f"Tools cache refreshed: {len(self.tools_by_name)} tools")
        except Exception as e:
            logger.error(f"Failed to refresh tools: {e}")
            raise

    def is_initialized(self) -> bool:
        """Check if pool is initialized"""
        return self._initialized
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy (cached status)"""
        return self._connection_healthy
    
    def get_available_tool_names(self) -> List[str]:
        """Get list of all available tool names"""
        return list(self.tools_by_name.keys())
    

# Convenience function for backward compatibility
async def get_mcp_client_pool() -> MCPClientPool:
    """Get the shared MCP Client Pool instance"""
    return await MCPClientPool.get_instance()


















