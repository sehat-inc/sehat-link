from langchain_mcp_adapters.client import MultiServerMCPClient
from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.tool_manager import MCPToolManager, MCPClientPool

from core.logging import get_logger

logger = get_logger("TOOL NODE")

async def mcp_tool_node(state: MedicalAgentState):
    """
    Custom MCP tools execution node with Enhanced Error Handling
    """

    # Extract Last Message
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        logger.warning("No tool call found in last message")
        return {"messages": [], "error_count": state.get("error_count", 0)}

    # Setup MCP

    pool = await MCPClientPool.get_instance()


    # Execute all Tools Calls
    tool_messages = await pool.execute_tool_calls(last_message.tool_calls)
    
    logger.info(f"TOOL RESPONSE: {tool_messages}")
    
    error_count = sum(1 for msg in tool_messages if hasattr(msg, "status") and msg.status == "error")
    
    # Update state
    return {
        "messages": tool_messages,
        "tool_call_count": state.get("tool_call_count", 0) + len(tool_messages),
        "error_count": state.get("error_count", 0) + error_count
    }
