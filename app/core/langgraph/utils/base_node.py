from typing import AsyncIterator, Dict, Any, Tuple, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

from core.langgraph.utils.helper import safe_str
from core.langgraph.utils.state import MedicalAgentState
from core.logging import get_logger
from core.langgraph.utils.tool_manager import MCPToolManager

logger = get_logger("NODE LOGIC")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



class Node:
    """
    Base Node Interface :) 
    
    Remarks:
        I did this solely because after re-writing this code 10 times did I realise that all nodes must be 
        uniform and I cant just create functions like in docs otherwise it will get extremely
        hard to scale when there are more than 5 agents
    """
    name: str

    def __init__(self, 
                 name: str, 
                 api_key: str = str(GEMINI_API_KEY), 
                 model: str = "gemini-2.5-flash", 
                 temperature: float = 0.7,
                 mcp_manager: Optional[MCPToolManager] = None,
                 allowed_tools: List[str] = None):
        
        self.name = name
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature,
            convert_system_message_to_human=True,
        )
        self.mcp_manager = mcp_manager
        self.allowed_tools = allowed_tools or []

    async def ainvoke(self, prompt: str, user_prompt: str = "") -> str:
        if user_prompt != "":
            message = [
                ("system", prompt),
                ("human", user_prompt)
            ]
            resp = await self.llm.ainvoke(message)

            return safe_str(resp.content)

        resp = await self.llm.ainvoke(prompt)
        return safe_str(resp.content)
    
    async def astream(self, messages: List[Tuple[str, str]]) -> AsyncIterator[str]:
       
        async for chunk in self.llm.astream(messages):
            yield safe_str(chunk.content)


    async def __call__(self, state: MedicalAgentState) -> Dict[str, Any]:
        raise NotImplementedError("Node Children must implement __call__")

    def has_tool_access(self, tool_name: str) -> bool:
        """Check if this node has access to a specific tool"""
        if not self.mcp_manager:
            return False
        if not self.allowed_tools:
            return False
        return tool_name in self.allowed_tools
    
    async def use_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Use an MCP tool if available"""
        if not self.has_tool_access(tool_name):
            raise ValueError(f"Node does not have access to tool: {tool_name}")
        
        return await self.mcp_manager.call_tool(tool_name, arguments) 

