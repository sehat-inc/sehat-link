from typing import AsyncIterator, Dict, Any, Tuple, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

from langchain_mcp_adapters.client import MultiServerMCPClient

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
                 temperature: float = 0.7
                 ):
        
        self.name = name
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature
        )
        


