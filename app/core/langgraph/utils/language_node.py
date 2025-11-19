from typing import Dict, Any

from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.helper import safe_str
from core.prompts.mcp_client_prompts import language_detector_prompt


class LanguageDetectorNode(Node):
    """
    Language Detector Node 
    Checks for languages and type of converstation style    
    """
    
    def __init__(self, name: str = "language_detector", temperature: float = 0.3):
       super().__init__(name=name, temperature=temperature) 

    async def __call__(self, state: MedicalAgentState) -> Dict[str, Any]:
        

        recent_messages = state["messages"][-5:]
        text = "\n".join(safe_str(m.content) for m in recent_messages)

        query_prompt = language_detector_prompt(text)

        response = await self.ainvoke(query_prompt)
        
        return {
            "detected_language": response,
        }
