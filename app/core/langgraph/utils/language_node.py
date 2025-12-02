from typing import Dict, Any

from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.helper import safe_str
from core.prompts.mcp_client_prompts import language_detector_prompt
from core.logging import get_logger

logger = get_logger("LANGUAGE_NODE")


class LanguageDetectorNode(Node):
    """
    Language Detector Node 
    Checks for languages and type of converstation style    
    """
    
    def __init__(self, name: str = "language_detector", temperature: float = 0.3):
        super().__init__(name=name, temperature=temperature) 
        
        self.SIMPLE_GREETINGS = {
            'hi', 'hello', 'hey', 'asalam', 'salam', 'namaste', 'kia haal hain',
            'kaisay ho'
        }
    
    async def run(self, state: MedicalAgentState) -> Dict[str, Any]:
        
        
        if state.get("preferred_language"):
            logger.info(f"Language already detected: {state['preferred_language']}")
            return {}

        messages = state.get("messages", [])
        if len(messages) <= 2:  # System + first user message
            logger.info("Skipping language detection - too early in conversation")
            return {"preferred_language": "English"}

        last_message = safe_str(messages[-1].content).lower().strip()
        if len(last_message.split()) <= 3:

            if any(greeting in last_message for greeting in self.SIMPLE_GREETINGS):
                logger.info("Simple greeting detected - defaulting to English")
                return {"preferred_language": "English"}

        recent_messages = messages[-5:]
        text = "\n".join(safe_str(m.content) for m in recent_messages)

        if len(text.strip()) < 10:
            logger.warning("Insufficient content for language detection")
            return {"preferred_language": "English"}
        
        logger.info(f"Running language detection on {len(recent_messages)} messages...")
        query_prompt = language_detector_prompt(text)
        response = await self.llm.ainvoke(query_prompt)

        logger.info(f"Language detected: {response}")
        return {"preferred_language": response}


       
