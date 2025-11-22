from typing import Dict, Any

from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.prompts.mcp_client_prompts import urgency_detector_prompt
from core.logging import get_logger

logger = get_logger("URGENCY AGENT")


class UrgencyDetectorNode(Node):
    """
    Checks Urgency 
    
    State:
        detected_urgency: Literal["Emergency", "High", "Medium", "Low"]
    """
    def __init__(self, name: str = "urgency_detector", temperature: float = 0.6):
        super().__init__(name=name, temperature=temperature)
        
    async def run(self, state: MedicalAgentState):
        
        symptoms = state.get("symptoms_collected", [])

        if not symptoms:
            logger.info("UrgencyDetectorNode skipped - no symptoms collected yet")
            # Return state unchanged
            return {"current_agent": "symptom_agent"}
        
        symptom_lines = []
        for s in state["symptoms_collected"]:
            line = f"""- Symptom: {s.get('symptom', 'unknown')},
                   Severity: {s.get('severity', 'unknown')},
                   Duration: {s.get('duration', 'unknown')},
                   Location: {s.get('location', 'unknown')}"""
            symptom_lines.append(line)

        symptom_text = "\n".join(symptom_lines)
        
        prompt = urgency_detector_prompt()
        
        message = f"{prompt} \nPatient symptoms: \n{symptom_text}"

        response = await self.llm.ainvoke(message)

        return {
            "detected_urgency": response,
        }

