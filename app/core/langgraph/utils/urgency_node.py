from typing import Dict, Any, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.prompts.mcp_client_prompts import urgency_detector_prompt
from core.logging import get_logger

logger = get_logger("URGENCY AGENT")


class UrgencyResponse(BaseModel):
    urgency_level: Literal["Emergency", "High", "Medium", "Low"] = Field(
        description="The urgency level based on symptoms"
    )
    reasoning: str = Field(description="Brief reasoning for the urgency level")


class UrgencyDetectorNode(Node):
    """
    Checks Urgency - RUNS ONLY ONCE when symptom agent hands off.
    
    This node analyzes all collected symptoms and red flags to determine
    the urgency level of the patient's condition.
    
    State:
        detected_urgency: Literal["Emergency", "High", "Medium", "Low"]
        urgency_checked: bool - Flag to prevent re-running
    """
    def __init__(self, name: str = "urgency_detector", temperature: float = 0.3):
        super().__init__(name=name, temperature=temperature)
        
    async def run(self, state: MedicalAgentState):
        
        # Check if urgency has already been determined this session
        if state.get("urgency_checked", False):
            logger.info("UrgencyDetectorNode skipped - already checked this session")
            return {}
        
        symptoms = state.get("symptoms_collected", [])
        red_flags = state.get("red_flags", [])
        disease_name = state.get("disease_name", "")

        # If no symptoms collected, set default and mark as checked
        if not symptoms and not red_flags:
            logger.info("UrgencyDetectorNode - no symptoms or red flags, setting Low urgency")
            return {
                "detected_urgency": "Low",
                "urgency_checked": True
            }
        
        # Build symptom summary for LLM
        symptom_lines = []
        for s in symptoms:
            line = f"- {s.get('symptom', 'unknown')}: severity={s.get('severity', 'unknown')}, duration={s.get('duration', 'unknown')}, location={s.get('location', 'unknown')}"
            symptom_lines.append(line)
        
        symptom_text = "\n".join(symptom_lines) if symptom_lines else "No specific symptoms recorded"
        red_flag_text = ", ".join(red_flags) if red_flags else "None"
        
        prompt = f"""You are a medical urgency classifier. Based on the patient's symptoms and red flags, determine the urgency level.

**Urgency Levels:**
- **Emergency**: Life-threatening symptoms requiring immediate attention (chest pain, difficulty breathing, severe bleeding, loss of consciousness, stroke symptoms)
- **High**: Serious symptoms that need prompt medical attention within hours (high fever with confusion, severe pain, signs of infection spreading)
- **Medium**: Symptoms that should be seen by a doctor soon but not immediately (persistent moderate pain, fever lasting days, concerning but stable symptoms)
- **Low**: Minor symptoms that can be managed with self-care or routine appointment (mild cold, minor aches, general wellness questions)

**Patient Information:**
Disease/Condition Identified: {disease_name or "Not yet identified"}

Symptoms:
{symptom_text}

Red Flags Detected: {red_flag_text}

Based on this information, classify the urgency level."""

        try:
            structured_llm = self.llm.with_structured_output(schema=UrgencyResponse)
            response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
            
            if response and hasattr(response, 'urgency_level'):
                urgency = response.urgency_level
                logger.info(f"Urgency determined: {urgency} - {response.reasoning}")
            else:
                # Fallback based on red flags
                urgency = "High" if red_flags else "Medium"
                logger.warning(f"Structured response failed, using fallback: {urgency}")
                
        except Exception as e:
            logger.error(f"UrgencyDetectorNode error: {e}")
            # Conservative fallback - if we have red flags, mark as High
            urgency = "High" if red_flags else "Medium"

        return {
            "detected_urgency": urgency,
            "urgency_checked": True
        }

