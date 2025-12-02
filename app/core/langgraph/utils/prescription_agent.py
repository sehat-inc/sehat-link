import json
from typing import Dict, Any
from langchain_core.messages import AIMessage

from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.helper import safe_str
from core.logging import get_logger

logger = get_logger("PRESCRIPTION AGENT")


class PrescriptionAgent(Node):
    """
    Detects prescription images, extracts medications (name, dose, freq, duration),
    cleans malformed LLM JSON, and updates MedicalAgentState safely.
    """

    def __init__(self, name: str = "prescription_agent", temperature: float = 0.0):
        super().__init__(name=name, temperature=temperature)

    async def run(self, state: MedicalAgentState) -> Dict[str, Any]:
        msgs = state.get("messages", [])
        if not msgs:
            return {}

        image_data = None 
        for msg in reversed(msgs):
            if msg.__class__.__name__ == "HumanMessage":
                if hasattr(msg, "content") and isinstance(msg.content, list):
                    logger.info(f"PRESCRIPTION MESSAGE BEING SEEN: {msg}")
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_data = item.get("image_url")
                            break
        

        if not image_data:
            return {}   # No image → do nothing

        system_prompt = """
        You are a clinical prescription OCR expert.
        Extract ONLY the medications list in this exact JSON shape:
        {
            "medications": [
                {"name": "...", "dose": "...", "frequency": "...", "duration": "..."}
            ]
        }
        Stricktly follow this json structure everytime you analyse the image.
        Intelligently infer dose, frequency and duration if not given clearly but never fill fake data into json.
        If unsure about any field, set it to null.
        Return JSON only.
        """


        try:
            # Replace the LLM call section with:
            raw_resp = await self.llm.ainvoke([
            ("system", system_prompt),
            ("human", [
                {"type": "text", "text": "Extract medications."},
                {"type": "image_url", "image_url": image_data}  
            ]) # type: ignore
        ])
            llm_output = safe_str(raw_resp.content)
        except Exception as e:
            logger.error(f"Prescription Agent Error: {e}")
            return {
                "messages": [AIMessage(content=f"No medications found")],
            }

        json_str = llm_output.strip()

        # Remove markdown fences
        if json_str.startswith("```"):
            json_str = json_str.lstrip("`")
        if "```" in json_str:
            json_str = json_str.replace("```", "")

        # Extract substring between first { and last }
        start = json_str.find("{")
        end = json_str.rfind("}")

        parsed: Dict[str, Any] = {"medications": []}

        if start != -1 and end != -1:
            try:
                candidate = json.loads(json_str[start:end + 1])
                # ensure expected shape
                meds = candidate.get("medications", [])
                if isinstance(meds, list):
                    parsed["medications"] = meds
            except Exception:
                # fall back silently
                pass

        if parsed["medications"]:
            med_list = "\n".join([
                f"- {m.get('name', 'Unknown')} ({m.get('dose', 'N/A')}, {m.get('frequency', 'N/A')}, {m.get('duration', 'N/A')})"
                for m in parsed["medications"]
            ])
            response_msg = f"I've extracted these medications from your prescription:\n{med_list}"
        else:
            response_msg = "I couldnt extract any medications from this image. Please ensure its a clear prescription photo."

        return {
            "prescription_data": parsed,
            "prescription_processed": True,
            "messsages": [AIMessage(content=response_msg)],
            "user_messages": [AIMessage(content=response_msg)]
        }
