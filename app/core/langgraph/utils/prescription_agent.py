import json
from typing import Dict, Any
from langchain_core.messages import AIMessage, BaseMessage

from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.helper import safe_str

class PrescriptionAgent(Node):
    """
    Detects prescription images, extracts medications (name, dose, freq, duration),
    cleans malformed LLM JSON, and updates MedicalAgentState safely.
    """

    def __init__(self, name: str = "prescription_agent", temperature: float = 0.0):
        super().__init__(name=name, temperature=temperature)

    async def __call__(self, state: MedicalAgentState) -> Dict[str, Any]:
        msgs = state.get("messages", [])
        if not msgs:
            return {}

        last_msg: BaseMessage = msgs[-1]

        # ------------------------------
        # 1. Detect if image is present
        # ------------------------------
        # In the PrescriptionAgent.__call__ method, replace the image detection logic with:
        image_data = None
        if hasattr(last_msg, "content") and isinstance(last_msg.content, list):
            for item in last_msg.content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_data = item.get("image_url")
                    break
        

        if not image_data:
            return {}   # No image → do nothing

        # ------------------------------
        # 2. Build Vision prompt
        # ------------------------------
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

        user_prompt = {
            "image": image_data,
            "text": "Extract medications."
        }

        # ------------------------------
        # 3. Call Gemini Vision Model
        # ------------------------------
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
            return {
                "messages": [AIMessage(content=f"PrescriptionAgent error: {e}")],
                "current_agent": "frontend"
            }

        # ------------------------------
        # 4. Clean & extract JSON safely
        # ------------------------------
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

        # ------------------------------
        # 5. Prepare delta update
        # ------------------------------
        delta: Dict[str, Any] = {}

        # Write final cleaned prescription result
        delta["prescription_data"] = parsed

        # Tell the rest of the system that we processed the prescription
        delta["messages"] = [
            AIMessage(content="Prescription processed successfully.")
        ]

        # Hand control back (you can choose a different next agent)
        delta["current_agent"] = "frontend"

        return delta
