from typing import Dict, Any, Optional, Union, AsyncGenerator, List
from langsmith import traceable
from langchain.agents import create_agent
from langchain_core import messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json
from langchain_mcp_adapters.client import MultiServerMCPClient

from pydantic import BaseModel, Field

from langchain.tools import ToolRuntime
from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.helper import safe_str
from core.prompts.mcp_client_prompts import symptom_agent_prompt
from core.langgraph.utils.tool_manager import MCPToolManager
from core.logging import get_logger
from core.langgraph.utils.helper import (
    safe_str
)


logger = get_logger("SYMPTOM AGENT")


class Symptom(BaseModel):
    symptom: Optional[str] = Field(description="The name of the symptom")
    duration: Optional[str] = Field(description="The duration of how long the symptom have occured")
    severity: Optional[str] = Field(description="Severity: mild, moderate, severe, etc.")
    location: Optional[str] = Field(description="Where the patient is feeling the symptom")
    additional_details: Optional[str] = Field(description="Any additional details about the symptom")
     

class SymptomAgentFeedback(BaseModel):
    response: str = Field(description="The calm and empathetic response to the user's query")
    symptoms: Optional[List[Symptom]] = Field(default_factory=list)
    programme_trigger: bool = Field(description="True if handoff to Programme agent should occur else false")
    doctor_trigger: bool = Field(description="True if handoff to Doctor agent should occur else false")
    shared_facts: Optional[List[str]] = Field(default_factory=list, description="Shared facts for other agents")
    shared_warnings: Optional[List[str]] = Field(default_factory=list, description="Shared warnings for other agents")
    red_flags: Optional[List[str]] = Field(default_factory=list, description="Shared Red Flags for other agents")
    symptom_research_result: Optional[str] = Field(default_factory=str, description="The Research Results/Summary given by LLM if Tool is called.")

class SymptomAgentNode(Node):
    """
    Main Symptom Agent that will record and summaarize symptoms as well as
    choose whether to call MCP Deep Research tool

    """
    def __init__(self,
                 name: str = "symptom_agent", 
                 temperature: float = 0.7): 
        super().__init__(name=name, temperature=temperature)
       
        self.ALLOWED_TOOLS = [
            "Symptom_Knowledge_Base_Smart_Query",
            "Symptom_Knowledge_Base_Direct_Query",
        ]
    
    @traceable
    async def run(self, state: MedicalAgentState):
        """Main node execution"""
        
        client = MultiServerMCPClient({
            "sehat-link": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            }
        })


        delta = {}
        
        tools = await client.get_tools()
        
        filtered_tools = [
            tool for tool in tools
            if tool.name in self.ALLOWED_TOOLS
        ]

        logger.info(f"Allowed Tools for Symptom: {[t.name for t in filtered_tools]}")
        
        model_with_tools = self.llm.bind_tools(filtered_tools)

        # Prepare Messages
        messages = list(state["messages"])
        system_prompt = symptom_agent_prompt(state)
        system_message = SystemMessage(
            content=system_prompt
        )
        messages = [system_message] + messages

        tool_llm_response = await model_with_tools.ainvoke(messages)
        
        last_user_msg = messages[-1].content if messages else ""
        
        structured_prompt = f"""
    # ROLE: Medical Conversation Router & Data Parser

    You are the system logic engine for Sehat Link. Your job is to parse the output from "Nora" (the Symptom Agent) and the User's latest message to determine the next system state.

    You must map the unstructured XML/Text output into a strict JSON structure matching the `SymptomAgentFeedback` schema.

    # INPUTS
    1. **User's Last Message:** "{last_user_msg}"
    2. **Nora's (Agent) Response:** 
    {tool_llm_response}

    # INSTRUCTIONS

    ## 1. Data Parsing (XML to JSON)
    You must extract data from Nora's XML tags and map them to the output schema:

    - **response**: Extract text from `<response>...</response>`.
    - **symptoms**: Parse the JSON inside `<data_extraction>` -> `symptoms_collected`. Map fields:
        - `name` -> `symptom`
        - `severity` -> `severity` (if missing, put "unknown")
        - `duration` -> `duration`
        - `location` -> `location`
        - `details`/`type` -> `additional_details`
    - **shared_facts**: Extract from `<data_extraction>` -> `shared_facts`.
    - **shared_warnings**: Extract from `<data_extraction>` -> `shared_warnings`.
    - **red_flags**: Extract from `<data_extraction>` -> `red_flags`.
    - **symptom_research_result**: Extract text from `<symptom_research_result>`. 
      - **IMPORTANT:** The output schema requires this to be a **Dict**. 
      - Format it as: `{{ "summary": "extracted text..." }}`. If empty, use `{{"summary": null}}`.

    ## 2. Routing Logic (Triggers)
    Determine `doctor_trigger` and `programme_trigger`. 
    **DEFAULT TO FALSE** unless specific criteria are met.

    ### A. Doctor Trigger (`doctor_trigger`)
    **Set to TRUE only if:**
    1. The User **EXPLICITLY** asks for a doctor/specialist in `last_user_msg` (e.g., "find me a doctor", "I need to see someone", "book appointment").
    2. The User replies "Yes" to a previous offer to find a doctor.
    
    **Set to FALSE if:**
    - Nora's `<action>` is `offer_doctor_search` BUT the user has NOT said "yes" yet. (Nora is *offering*, not confirming).
    - Nora's `<action>` is `continue_gathering`, `call_smart_query`, or `call_direct_query`.
    - User is still describing symptoms.

    ### B. Programme Trigger (`programme_trigger`)
    **Set to TRUE only if:**
    1. User mentions financial difficulty (e.g., "cannot afford", "too expensive", "no money").
    2. User asks about government schemes, insurance, Sehat Card, or free clinics.
    
    **Set to FALSE otherwise.**

    # OUTPUT SCHEMA (JSON)
    Target class: `SymptomAgentFeedback`

    {{
        "response": "String",
        "symptoms": [List of Symptom objects],
        "programme_trigger": Boolean,
        "doctor_trigger": Boolean,
        "shared_facts": [List of strings],
        "shared_warnings": [List of strings],
        "red_flags": [List of strings],
        "symptom_research_result": {{ "summary": "String or Null" }}
    }}

    # EXAMPLES

    ## Example 1: Gathering Info (English)
    **User:** "I have a throbbing headache on the left side."
    **Nora Action:** `<action>continue_gathering</action>`
    **Nora Data:** `<data_extraction> {{ "symptoms_collected": [name": "headache", "severity": "severe", "location": "left side"] }} ...`

    **Output:**
    ```json
    {{
        "response": "I understand. How long have you had this headache?",
        "symptoms": [
            {{
                "symptom": "headache", 
                "duration": "unknown", 
                "location": "left side", 
                "additional_details": "severity: severe"
            }}
        ],
        "programme_trigger": false,
        "doctor_trigger": false,
        "shared_facts": [],
        "shared_warnings": [],
        "red_flags": [],
        "symptom_research_result": {{ "summary": null }}
    }}
    ```

    ## Example 2: Tool Use (Urdu/English)
    **User:** "Mujhe ajeeb se chakkar aa rahe hain drug lene ke baad." (I am feeling dizzy after taking drug).
    **Nora Action:** `<action>call_direct_query</action>`
    **Nora Data:** `<data_extraction> {{ "shared_warnings": ["potential drug reaction"] }} ...`

    **Output:**
    ```json
    {{
        "response": "Main check karti hoon ke yeh dawa ka reaction toh nahi.",
        "symptoms": [],
        "programme_trigger": false,
        "doctor_trigger": false,
        "shared_facts": [],
        "shared_warnings": ["potential drug reaction"],
        "red_flags": [],
        "symptom_research_result": {{ "summary": null }}
    }}
    ```

    ## Example 3: Explicit Doctor Handoff (Urdu)
    **User:** "Jee haan, please kisi doctor ko dikha dein." (Yes, please show to a doctor).
    **Nora Action:** `<action>offer_doctor_search</action>` (Nora acknowledges and prepares to switch).
    **Nora Data:** `<data_extraction> {{ "symptoms_collected": [...] }}`

    **Output:**
    ```json
    {{
        "response": "Theek hai, main aapko doctor dhoondne mein madad karti hoon.",
        "symptoms": [...],
        "programme_trigger": false,
        "doctor_trigger": true, 
        "shared_facts": [],
        "shared_warnings": [],
        "red_flags": [],
        "symptom_research_result": {{ "summary": null }}
    }}
    ```
    *(Note: `doctor_trigger` is true because User said "Jee haan" explicitly).*

    ## Example 4: Programme/Financial Handoff (English)
    **User:** "I really need help but I don't have any money for a private clinic."
    **Nora Action:** `<action>offer_doctor_search</action>`

    **Output:**
    ```json
    {{
        "response": "I understand your financial concern...",
        "symptoms": [],
        "programme_trigger": true,
        "doctor_trigger": false,
        "shared_facts": [],
        "shared_warnings": [],
        "red_flags": [],
        "symptom_research_result": {{ "summary": null }}
    }}
    ```
    *(Note: `programme_trigger` is true due to "don't have any money").*
    
    ## Example 5: Research Result Parsing
    **User:** (Silent - processing)
    **Nora Response:** `<symptom_research_result>Symptoms align with Gastritis.</symptom_research_result>`
    
    **Output:**
    ```json
    {{
        "response": "...",
        "symptoms": [],
        "programme_trigger": false,
        "doctor_trigger": false,
        "shared_facts": [],
        "shared_warnings": [],
        "red_flags": [],
        "symptom_research_result": {{ "summary": "Symptoms align with Gastritis." }}
    }}
    ```
    """

        try:
            struct_system_message = [HumanMessage(content=structured_prompt)] 
            structured_llm = self.llm.with_structured_output(
                schema=SymptomAgentFeedback
            )

            response = await structured_llm.ainvoke(struct_system_message)
        except Exception as e:
            logger.error(f"Structured LLM Not Working: {e}")
            delta["user_messages"] = [
                AIMessage(content="Could you please repeat that again?")
            ]
            delta["messages"] = []   # no tool routing in error case
            return delta

        if isinstance(response, BaseModel):
            parsed = response.dict()
        elif isinstance(response, dict):
            parsed = response
        else:
            raise ValueError(f"Unexpected response type: {type(response)} | {response}")

        response_text = parsed.get("response", "")
        new_symptoms = parsed.get("symptoms") or []
        programme_trigger = parsed.get("programme_trigger", False)
        doctor_trigger = parsed.get("doctor_trigger", False)
        shared_facts = parsed.get("shared_facts", [])
        shared_warnings = parsed.get("shared_warnings", [])
        red_flags = parsed.get("red_flags", [])
        symptom_research_result = parsed.get("symptom_research_result", {})
        

        logger.info(f"SYMPTOM TRIGGERS-----------: \nPROGRAM: {programme_trigger}\nDOCTOR: {doctor_trigger}")

        if programme_trigger == True or programme_trigger == "True":
            delta["current_agent"] = "programme_eligibility_agent"
                  
        
        if doctor_trigger == True or doctor_trigger == "True":
            delta["current_agent"] = "doctor_agent"
        
        delta["shared_facts"] = shared_facts
        delta["shared_warnings"] = shared_warnings
        delta["red_flags"] = red_flags
        delta["symptom_research_result"] = symptom_research_result
        delta["symptoms_collected"] = new_symptoms
        delta["messages"] = [tool_llm_response]
        delta["user_messages"] = [
            AIMessage(
                content=response_text
            )
        ]

        return delta
