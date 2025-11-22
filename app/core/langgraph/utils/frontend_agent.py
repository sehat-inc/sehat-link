from typing import Dict, Any, Union, AsyncGenerator
import json
from langsmith import traceable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_mcp_adapters.client import MultiServerMCPClient

from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.helper import safe_str
from core.prompts.mcp_client_prompts import frontend_agent_prompt
from core.logging import get_logger
from core.langgraph.utils.helper import (
    safe_str
)
import re
import json


class FrontendFeedback(BaseModel):
    response: str = Field(description="The calm and empathetic response to the user's query")
    symptom_trigger: bool = Field(description="True if handoff to Symptom agent should occur else false")
    programme_trigger: bool = Field(description="True if handoff to Programme agent should occur else false")
    doctor_trigger: bool = Field(description="True if handoff to Doctor agent should occur else false")


logger = get_logger("FRONTEND AGENT")

class TriageAgent(Node):
    """
    Main Triage Agent that will initialize conversation with the user.
    """
    def __init__(self,
                 name: str = "triage_agent",
                 temperature: float = 0.7):
        super().__init__(name=name, temperature=temperature)
    
        self.ALLOWED_TOOLS = [
            "Baba_Qadeer_Tool"
        ]



    @traceable
    async def run(self, state: MedicalAgentState):
        """
        Execution Logic
        """
        delta = {}
        
        client = MultiServerMCPClient({
            "sehat-link": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            }
        })

        # Bind Tools with Model (VERY IMPORTANT)
        tools = await client.get_tools()

        filtered_tools = [
            tool for tool in tools
            if tool.name in self.ALLOWED_TOOLS
        ]

        logger.info(f"Allowed Tools for Program: {[t.name for t in filtered_tools]}")
        
        model_with_tools = self.llm.bind_tools(filtered_tools)
        

        # Prepare Messages
        messages = list(state["user_messages"])
        system_prompt = frontend_agent_prompt(state)
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            system_message = SystemMessage(content=system_prompt)
            init_messages = [system_message] + messages
        else:
            init_messages = [SystemMessage(content=system_prompt)] + messages

        response = await model_with_tools.ainvoke(init_messages)
    
        logger.info(f"First LLM: {response}")

        # Last User Message 
        last_user_msg = messages[-1].content if messages else ""
        
        routing_prompt = f"""
        # ROLE: Backend Logic Parser & Router
        You are the **Navigation Controller** for the Sehat Link system. 
        Your job is to analyze the interaction between the **User** and **Ms Sehat** (Receptionist) to extract the clean response and determine the immediate routing destination.

        # INPUT DATA
        **USER'S LAST MESSAGE:** "{last_user_msg}"
        **ASSISTANT'S RAW RESPONSE:** "{response}"

        # 1. RESPONSE EXTRACTION RULE
        - Extract the **clean conversational text** spoken by the assistant.
        - **REMOVE** any XML tags (like `<router>`, `<response>`), JSON blocks, or internal thought processes.
        - If the Assistant's response was empty (because it was just routing), return an empty string "".

        # 2. ROUTING LOGIC (Strict Hierarchy)
        Analyze the **User's Last Message** to determine the trigger. Only **ONE** trigger can be True.

        ### A. symptom_trigger (Priority: High)
        - **Set TRUE if:**
            - User describes **physical symptoms, pain, or distress**.
            - User asks for a **medical opinion** ("Is this dangerous?", "What is this?").
            - *Examples:* "Mera sar dard hai", "I have fever", "Feeling dizzy", "chest pain".
        - **Set FALSE if:**
            - User mentions a condition ONLY to find a location (e.g., "I have fever, where is the hospital?" -> This is a Facility Search).

        ### B. programme_trigger (Priority: Medium)
        - **Set TRUE if:**
            - **FACILITY SEARCH:** User asks to find/locate a **hospital, clinic, pharmacy, lab, or Basic Health Unit**.
            - **PROGRAMS:** User asks about **Sehat Sahulat Card, Bait-ul-Maal, Govt schemes, or eligibility**.
            - **FINANCIAL:** User mentions **affordability/money** ("I can't afford this", "Free treatment").
            - *Examples:* "Where is the nearest hospital?", "Find pharmacy", "Sehat card check", "Cheap clinic".

        ### C. doctor_trigger (Priority: Low)
        - **Set TRUE if:**
            - User explicitly asks to **book an appointment** with a specific doctor.
            - User asks to **connect/speak to a human doctor** (Tele-health).
            - *Examples:* "Book appointment with Dr. Ali", "Connect me to a real person", "Schedule visit".
        - **Set FALSE if:**
            - User is just browsing for lists of doctors (Route to Programme/Facility agent instead).

        # 3. OUTPUT FORMAT (JSON)
        Return a valid JSON object:

        {{
            "response": "Clean text of assistant response (or empty string)",
            "symptom_trigger": true | false,
            "programme_trigger": true | false,
            "doctor_trigger": true | false
        }}

        # FEW-SHOT REASONING (For Accuracy)

        **Ex 1: Facility Lookup (Goes to Programme Agent)**
        *User:* "Qareebi hospital kahan hai?"
        *Reasoning:* User wants a facility location.
        *Result:* {{"symptom_trigger": false, "programme_trigger": true, "doctor_trigger": false}}

        **Ex 2: Symptom Complaint (Goes to Symptom Agent)**
        *User:* "Mujhay subah se ulti aa rahi hai." (Vomiting since morning)
        *Reasoning:* User is describing a medical condition.
        *Result:* {{"symptom_trigger": true, "programme_trigger": false, "doctor_trigger": false}}

        **Ex 3: Booking Request (Goes to Doctor Agent)**
        *User:* "Please book a slot with Dr. Ayesha."
        *Reasoning:* Explicit booking intent.
        *Result:* {{"symptom_trigger": false, "programme_trigger": false, "doctor_trigger": true}}

        **Ex 4: Financial/Govt Help (Goes to Programme Agent)**
        *User:* "I am poor, do you have free service?"
        *Reasoning:* Financial aid query.
        *Result:* {{"symptom_trigger": false, "programme_trigger": true, "doctor_trigger": false}}

        Parse the input now.
        """
        
        try:
            structured_llm = self.llm.with_structured_output(
                schema=FrontendFeedback.model_json_schema(), method="json_schema"
            )
            struct_messages = messages + [routing_prompt]
            response = await structured_llm.ainvoke(struct_messages)
            
            logger.info(f"SECOND LLM RESPONSE: {response}")
            
        except Exception as e:
            print(f"Frontend Error: {e}")
            response = "I'm sorry, I encountered a technical issue. Could you please repeat that?"
    

        symptom_trigger = response["symptom_trigger"] 
        programme_trigger = response["programme_trigger"] 
        doctor_trigger = response["doctor_trigger"] 
        logger.info(f"TRIAGE TRIGGERS-----------\nSYMPTOM: {symptom_trigger}\nPROGRAM: {programme_trigger}\nDOCTOR: {doctor_trigger}")
       

        response_text = response["response"]
        
        if symptom_trigger == True or symptom_trigger == "True":
            delta["current_agent"] = "symptom_agent"
            
            return delta

        if programme_trigger == True or programme_trigger == "True":
            delta["current_agent"] = "programme_eligibility_agent"
            
            return delta
        
        if doctor_trigger == True or doctor_trigger == "True":
            delta["current_agent"] = "doctor_agent"
        
            return delta
        
        delta["current_agent"] = "triage_agent"
        delta["messages"] = [
            AIMessage(
                content=response_text
            )
        ]
        delta["user_messages"] = [
            AIMessage(
                content=response_text
            )
        ]

        
        return delta
