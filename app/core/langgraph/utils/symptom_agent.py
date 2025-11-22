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
    location: Optional[str] = Field(description="Where the patient is feeling the symptom")
    additional_details: Optional[str] = Field(description="Any additional details about the symptom")
    

class SymptomAgentFeedback(BaseModel):
    response: str = Field(description="The calm and empathetic response to the user's query")
    symptoms: Optional[List[Symptom]]
    programme_trigger: bool = Field(description="True if handoff to Programme agent should occur else false")
    doctor_trigger: bool = Field(description="True if handoff to Doctor agent should occur else false")

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
        You are responsible for parsing all information into the correct format and determining routing triggers based on the conversation context.

        IMPORTANT: Always check the <action></action> tag in the ASSISTANT RESPONSE to understand the current flow state.

        ACTION TAG MEANINGS:
        - <action>continue_gathering</action>: Symptom agent is still collecting information → Keep ALL triggers FALSE
        - <action>use_research_tool</action>: Symptom agent is researching symptoms → Keep ALL triggers FALSE  
        - <action>offer_doctor_search</action>: Symptom agent is ASKING if user wants doctor recommendations → Keep ALL triggers FALSE (this is just an offer, not confirmation)

        CRITICAL ROUTING RULES:

        1. **doctor_trigger = True** ONLY when:
           - User EXPLICITLY agrees to find/see a doctor in their MOST RECENT message
           - User confirms they want doctor recommendations or referrals
           - User says "yes" to finding a doctor (after being asked)
           - Examples: "yes, find me a doctor", "haan, doctor recommend karein", "I want to see a doctor", "please help me find a doctor"
           
           **doctor_trigger = False** when:
           - Symptom agent is just offering to find a doctor (action = offer_doctor_search)
           - User is still describing symptoms
           - User hasn't explicitly agreed to finding a doctor yet
           - User is asking questions about their symptoms
           - Examples: "what's wrong with me?", "is this serious?", "I also have pain"

        2. **programme_trigger = True** ONLY when:
           - User asks about healthcare programs, financial assistance, or eligibility
           - User mentions inability to afford treatment
           - User asks about insurance, subsidies, or free services
           - Examples: "I can't afford this", "do you have free programs?", "insurance coverage", "financial help"
           
           **programme_trigger = False** for all other queries

        3. **symptom_trigger = True** ONLY when:
           - User mentions NEW symptoms in their most recent message
           - User needs to return to symptom discussion after being in doctor/program flow
           - Examples: "I also have a cough now", "my condition changed", "new symptom appeared"
           
           **symptom_trigger = False** when:
           - User is confirming doctor search
           - User is discussing logistics or location
           - Action tag shows continue_gathering or use_research_tool

        CURRENT CONVERSATION: USER'S MOST RECENT MESSAGE: {last_user_msg}
        ASSISTANT'S RESPONSE: {tool_llm_response}

        DECISION PROCESS:
        1. First, check the <action> tag in ASSISTANT RESPONSE
        2. If action is "continue_gathering" or "use_research_tool" → ALL triggers = False
        3. If action is "offer_doctor_search" → Check if USER explicitly agreed in their message
        4. Analyze USER'S MOST RECENT MESSAGE intent and keywords
        5. Set triggers accordingly

        OUTPUT STRUCTURE:
        {{
            "response": "Extract the text from <response></response> tags in ASSISTANT RESPONSE",
            "doctor_trigger": true/false,
            "programme_trigger": true/false,
            "symptom_trigger": true/false,
            "symptoms": [
                {{
                    "symptom": "name of symptom",
                    "severity": "mild/moderate/severe/unknown",
                    "duration": "timeframe or unknown",
                    "location": "body location or n/a",
                    "additional_details": "any extra context"
                }}
            ]
        }}

        CRITICAL REMINDERS:
        - "offer_doctor_search" is NOT the same as user agreeing → doctor_trigger stays FALSE until explicit user confirmation
        - Only the user's EXPLICIT agreement triggers doctor_trigger = True
        - When in doubt, keep triggers FALSE to avoid premature routing
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

        logger.info(f"SYMPTOM TRIGGERS-----------\: \nPROGRAM: {programme_trigger}\nDOCTOR: {doctor_trigger}")

        if programme_trigger == True or programme_trigger == "True":
            delta["current_agent"] = "programme_eligibility_agent"
                  
        
        if doctor_trigger == True or doctor_trigger == "True":
            delta["current_agent"] = "doctor_agent"
        
 
        delta["symptoms_collected"] = new_symptoms
        delta["messages"] = [tool_llm_response]
        delta["user_messages"] = [
            AIMessage(
                content=response_text
            )
        ]

        return delta
