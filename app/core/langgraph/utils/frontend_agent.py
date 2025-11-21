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
            "Programme_Eligibility_KB_Direct_Query",
            "Programme_Eligibility_KB_Smart_Query",
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
        
        routing_prompt=f"""
        You are responsible for analyzing the conversation and determining the routing as well 
        as structured output by parsing the main content of the information.
        Rules:
        - Set `"symptom_trigger": true` ONLY if the user mentions symptoms or a health concern.
        - Set `"programme_trigger": true` ONLY if the user mentions healthcare programmes, insurance, or eligibility.
        - Set `"doctor_trigger": true` ONLY if the user mentions he needs to find doctors or needs help finding relevant doctors.
        - If both are irrelevant → both should be false.
        - **You must NEVER set both to true at the same time.**

        USER LAST MESSAGE: {last_user_msg}

        ASSISTANT RESPONSE: {response}

        We need the output in the following format:

        {{
            "response": The response given by the assistant. Make sure the main text as is.
            "symptom_trigger": true | false
            "programme_trigger": true | false
            "doctor_trigger": true | false
        }}

        You do NOT need to mention routing or state changes to the user — just produce the correct structured output.
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
            delta["symptom_init"] = True
            delta["symptom_trigger"] = True
            
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
