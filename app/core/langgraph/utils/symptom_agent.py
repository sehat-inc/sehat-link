from typing import Dict, Any, Optional, Union, AsyncGenerator, List
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

class SymptomAgentNode(Node):
    """
    Main Symptom Agent that will record and summaarize symptoms as well as
    choose whether to call MCP Deep Research tool

    """
    def __init__(self,
                 name: str = "symptom_agent", 
                 temperature: float = 0.7): 
        super().__init__(name=name, temperature=temperature)
       
        

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
        model_with_tools = self.llm.bind_tools(tools)

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
        You are responsible for Parsing All information given into the correct
        format. You are also tasked with figuring out the routing as structured output
        determined by the conversation given.
        RULES:
            - Set `"programme_trigger": true` ONLY if the user mentions healthcare programmes, insurance, or eligibility.

        USER RESPONSE: {last_user_msg}
        ASSISTANT RESPONSE: {tool_llm_response}
        "response": "The response given the LLM lastly" 
        "programme_trigger": true | false
        {{
            "symptom": "headache",
            "severity": "moderate",
            "duration": "3 days",
            "location": "temples",
            "additional_details": "worse in morning"
        }}
        """

        try:
            struct_system_message = [SystemMessage(content=structured_prompt)] + list(state["messages"])
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

        if programme_trigger == True or programme_trigger == "True":
            delta["current_agent"] = "programme_eligibility_agent"
    
        
        delta["symptoms_collected"] = new_symptoms
        delta["messages"] = [tool_llm_response]
        delta["user_messages"] = [
            AIMessage(
                content=response_text
            )
        ]

        return delta
