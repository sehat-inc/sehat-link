from typing import Dict, Any, Optional, Union, AsyncGenerator, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langsmith import traceable
from pydantic import BaseModel, Field

from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.prompts.mcp_client_prompts import program_eligibility_agent_prompt
from core.langgraph.utils.tool_manager import MCPToolManager
from core.logging import get_logger


logger = get_logger("PROGRAM AGENT")


class ProgramFeedbackAgent(BaseModel):
    response: str = Field(description="The calm and empathetic response to the user's query")
    baitul_maal_program_eligibility: str = Field(description="Based on the users response to whether he is eligible for Pakistan Bait Ul Maal Program. True | False | Not Mentioned")
    sehat_sahulat_program_eligibility: str = Field(description="Based on the users response to whether he is eligible for Sehat Sahulat Program. True | False | Not Mentioned")
    symptom_trigger: bool = Field(description="True if handoff to Symptoms agent should occur else false")
    doctor_trigger: bool = Field(description="True if handoff to Doctor agent should occur else false")

class ProgrammeEligibilityNode(Node):
    """
    Main Programme Agent that will record and summaarize findings of medical health programs and doctors as well as
    choose whether to call MCP Deep Research tool

    """
    def __init__(self,
                 name: str = "program_agent",
                 temperature: float = 0.7):
        super().__init__(name=name, temperature=temperature)
       
        self.ALLOWED_TOOLS = [
            "Programme_Eligibility_KB_Direct_Query",
            "Programme_Eligibility_KB_Smart_Query"
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

        logger.info(f"Allowed Tools for Programme: {[t.name for t in filtered_tools]}")


        model_with_tools = self.llm.bind_tools(filtered_tools)
        
        try:
            # Prepare Messages
            messages = list(state["messages"])
            logger.info("Succesfully Got messages")
            system_prompt = program_eligibility_agent_prompt(state)
            logger.info("Succesfully Got system prompt")
            system_message = SystemMessage(content=system_prompt)
            logger.info("Succesfully Got system prompt")
            messages = [system_message] + messages
            logger.info("Successfully Created messages v2 Prompts") 
        except Exception as e:
            logger.error(f"Error in Creating System Prompt and Conversation History: {e}")
            messages = ""
        
        try:
            tool_llm_response = await model_with_tools.ainvoke(messages)
        except Exception as e:
            logger.error(f"Failed in Getting LLM TOOL Response: {e}")
            tool_llm_response = "Due to technical issues could you please repeat that..."

        last_user_msg = messages[-1].content if messages else ""
        
        structured_prompt = f"""
        You are responsible for Parsing All information given into the correct
        format. You are also tasked with figuring out the routing as structured output
        determined by the conversation given.
        RULES:
            - Set `"symptom_trigger": true` ONLY if the user mentions about his symptoms ONLY or if he explicitly asks for help about his health and requires medical knowledge.

        USER RESPONSE: {last_user_msg}
        ASSISTANT RESPONSE: {tool_llm_response}
        "response": "The response given the LLM lastly" 
        "symptom_trigger": true | false
        "sehat_sahulat_program_eligibility": true | false | not mentioned - Based on if User Explicitly states he is Eligible or not for Sehat Sahulat Program
        "baitul_maal_program_eligibility": true | false | not mentioned - Based on if User Explicitly states he is Eligible or not for Pakistan Bait Ul Maal Program

        """

        try:
            struct_system_message = [SystemMessage(content=structured_prompt)] + list(state["messages"])
            structured_llm = self.llm.with_structured_output(
                schema=ProgramFeedbackAgent
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
        symptom_trigger = parsed.get("symptom_trigger", False)
        doctor_trigger = parsed.get("doctor_trigger", False)
        sehat_sahulat_program_eligibility = parsed.get("sehat_sahulat_program_eligibility", "True")
        baitul_maal_program_eligibility = parsed.get("baitul_maal_program_eligibility", "True")
        
        if symptom_trigger == True or symptom_trigger == "True":
            delta["current_agent"] = "symptom_agent"
    
        if doctor_trigger == True or doctor_trigger == "True":
            delta["current_agent"] = "doctor_agent"
        
        delta["sehat_sahulat_program_eligibility"] = sehat_sahulat_program_eligibility
        delta["baitul_maal_program_eligibility"] = baitul_maal_program_eligibility
        delta["messages"] = [tool_llm_response]
        delta["user_messages"] = [
            AIMessage(
                content=response_text
            )
        ]

        return delta
