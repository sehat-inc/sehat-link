from typing import Optional, List
from langchain.messages import HumanMessage
from langchain_core.messages import AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from pydantic import BaseModel, Field

from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.prompts.mcp_client_prompts import doctor_finder_agent_prompt
from core.logging import get_logger


logger = get_logger("DOCTOR AGENT")


class Doctor(BaseModel):
    doctor_name: Optional[str] = Field(description="The name of the doctor user has agreed to meet/call")
    doctor_specialization: Optional[str] = Field(description="The profession/specialization of the doctor the user has agreed to meet/call")


class DoctorFeedback(BaseModel):
    response: str = Field(description="The calm and empathetic response to the user's query")
    doctor: Optional[List[Doctor]] 
    symptom_trigger: bool = Field(description="True if handoff to Symptoms agent should occur else false")
    programme_trigger: bool = Field(description="True if handoff to Programme agent should occur else false")
    call_trigger : bool = Field(description="True if User has explicitly Agreed to Call Doctor else False")

class DoctorAgentNode(Node):
    """
    Main Doctor Agent 
    choose whether to call MCP Deep Research tool

    """
    def __init__(self,
                 name: str = "doctor_agent",
                 temperature: float = 0.7):
        super().__init__(name=name, temperature=temperature)
       
        self.ALLOWED_TOOLS = [
            "Doctor_KB_Smart_Query"
        ]

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

        logger.info(f"Allowed Tools for Doctor: {[t.name for t in filtered_tools]}")


        model_with_tools = self.llm.bind_tools(filtered_tools)
        
        try:
            # Prepare Messages
            messages = list(state["messages"])
            system_prompt = doctor_finder_agent_prompt(state)
            system_message = SystemMessage(content=system_prompt)
            messages = [system_message] + messages
            logger.info("Successfully Created messages Prompts")
        except Exception as e:
            logger.error(f"Error in Creating System Prompt and Conversation History: {e}")
            messages = ""
        
        try:
            tool_llm_response = await model_with_tools.ainvoke(messages)
            logger.info(f"DOCTOR TOOL LLM RESPONSE: {tool_llm_response}")
        except Exception as e:
            logger.error(f"Failed in Getting LLM TOOL Response: {e}")
            tool_llm_response = "Due to technical issues could you please repeat that..."

        last_user_msg = messages[-1].content if messages else ""
        
        structured_prompt = f"""
        You are responsible for parsing all information into the correct format and determining the routing based on the conversation.

        CRITICAL ROUTING RULES (Read carefully):

        1. **symptom_trigger = True** ONLY when:
           - The user's MOST RECENT message explicitly mentions NEW symptoms, health concerns, or medical issues
           - The user is asking for medical advice, diagnosis, or symptom information
           - Examples: "I have a headache", "my fever is getting worse", "what could cause this pain?"
           
           **symptom_trigger = False** when:
           - User is asking for doctor recommendations or referrals
           - User is discussing logistics (location, appointment booking, contact details)
           - User is responding to questions about their location, preferences, or availability
           - User is confirming or agreeing to see a doctor
           - Examples: "recommend me a doctor", "I'm in Multan", "yes, I want to see a doctor"

        2. **call_trigger = True** ONLY when:
           - Check the <call_trigger> in the Assistant Response for whether it is True or False
           - The user EXPLICITLY agrees to call or meet a specific doctor
           - The user confirms they want to proceed with contacting a doctor
           - Examples: "yes, call them", "I agree to meet Dr. Smith", "please schedule an appointment"
           
           **call_trigger = False** when:
           - User is just asking for recommendations without committing
           - User is still gathering information about doctors
           - Examples: "show me doctors", "who can I see?", "recommend a doctor"

        3. **programme_trigger = True** ONLY when:
           - The user asks about healthcare programs, financial assistance, or eligibility
           - The user mentions inability to afford treatment or asks about free services
           - Examples: "I can't afford this", "are there any free programs?", "do you have financial assistance?"
           
           **programme_trigger = False** for all other queries

        CURRENT CONVERSATION CONTEXT:
        USER'S MOST RECENT MESSAGE: {last_user_msg}
        ASSISTANT'S RESPONSE: {tool_llm_response}

        Based on the user's MOST RECENT message, determine the appropriate routing triggers.

        OUTPUT FORMAT:
        - response: The Response given by Assistant Response. Paste as is.(translate to Urdu if needed)
        - symptom_trigger: Boolean (true ONLY if user is discussing NEW symptoms/medical issues in their latest message)
        - programme_trigger: Boolean (true ONLY if user asks about programs/financial help in their latest message)
        - call_trigger: Boolean (true ONLY if user explicitly agrees to contact a doctor)
        - doctor_name: String (only if user agreed to meet/call a specific doctor)
        - doctor_specialization: String (only if user agreed to meet/call a doctor with known specialty)

        REMEMBER: Analyze the user's MOST RECENT message intent. If they're asking for doctor recommendations or location information, that is NOT a symptom discussion, so symptom_trigger should be FALSE.
        """
        
        try:
            struct_system_message = [HumanMessage(content=structured_prompt)]
            structured_llm = self.llm.with_structured_output(
                schema=DoctorFeedback
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
        programme_trigger = parsed.get("programme_trigger", False)
        doctor_collected = parsed.get("doctor") or []
        call_trigger = parsed.get("call_trigger", False)

        logger.info(f"SYMPTOM TRIGGERS-----------: \nPROGRAM: {programme_trigger}\nSYMPTOM: {symptom_trigger}")

        if symptom_trigger == True or symptom_trigger == "True":
            delta["current_agent"] = "symptom_agent"
    
        if programme_trigger == True or programme_trigger == "True":
            delta["current_agent"] = "programme_eligibility_agent"
        
        if doctor_collected and len(doctor_collected) > 0:
            delta["required_specialty"] = doctor_collected[0].get("doctor_specialization")
            delta["doctor_collected"] = doctor_collected
        else:
            delta["required_specialty"] = None
            delta["doctor_collected"] = []
        
        delta["call_trigger"] = call_trigger
        delta["messages"] = [tool_llm_response]
        delta["user_messages"] = [
            AIMessage(
                content=response_text
            )
        ]

        return delta

