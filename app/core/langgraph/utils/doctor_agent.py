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
    shared_facts: Optional[List[str]] = Field(default_factory=list, description="Shared facts for other agents")
    shared_warnings: Optional[List[str]] = Field(default_factory=list, description="Shared warnings for other agents")
    red_flags: Optional[List[str]] = Field(default_factory=list, description="Shared Red Flags for other agents")
    

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
        # ROLE: Conversation State Parser (Doctor Agent)
        You are responsible for parsing the conversation between a User and the Doctor/Facility Agent. 
        Your goal is to Extract Information and Determine Routing based on the User's **Intent**.

        # INPUT CONTEXT
        **User's Last Message:** "{last_user_msg}"
        **Assistant's Response:** "{tool_llm_response}"

        # 1. ROUTING LOGIC (CRITICAL)

        ### A. symptom_trigger (Handoff to Medical Diagnosis Agent)
        **definition:** Should we switch to the Medical Agent for diagnosis/advice?
        *   **Set TRUE if:**
            *   User asks for a **diagnosis**: "What does this pain mean?", "Is this dangerous?"
            *   User asks for **medical advice/remedies**: "What should I take?", "Home remedies for flu."
        *   **Set FALSE if (CRITICAL DISTINCTION):**
            *   User mentions symptoms **only to refine the doctor search**. 
            *   *Example:* "I have a heart problem, find me a doctor." -> **FALSE** (Keep in Doctor Agent to search for Cardiologists).
            *   *Example:* "My stomach hurts, recommend a specialist." -> **FALSE** (Keep in Doctor Agent to search for Gastroenterologists).

        ### B. programme_trigger (Handoff to Eligibility Agent)
        **definition:** Should we switch to the Program/Financial Aid Agent?
        *   **Set TRUE if:**
            *   User mentions **financial constraints**: "I cannot afford this", "It is too expensive".
            *   User asks about **government programs**: "Sehat Card", "Bait-ul-Maal", "Free treatment".
        *   **Set FALSE if:**
            *   User is just asking about doctor fees (standard inquiry).

        ### C. call_trigger (Conversion/Booking)
        **definition:** Has the user agreed to proceed with a specific doctor?
        *   **Set TRUE if:**
            *   User explicitly agrees to **contact/meet** a specific doctor recommended by the assistant.
            *   *Examples:* "Yes, call him", "Book the appointment", "I will visit Dr. Ali".
        *   **Set FALSE if:**
            *   User is asking for more options or details.

        # 2. INFORMATION EXTRACTION RULES

        ### A. Shared Facts (Permanent User Info)
        *   Extract concrete details that define the user's profile.
        *   *Include:* Name, City/Location, Phone Number, Explicit Medical Condition (e.g., "I am diabetic"), Budget constraints.

        ### B. Shared Warnings (Behavioral)
        *   Extract behavioral cues relevant to other agents.
        *   *Examples:* "User gets angry easily", "User prefers Urdu only", "User is impatient".

        ### C. Red Flags (Medical Emergencies)
        *   Detect life-threatening keywords.
        *   *Examples:* "Chest pain", "Unconscious", "Bleeding heavily", "Suicidal thoughts".

        # 3. OUTPUT FORMAT

        Return a Valid JSON object matching this structure:

        {{
            "response": "The Assistant's response text (clean text only)",
            "symptom_trigger": true | false,
            "programme_trigger": true | false,
            "call_trigger": true | false,
            "doctor_name": "Name of doctor if call_trigger is true, else null",
            "doctor_specialization": "Specialization if call_trigger is true, else null",
            "shared_facts": ["Fact 1", "Fact 2"],
            "shared_warnings": ["Warning 1"],
            "red_flags": ["Emergency Indicator"]
        }}

        # FEW-SHOT EXAMPLES (Mental Chain of Thought)

        **Ex 1: Symptom for Search (Stay with Doctor Agent)**
        *User:* "I have a severe skin rash, please find me a specialist."
        *Logic:* User has symptoms ("skin rash") but the **intent** is finding a doctor. Doctor Agent needs to use this info to search for Dermatologists. NOT a diagnosis request.
        *Output:* `symptom_trigger: false`, `shared_facts: ["User has skin rash"]`

        **Ex 2: Diagnosis Request (Switch to Symptom Agent)**
        *User:* "I have a skin rash. Is it contagious? What cream should I use?"
        *Logic:* User is asking for medical knowledge/advice.
        *Output:* `symptom_trigger: true`

        **Ex 3: Booking Agreement**
        *User:* "Dr. Sarah looks good. Please book an appointment with her."
        *Logic:* Explicit agreement.
        *Output:* `call_trigger: true`, `doctor_name: "Dr. Sarah"`

        **Ex 4: Financial Issue**
        *User:* "I can't pay private fees. Do you know any government hospitals?"
        *Logic:* Financial constraint + Gov hospital request.
        *Output:* `programme_trigger: true`

        Parse the current interaction now.
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

