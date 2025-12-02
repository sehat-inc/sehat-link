from typing import Dict, Any, Optional, Union, AsyncGenerator, List
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langsmith import traceable
from pydantic import BaseModel, Field
from core.langgraph.utils.tool_manager import MCPClientPool

from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.helper import extract_llm_content, extract_user_message
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
    shared_facts: Optional[List[str]] = Field(default_factory=list, description="Shared facts for other agents")
    shared_warnings: Optional[List[str]] = Field(default_factory=list, description="Shared warnings for other agents")
    red_flags: Optional[List[str]] = Field(default_factory=list, description="Shared Red Flags for other agents")

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
            "Programme_Eligibility_KB_Smart_Query",
            "Find_Nearest_Medical_Facility"
        ]
    
    @traceable
    async def run(self, state: MedicalAgentState):
        """Main node execution"""
        


        delta = {}

        pool = await MCPClientPool.get_instance()

        
        filtered_tools = await pool.get_tools(self.ALLOWED_TOOLS)


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

        # Extract clean content from LLM response and user message
        last_user_msg = extract_user_message(messages)
        agent_response_text = extract_llm_content(tool_llm_response)
        
        structured_prompt = f"""
        # ROLE: Conversation Parser & Router
        You are the **Backend Logic Processor** for Sehat Link. Your job is to analyze the conversation between a User and "Iris" (the Program Eligibility Agent). 
        You must extract structured data, determine routing triggers, and update the user's profile state.

        # INPUT CONTEXT
        **User's Last Message:** "{last_user_msg}"
        **Iris's (Agent) Response:** "{agent_response_text}"

        # OBJECTIVE
        Generate a valid JSON object matching the `ProgramFeedbackAgent` schema based on the analysis below.

        # EXTRACTION RULES

        ### 1. Response Field
        - Extract the clean, conversational text from Iris's response.
        - **Remove** any XML tags like `<response>`, `<action>`, or JSON blocks. Just the natural language meant for the user.

        ### 2. Eligibility Status (CRITICAL)
        - **baitul_maal_program_eligibility** & **sehat_sahulat_program_eligibility**
        - **"True"**: ONLY if the *User* explicitly confirms they are eligible (e.g., "I checked the link, I am eligible").
        - **"False"**: ONLY if the *User* explicitly says they are not eligible (e.g., "The site says I am not found").
        - **"Not Mentioned"**: Default. Use this if the user is just asking questions, checking links, or if the status is unknown. 
        - **DO NOT** guess based on income or age. Only trust the User's explicit confirmation of the official check.

        ### 3. Shared Facts & Warnings
        - **shared_facts**: Extract permanent user details mentioned (Name, Location, Specific Disease, Age, Phone Number).
            - *Format:* "User has diabetes", "User lives in Karachi", "User CNIC ends in 789".
        - **shared_warnings**: Extract behavioral constraints (e.g., "User is aggressive", "User prefers English only").
        - **red_flags**: Extract **MEDICAL EMERGENCIES** (e.g., "Chest pain", "Suicidal ideation", "Unconscious", "Breathing difficulty").

        ### 4. Routing Triggers (ROUTING LOGIC)

        **A) symptom_trigger (Boolean)**
        *Set to TRUE if:*
        - The user is describing symptoms *and* asking for a diagnosis or medical advice (e.g., "Why does my head hurt?", "Is this dangerous?").
        - The user asks purely medical questions (e.g., "What are the symptoms of Dengue?").
        *Set to FALSE if:*
        - The user mentions symptoms *only* to find a facility (e.g., "I have fever, find a clinic"). Iris handles facility routing.
        - The user is just chatting or asking about program eligibility.

        **B) doctor_trigger (Boolean)**
        *Set to TRUE if:*
        - The user explicitly asks for a human: "I want to talk to a real doctor", "Connect me to a human".
        - The user is unsatisfied with the AI's help and demands escalation.

        ---

        # FEW-SHOT EXAMPLES (Mental Chain of Thought)

        ## Example 1: Facility Lookup (No Trigger)
        **User:** "I have a high fever and need to find the nearest clinic in Lahore."
        **Iris:** "I found 3 clinics near you in Lahore..."
        **Analysis:** User mentioned fever, but the INTENT was finding a facility. Iris handled it. No medical diagnosis needed.
        **Output:**
        {{
            "response": "I found 3 clinics near you in Lahore...",
            "symptom_trigger": false,
            "doctor_trigger": false,
            "shared_facts": ["User is in Lahore", "User has high fever"],
            "sehat_sahulat_program_eligibility": "Not Mentioned",
            "baitul_maal_program_eligibility": "Not Mentioned"
        }}

        ## Example 2: Medical Advice (Symptom Trigger)
        **User:** "I have a high fever and red spots on my body. Is this Dengue? What should I take?"
        **Iris:** "I am not a doctor, but I can help you find a hospital."
        **Analysis:** User is asking "Is this Dengue?" and "What should I take?". This requires the Symptom/Medical Agent.
        **Output:**
        {{
            "response": "I am not a doctor, but I can help you find a hospital.",
            "symptom_trigger": true,
            "doctor_trigger": false,
            "shared_facts": ["Symptoms: Fever, Red spots"],
            "sehat_sahulat_program_eligibility": "Not Mentioned",
            "baitul_maal_program_eligibility": "Not Mentioned"
        }}

        ## Example 3: Eligibility Confirmation
        **User:** "Thanks, I clicked the link you gave. It says I am eligible for Sehat Card!"
        **Iris:** "That is great news! Would you like to find a panel hospital?"
        **Analysis:** User confirmed eligibility explicitly.
        **Output:**
        {{
            "response": "That is great news! Would you like to find a panel hospital?",
            "symptom_trigger": false,
            "doctor_trigger": false,
            "sehat_sahulat_program_eligibility": "True",
            "baitul_maal_program_eligibility": "Not Mentioned"
        }}

        ## Example 4: Emergency (Red Flags)
        **User:** "My father is having severe chest pain and can't breathe!"
        **Iris:** "Please go to the nearest emergency room immediately!"
        **Analysis:** Medical Emergency.
        **Output:**
        {{
            "response": "Please go to the nearest emergency room immediately!",
            "symptom_trigger": false,
            "doctor_trigger": false,
            "red_flags": ["Severe Chest Pain", "Breathing Difficulty"],
            "shared_facts": ["Father is patient"]
        }}

        # NEGATIVE PROMPTING (Guidelines)
        - **DO NOT** set `symptom_trigger` to true just because the word "pain" or "fever" is used. Look for the *intent* of diagnosis.
        - **DO NOT** hallucinate eligibility. If the user says "I might be eligible", the status is "Not Mentioned".
        - **DO NOT** include XML tags in the `response` field. Clean text only.

        Parse the current interaction now.
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

        # Handle None response from structured LLM
        if response is None:
            logger.warning("Structured LLM returned None, using fallback response")
            # Extract response from tool_llm_response if available
            fallback_text = "I'm processing your request. Could you please provide more details?"
            if hasattr(tool_llm_response, 'content'):
                content = tool_llm_response.content
                if isinstance(content, str):
                    fallback_text = content
                elif isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    if isinstance(first_item, dict) and 'text' in first_item:
                        # Extract text and clean XML tags
                        import re
                        raw_text = first_item.get('text', '')
                        # Remove XML tags
                        clean_text = re.sub(r'<[^>]+>', '', raw_text).strip()
                        if clean_text:
                            fallback_text = clean_text
            
            delta["user_messages"] = [AIMessage(content=fallback_text)]
            delta["messages"] = [tool_llm_response] if hasattr(tool_llm_response, 'content') else []
            return delta

        if isinstance(response, BaseModel):
            parsed = response.dict()
        elif isinstance(response, dict):
            parsed = response
        else:
            logger.error(f"Unexpected response type: {type(response)} | {response}")
            delta["user_messages"] = [AIMessage(content="Could you please repeat that again?")]
            delta["messages"] = []
            return delta
        
        response_text = parsed.get("response", "")
        symptom_trigger = parsed.get("symptom_trigger", False)
        doctor_trigger = parsed.get("doctor_trigger", False)
        sehat_sahulat_program_eligibility = parsed.get("sehat_sahulat_program_eligibility", "True")
        baitul_maal_program_eligibility = parsed.get("baitul_maal_program_eligibility", "True")
        shared_facts = parsed.get("shared_facts", "")
        shared_warnings = parsed.get("shared_warnings", "")
        red_flags = parsed.get("red_flags", "")
        
        if symptom_trigger == True: 
            delta["current_agent"] = "symptom_agent"
    
        if doctor_trigger == True: 
            delta["current_agent"] = "doctor_agent"
        
        delta["shared_facts"] = shared_facts
        delta["shared_warnings"] = shared_warnings
        delta["red_flags"] = red_flags
        delta["sehat_sahulat_program_eligibility"] = sehat_sahulat_program_eligibility
        delta["baitul_maal_program_eligibility"] = baitul_maal_program_eligibility
        delta["messages"] = [tool_llm_response]
        delta["user_messages"] = [
            AIMessage(
                content=response_text
            )
        ]

        return delta
