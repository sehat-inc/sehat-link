from typing import Dict, Any, Optional, Union, AsyncGenerator, List
import re
import json
from langsmith import traceable
from langchain.agents import create_agent
from langchain_core import messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from pydantic import BaseModel, Field

from core.langgraph.utils.tool_manager import MCPClientPool
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


def extract_llm_content(llm_response) -> str:
    """
    Extract only the text content from an LLM response, removing metadata and signatures.
    """
    if not hasattr(llm_response, 'content'):
        return str(llm_response)
    
    content = llm_response.content
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Extract text, ignore 'extras' with signatures
                if 'text' in item:
                    text_parts.append(item['text'])
            elif isinstance(item, str):
                text_parts.append(item)
        return ''.join(text_parts)
    
    return str(content)


def extract_user_message(messages: list) -> str:
    """
    Extract the actual user message from message history.
    If the last message is a ToolMessage (tool output), return "[Tool was called]" 
    and find the actual last human message.
    """
    if not messages:
        return ""
    
    last_msg = messages[-1]
    
    # If it's a ToolMessage, the "user message" for context is the tool result
    # But we want to show the actual user's question, not the tool output
    if isinstance(last_msg, ToolMessage):
        # Find the last actual HumanMessage
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Extract text from list content
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'])
                        elif isinstance(item, str):
                            texts.append(item)
                    return ''.join(texts)
        return "[Tool was called]"
    
    # Regular message extraction
    content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
    
    if isinstance(content, str):
        # Check if this looks like a tool output JSON (starts with { and has strategy/results)
        if content.strip().startswith('{') and ('"strategy"' in content or '"results"' in content):
            return "[Tool was called]"
        return content
    
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
            elif isinstance(item, str):
                texts.append(item)
        result = ''.join(texts)
        # Check if concatenated result looks like tool output
        if result.strip().startswith('{') and ('"strategy"' in result or '"results"' in result):
            return "[Tool was called]"
        return result
    
    return str(content)


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
    disease_name: Optional[str] = Field(description="One word summary of the disease the user has or most critical symptom")

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
        
        delta = {}
        
        pool = await MCPClientPool.get_instance()

        
        filtered_tools = await pool.get_tools(self.ALLOWED_TOOLS)
        

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
        
        # Extract actual user message (handles tool output case)
        last_user_msg = extract_user_message(messages)
        
        # Extract only the text content from the LLM response (removes metadata, signatures, etc.)
        agent_response_text = extract_llm_content(tool_llm_response)
        
        structured_prompt = f"""# ROLE: Medical Conversation Router & Data Parser

        You are the system logic engine for Sehat Link. Your job is to parse the output from "Nora" (the Symptom Agent) and the User's latest message to determine the next system state.

        You must map the unstructured XML/Text output into a strict JSON structure matching the \`SymptomAgentFeedback\` schema.

        # INPUTS
        1. **User's Last Message:** "{last_user_msg}"
        2. **Nora's (Agent) Response:** 
        {agent_response_text}

        # INSTRUCTIONS

        ## 1. Data Parsing (XML to JSON)
        You must extract data from Nora's XML tags and map them to the output schema:

        - **response**: Extract text from \`<response>...</response>\`.
        - **symptoms**: Parse the JSON inside \`<data_extraction>\` -> \`symptoms_collected\`. Map fields:
            - \`name\` -> \`symptom\`
            - \`severity\` -> \`severity\` (if missing, put "unknown")
            - \`duration\` -> \`duration\`
            - \`location\` -> \`location\`
            - \`details\`/\`type\` -> \`additional_details\`
        - **shared_facts**: Extract from \`<data_extraction>\` -> \`shared_facts\`.
        - **shared_warnings**: Extract from \`<data_extraction>\` -> \`shared_warnings\`.
        - **red_flags**: Extract from \`<data_extraction>\` -> \`red_flags\`.
        - **symptom_research_result**: Extract text from \`<symptom_research_result>\`. 
          - Format it as: \`{{ "summary": "extracted text..." }}\`. If empty, use \`{{ "summary": null }}\`.
        
        ## 2. Disease Name Extraction (CRITICAL)
        Extract \`disease_name\` based on the \`symptom_research_result\` or the conversation context.
        
        **Rules:**
        1. **Identified Disease:** If the agent explicitly mentions a likely condition (e.g., "Symptoms align with Migraine", "Possible Dengue"), use that name (1-2 words).
        2. **Critical Symptom:** If no specific disease is named but the user has a MAJOR/CRITICAL symptom (e.g., "Chest Pain", "Breathing Difficulty"), use that symptom as the name.
        3. **Fallback (Default):** If symptoms are minor, unclear, or the agent is still gathering information without a hypothesis, **YOU MUST USE "No Disease"**.
        
        *Examples:*
        - "Possible Malaria" -> "Malaria"
        - "Severe crushing chest pain" -> "Chest Pain"
        - "I have a headache" (Initial gathering) -> "No Disease"

        ## 3. Routing Logic (Triggers)
        Determine \`doctor_trigger\` and \`programme_trigger\`. 
        **DEFAULT TO FALSE** unless specific criteria are met.

        ### A. Doctor Trigger (\`doctor_trigger\`)
        **Set to TRUE only if:**
        1. The User **EXPLICITLY** asks for a doctor/specialist in \`last_user_msg\` (e.g., "find me a doctor", "I need to see someone", "book appointment").
        2. The User replies "Yes" to a previous offer to find a doctor.
        
        **Set to FALSE if:**
        - Nora's \`<action>\` is \`offer_doctor_search\` BUT the user has NOT said "yes" yet. (Nora is *offering*, not confirming).
        - Nora's \`<action>\` is \`continue_gathering\`, \`call_smart_query\`, or \`call_direct_query\`.
        - User is still describing symptoms.

        ### B. Programme Trigger (\`programme_trigger\`)
        **Set to TRUE only if:**
        1. User mentions financial difficulty (e.g., "cannot afford", "too expensive", "no money").
        2. User asks about government schemes, insurance, Sehat Card, or free clinics.
        
        **Set to FALSE otherwise.**

        # OUTPUT SCHEMA (JSON)
        Target class: \`SymptomAgentFeedback\`

        {{
            "response": "String",
            "symptoms": [List of Symptom objects],
            "programme_trigger": Boolean,
            "doctor_trigger": Boolean,
            "shared_facts": [List of strings],
            "shared_warnings": [List of strings],
            "red_flags": [List of strings],
            "symptom_research_result": {{ "summary": "String or Null" }},
            "disease_name": "String" 
        }}

        # EXAMPLES

        ## Example 1: Gathering Info (No Disease Yet)
        **User:** "I have a throbbing headache on the left side."
        **Nora Action:** \`<action>continue_gathering</action>\`
        **Nora Data:** \`<data_extraction> {{ "symptoms_collected": [...] }} ...\`
        
        **Output:**
        \`\`\`json
        {{
            "response": "I understand. How long have you had this headache?",
            "symptoms": [
                {{ "symptom": "headache", "duration": "unknown", "location": "left side", "additional_details": "severity: severe" }}
            ],
            "programme_trigger": false,
            "doctor_trigger": false,
            "shared_facts": [],
            "shared_warnings": [],
            "red_flags": [],
            "symptom_research_result": {{ "summary": null }},
            "disease_name": "No Disease"
        }}
        \`\`\`

        ## Example 2: Identified Disease (Dengue)
        **User:** "I have high fever and spots on my body."
        **Nora Response:** \`<symptom_research_result>Symptoms strongly suggest Dengue Fever.</symptom_research_result>\`

        **Output:**
        \`\`\`json
        {{
            "response": "These signs are concerning for Dengue...",
            "symptoms": [{{ "symptom": "fever", "additional_details": "high" }}, {{ "symptom": "rash", "additional_details": "spots" }}],
            "programme_trigger": false,
            "doctor_trigger": false,
            "shared_facts": [],
            "shared_warnings": [],
            "red_flags": [],
            "symptom_research_result": {{ "summary": "Symptoms strongly suggest Dengue Fever." }},
            "disease_name": "Dengue Fever"
        }}
        \`\`\`

        ## Example 3: Explicit Doctor Handoff + Critical Symptom
        **User:** "Yes, please find a doctor. My chest pain is unbearable."
        **Nora Action:** \`<action>offer_doctor_search</action>\`

        **Output:**
        \`\`\`json
        {{
            "response": "I am finding a cardiologist immediately.",
            "symptoms": [],
            "programme_trigger": false,
            "doctor_trigger": true, 
            "shared_facts": [],
            "shared_warnings": [],
            "red_flags": ["Chest Pain"],
            "symptom_research_result": {{ "summary": "Potential Cardiac Event" }},
            "disease_name": "Chest Pain"
        }}
        \`\`\`

        ## Example 4: Programme/Financial Handoff
        **User:** "I really need help but I don't have any money."
        **Nora Action:** \`<action>offer_doctor_search</action>\`

        **Output:**
        \`\`\`json
        {{
            "response": "I understand your financial concern...",
            "symptoms": [],
            "programme_trigger": true,
            "doctor_trigger": false,
            "shared_facts": ["Financial constraint"],
            "shared_warnings": [],
            "red_flags": [],
            "symptom_research_result": {{ "summary": null }},
            "disease_name": "No Disease"
        }}
        \`\`\`
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

        # Handle None response from structured LLM
        if response is None:
            logger.warning("Structured LLM returned None, using fallback response")
            fallback_text = "I understand you're not feeling well. Could you tell me more about your symptoms?"
            if hasattr(tool_llm_response, 'content'):
                content = tool_llm_response.content
                if isinstance(content, str):
                    fallback_text = content
                elif isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    if isinstance(first_item, dict) and 'text' in first_item:
                        raw_text = first_item.get('text', '')
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
        new_symptoms = parsed.get("symptoms") or []
        programme_trigger = parsed.get("programme_trigger", False)
        doctor_trigger = parsed.get("doctor_trigger", False)
        shared_facts = parsed.get("shared_facts", [])
        shared_warnings = parsed.get("shared_warnings", [])
        red_flags = parsed.get("red_flags", [])
        symptom_research_result = parsed.get("symptom_research_result", "")
        disease_name = parsed.get("disease_name", "No Disease")

        logger.info(f"SYMPTOM TRIGGERS-----------: \nPROGRAM: {programme_trigger}\nDOCTOR: {doctor_trigger}")

        if programme_trigger == True: 
            delta["current_agent"] = "programme_eligibility_agent"
                  
        
        if doctor_trigger == True: 
            delta["current_agent"] = "doctor_agent"
        
        delta["disease_name"] = disease_name
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
