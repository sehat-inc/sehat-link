from typing import Dict, Any, Optional, Union, AsyncGenerator, List
from langchain_core import messages
from langchain_core.messages import AIMessage, HumanMessage
import json

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
import json

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


class SymptomAgentNode(Node):
    """
    Main Symptom Agent that will record and summaarize symptoms as well as
    choose whether to call MCP Deep Research tool

    """
    def __init__(self, 
                 name: str = "symptom_agent", 
                 temperature: float = 0.7, 
                 mcp_manager: Optional[MCPToolManager] = None, 
                 allowed_tools: List[str] = None):
        super().__init__(name=name, temperature=temperature, mcp_manager=mcp_manager, allowed_tools=allowed_tools)
    
    async def _call_mcp_research(self, symptoms: List[Dict[str, Any]]) -> str:
        """Call MCP tool for deep symptom research"""
        if not self.has_tool_access("Symptom Knowledge Base Smart Query"):
            logger.warning("No access to Symptom KB Smart Query tool")
            return ""
        
        query = f"Patient presenting with: {symptoms}. What are potential diagnoses and recommendations?"
        
        try:
            logger.info(f"Calling MCP tool with query: {query}")
            result = await self.use_mcp_tool(
                "Symptom Knowledge Base Smart Query",
                {
                    "question": query,
                    "namespace": "__default__",
                    "decompose": True, 
                    "top_k_per_query": 5,
                }
            )
            logger.info(f"MCP tool returned: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calling MCP tool: {e}")
            return ""

    async def _generate_symptoms_summary(self, db_search_results, state: MedicalAgentState) -> str:
        """Generate natural language summary of collected symptoms"""
        
        recent_msgs = state["messages"][-10:]
        msgs_text = "\n".join(safe_str(m) for m in recent_msgs)

        if not db_search_results:
            return "No symptoms collected yet."
        
        system_prompt =f""" 
                You are an expert medical summarizer. Your task is to summarize a patient's symptoms using two sources of information:
                1. Recent conversation messages with the patient.
                2. Additional relevant information retrieved from a Vector Database (medical data, symptom patterns, etc.).

                Speak directly to the patient in a polite, professional, and encouraging tone. Suggest the most relevant type of doctor or specialist if appropriate. Keep the summary concise (max 3-4 lines).  

                Never:
                - Give alarming predictions (e.g., "you might die" or "this is fatal")
                - Repeat greetings unnecessarily
                - Provide treatment instructions beyond suggesting a type of doctor
                - Invent symptoms not mentioned by the patient
                - Use casual, joking, or unprofessional language

                Few-shot examples:

                Example 1:
                Patient: "I have a headache and my body feels hot for two days."
                Vector DB info: "Symptoms consistent with mild viral infection, fever common."
                Summary: "You have been experiencing a headache and feeling warm for the past two days. These symptoms may suggest a mild infection. I recommend consulting a general physician for proper evaluation."

                Example 2:
                Patient: "My throat is sore, and I have a slight fever."
                Vector DB info: "Symptoms match common throat infections; mild fever is typical."
                Summary: "You have a sore throat and mild fever. These symptoms could indicate a throat infection. Consulting a general physician or ENT specialist would be advisable."

                Example 3:
                Patient: "I've been shivering and have a severe headache."
                Vector DB info: "Persistent headache with shivering could indicate systemic infection."
                Summary: "You have been experiencing shivering and a severe headache. This could indicate an infection that needs attention. Please consult a general physician soon."
                """

        user_prompt = f"""
            Recent conversation messages (last messages):\n{msgs_text}
            Vector DB search results:\n{db_search_results}

            Based on both the patient's messages and the Vector DB information, summarize the patient's symptoms clearly, speaking directly to them. 
            Briefly indicate the likely issue and recommend the type of doctor to consult. Keep it concise, professional, and encouraging.
            """

        response = await self.ainvoke(system_prompt, user_prompt)

        return response 

    async def __call__(self, state: MedicalAgentState) -> Dict[str, Any]:
        """Main node execution"""
        delta: Dict[str, Any] = {}
        symptom_threshold = 3

        recent_msgs = state["messages"][-10:]
        msgs_text = "\n".join(safe_str(m) for m in recent_msgs)

        
        # Get handoff context for initial greeting
        handoff_context = state.get("handoff_context", "")
        
        # Get existing symptoms
        existing_symptoms = state.get("symptoms_collected", [])
        symptoms_count = len([item for item in existing_symptoms if isinstance(item, dict)])
        
        logger.info(f"EXISTING SYMPTOM COUNT: {symptoms_count}")
        logger.info(f"SymptomAgent called. Symptoms collected: {symptoms_count}")
        
        # Build system prompt
        system_prompt = symptom_agent_prompt(state)
        

        user_prompt = f"""
        You are now responsible for attenting the patient. This is the chat history: {msgs_text}.
        This is the main reason why you were put responsible of the patient: "{handoff_context}". 

        Start by acknowledging what they shared, then ask ONE focused follow-up question to understand their symptoms better.
        Extract any symptoms mentioned in their original message.
        """
        
        try:
            # Get LLM response
            #response = await self.ainvoke(system_prompt, user_prompt)
            
            messages = [
                ("system", system_prompt),
                ("user", user_prompt)
            ]

            structured_llm = self.llm.with_structured_output(
                schema=SymptomAgentFeedback
            )

            response = await structured_llm.ainvoke(messages)
            
            if isinstance(response, BaseModel):
                parsed = response.dict()
            elif isinstance(response, dict):
                parsed = response
            else:
                raise ValueError(f"Unexpected response type: {type(response)} | {response}")

            
            response_text = parsed.get("response", "")
            new_symptoms = parsed.get("symptoms") or []

            
            try:
                updated_symptoms = existing_symptoms + new_symptoms
                logger.info(f"UPDATED SYMPTOMS: {updated_symptoms}")
            except Exception as e:
                logger.error("Error in Appending New Symptoms")
           
            try:
                # Prepare delta
                delta["symptoms_collected"] = new_symptoms
                delta["current_agent"] = "symptom_agent"
            except Exception as e:
                logger.error(f"Failed to Append Delta: {e}")
            
            new_symptoms_count = len([item for item in new_symptoms if isinstance(item, dict)])
            logger.info(f"NEW SYPTOM COUNT: {new_symptoms_count}")

            should_research = (int(new_symptoms_count) + int(symptoms_count)) > int(symptom_threshold)
            
            if should_research:
                delta["requires_deep_research"] = True

            if should_research and (state['requires_deep_research'] == True or state['requires_deep_research'] == None):
                logger.info(f"Threshold reached symptoms. Calling MCP research...")
                
                # Call MCP tool
                research_results = await self._call_mcp_research(updated_symptoms)
                
                # Generate summary
                symptoms_summary = await self._generate_symptoms_summary(research_results, state)
                
                delta["symptoms_summary"] = symptoms_summary
                delta["mcp_research_results"] = research_results
                delta["requires_deep_research"] = False
                
                
                delta["messages"] = [
                    AIMessage(
                        content=symptoms_summary
                    )
                ]
            else:
                # Continue collecting symptoms
                delta["messages"] = [
                    AIMessage(
                        content=response_text
                    )
                ]
            
            return delta
            
        except Exception as e:
            logger.error(f"SymptomAgent error: {e}", exc_info=True)
            return {
                "messages": [
                    AIMessage(
                        content="I'm sorry, I encountered an issue. Could you please repeat that?"
                    )
                ],
                "current_agent": "symptom_agent"
            } 

        
        
