from typing import Dict, Any, Optional, Union, AsyncGenerator, List
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
    
    async def generate_response(self, system_prompt: str, user_prompt: str, streaming: bool = False
                               ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Unified generator wrapper:
         - if streaming==True: returns an async generator (stream of token strings)
         - if False: returns the final response string (awaitable)
        NOTE: this method returns either a string or async generator. Caller must detect which.
        """
        # normalized messages structure expected by ChatGoogleGenerativeAI
        messages = [
            ("system", system_prompt), 
            ("user", user_prompt)
        ]

        if not streaming:
            # single-shot completion using your existing ainvoke wrapper
            # Use the base Node.ainvoke which wraps self.llm.ainvoke
            # (your ainvoke takes system prompt and user prompt)
            resp = await self.ainvoke(system_prompt, user_prompt)
            return safe_str(resp)

        # streaming path: return an async generator that yields token strings
        async def _stream_gen():
            # docs show: async for chunk in (await llm.astream(messages))
            stream_iter = self.astream(messages)
            async for chunk in stream_iter:
                # defensive canonicalization of token text from chunk
                token = getattr(chunk, "delta", None) or getattr(chunk, "content", None) or getattr(chunk, "text", None)
                if token is None:
                    token = str(chunk)
                yield str(token)
        return _stream_gen()

    
    def _extract_structured_data(self, response: str) -> Dict[str, Any]:
        """Extract response text and symptoms from LLM output"""
        result = {
            "response_text": response,
            "symptoms": []
        }
        
        # Extract response text
        if "<response>" in response and "</response>" in response:
            start = response.find("<response>") + len("<response>")
            end = response.find("</response>")
            result["response_text"] = response[start:end].strip()
        
        # Extract symptoms JSON
        if "<symptoms>" in response and "</symptoms>" in response:
            start = response.find("<symptoms>") + len("<symptoms>")
            end = response.find("</symptoms>")
            symptoms_json = response[start:end].strip()
            try:
                result["symptoms"] = json.loads(symptoms_json)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse symptoms JSON: {e}")
                result["symptoms"] = []
        
        return result

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

    async def _generate_symptoms_summary(self, db_search_results) -> str:
        """Generate natural language summary of collected symptoms"""
        if not db_search_results:
            return "No symptoms collected yet."
        
        system_prompt =f"""You are an expert summarizer of data. You escpecially know the needs of doctors and 
        healthcare professionals and how they like to review data of patients."""

        user_prompt = f"""Summarize the results of the following Vector DB Query. Organize the RESULTS ONLY.
        CONTEXT: \n{db_search_results}
        """

        response = await self.ainvoke(system_prompt, user_prompt)

        return response 

    async def __call__(self, state: MedicalAgentState) -> Dict[str, Any]:
        """Main node execution"""
        delta: Dict[str, Any] = {}
        symptom_threshold = 3
        
        # Get handoff context for initial greeting
        handoff_context = state.get("handoff_context", "")
        
        # Get existing symptoms
        existing_symptoms = state.get("symptoms_collected", [])
        symptoms_count = len([item for item in existing_symptoms if isinstance(item, dict)])
        
        logger.info(f"SymptomAgent called. Symptoms collected: {symptoms_count}")
        
        # Build system prompt
        system_prompt = symptom_agent_prompt(state)
        
        # Build user prompt
        if handoff_context:
            last_msg = state.get("messages", [])[-1] if state.get("messages") else None
            last_user_txt = safe_str(last_msg) if last_msg else ""

            user_prompt = f"""
            The patient was just handed off to you with this Handoff Context:
            "{handoff_context}" and this last message of the user \n{last_user_txt}

            Start by acknowledging what they shared, then ask ONE focused follow-up question to understand their symptoms better.
            Extract any symptoms mentioned in their original message.
            """
        else:
            logger.info("Using Last Message instead of Handoff Context")
            # Get last user message
            last_msg = state.get("messages", [])[-1] if state.get("messages") else None
            last_user_txt = safe_str(last_msg) if last_msg else ""
            
            user_prompt = f"""
            The patient was just handed off to you with this Handoff Context:
            "{last_user_txt}"

            Start by acknowledging what they shared, then ask ONE focused follow-up question to understand their symptoms better.
            Extract any symptoms mentioned in their original message.
            """
        
        try:
            # Get LLM response
            response = await self.ainvoke(system_prompt, user_prompt)
            
            # Extract structured data
            structured = self._extract_structured_data(response)
            response_text = structured["response_text"]
            new_symptoms = structured["symptoms"]
            
            logger.info(f"Collected {len(new_symptoms)} new symptoms.") 
            
            try:
                logger.info("DEBUG 1")
                # Prepare delta
                delta["symptoms_collected"] = new_symptoms
                logger.info("DEBUG 2")
                delta["current_agent"] = ["symptom_agent"]
                logger.info("DEBUG 3")
            except Exception as e:
                logger.error(f"Failed to Append Delta: {e}")
            
            logger.info("DEBUG 4")
            updated_symptoms = existing_symptoms + new_symptoms
            # Check if we should call MCP tool
            should_research = int(len(updated_symptoms)) > int(symptom_threshold)
            
            if should_research:
                logger.info(f"Threshold reached symptoms. Calling MCP research...")
                
                # Call MCP tool
                research_results = await self._call_mcp_research(updated_symptoms)
                
                # Generate summary
                symptoms_summary = await self._generate_symptoms_summary(updated_symptoms)
                
                delta["symptoms_summary"] = symptoms_summary
                delta["mcp_research_results"] = research_results
                delta["mcp_research_completed"] = True
                
                # Add response with research context
                summary_message = f"\n\n---\n**Symptoms Summary:**\n{symptoms_summary}\n\nI've gathered enough information. Let me analyze this..."
                
                delta["messages"] = [
                    AIMessage(
                        content=response_text + summary_message,
                        additional_kwargs={
                            "new_symptoms": new_symptoms,
                            "research_triggered": True
                        }
                    )
                ]
            else:
                # Continue collecting symptoms
                delta["messages"] = [
                    AIMessage(
                        content=response_text,
                        additional_kwargs={
                            "new_symptoms": new_symptoms,
                            "research_triggered": False
                        }
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

        
        
