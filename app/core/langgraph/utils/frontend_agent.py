from typing import Dict, Any, Union, AsyncGenerator
import json
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.helper import safe_str
from core.prompts.mcp_client_prompts import frontend_agent_prompt
from core.logging import get_logger
from core.langgraph.utils.helper import (
    safe_str
)

logger = get_logger("FRONTEND AGENT")

MAX_MEMORY = 6

class FrontendFeedback(BaseModel):
    response: str = Field(description="The calm and empathetic response to the user's query")
    symptom_trigger: bool = Field(description="True if handoff to Symptom agent should occur else false")
    programme_trigger: bool = Field(description="True if handoff to Symptom agent should occur else false")



class FrontendNode(Node):
    """
    This is the first Node that the user will talk to. Does all basic UX
    however, has no specializations and cannot give any medical recommendaitons.
    """
    def __init__(self, name: str = "frontend_agent", temperature: float = 0.7):
        super().__init__(name=name, temperature=temperature)
    

    async def generate_response(self, system_prompt: str, user_prompt: str, streaming: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """
        Unified generator wrapper:
         - if streaming==True: returns an async generator (stream of token strings)
         - if False: returns the final response string (awaitable)
        NOTE: this method returns either a string or async generator. Caller must detect which.
        """
        messages = [
            ("system", system_prompt), 
            ("user", user_prompt)
        ]

        if not streaming:
            structured_llm = self.llm.with_structured_output(
                schema=FrontendFeedback.model_json_schema(), method="json_schema"
            )
            resp = structured_llm.ainvoke(messages)
            #resp = await self.ainvoke(system_prompt, user_prompt)
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


    async def __call__(self, state: MedicalAgentState) -> Dict[str, Any]:
        recent_msgs = state["messages"][-3:]
        msgs_text = "\n".join(safe_str(m) for m in recent_msgs)
        
        # Getting the Last Message for Hand-off
        last_msg_obj = state.get("messages", [])[-1] if state.get("messages") else None
        last_user_txt = ""

        if last_msg_obj is not None:
            last_user_txt = safe_str(last_msg_obj)

        system_prompt = frontend_agent_prompt(state)
       
        user_prompt = f"User's last 3 text\n{msgs_text}\nReturn the output described above for the LAST user message"
        
        logger.info(f"LAST MESSAGES: {msgs_text}")
        
        messages = [
            ("system", system_prompt), 
            ("user", user_prompt)
        ]

        try:
            structured_llm = self.llm.with_structured_output(
                schema=FrontendFeedback.model_json_schema(), method="json_schema"
            )
            response = await structured_llm.ainvoke(messages)

        except Exception as e:
            print(f"Frontend Error: {e}")
            response = "I'm sorry, I encountered a technical issue. Could you please repeat that?"

        parsed = {
            "symptom_trigger": False,
            "programme_trigger": False
        }
        
        result = {
            "response_text": response,
            "router": {
                "symptom_trigger": False,
                "programme_trigger": False
            }
        }
        

        symptom_trigger = response["symptom_trigger"] 
        programme_trigger = response["programme_trigger"] 
        logger.info(f"TRIGGERS-----------\nSYMPTOM: {symptom_trigger}\nPROGRAM: {programme_trigger}")

        delta: Dict[str, Any] = {}
        

        if symptom_trigger == True: 
            bridging_message = (
                "I hear you — thank you for telling me that. "
                "I'm going to pass this to our clinical intake specialist who will ask a few focused questions to understand your symptoms better. "
                "They will take it from here."
                "If this is an emergency, please say so or call your local emergency number immediately."
            )
            delta["bridge_messages"] = [
                AIMessage(
                    content=bridging_message
                )
            ]
            delta["handoff_context"] = last_user_txt
            delta["current_agent"] = "symptom_agent"
            delta["symptom_init"] = True
            delta["symptom_trigger"] = True

            return delta

        if programme_trigger == True:
            bridging_message = (
                "I hear you — thank you for telling me that. "
                "I'm going to pass this to our programme specialist who will ask a few focused questions to understand your situation better. "
                "They will take it from here."
            )
            
            delta["bridge_messages"] = [
                AIMessage(
                    content=bridging_message
                )]
            delta["handoff_context"] = last_user_txt
            delta["current_agent"] = "programme_eligibility_agent"

            return delta

        delta["messages"] = [
            AIMessage(
                content=response["response"]
            )]
        delta["current_agent"] = "frontend_agent"

        return delta


