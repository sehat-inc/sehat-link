from typing import Dict, Any, Union, AsyncGenerator
import json
from langchain_core.messages import AIMessage, HumanMessage

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

class FrontendNode(Node):
    """
    This is the first Node that the user will talk to. Does all basic UX
    however, has no specializations and cannot give any medical recommendaitons.
    """
    def __init__(self, name: str = "frontend_agent", temperature: float = 0.7):
        super().__init__(name=name, temperature=temperature)
    

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
        
        try:
            response = await self.ainvoke(system_prompt, user_prompt) 
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
        
        logger.info("Starting FRONTEND Parsing")
        if "<response>" in response and "</response" in response:
            start = response.find("<response") + len("<response>")
            end = response.find("</response>")
            result["response_text"] = response[start:end].strip()
            logger.info(f"RESULT: {result['response_text']}")

        # Extract symptoms JSON
        logger.info("Starting Routing FRONTEND Parsing")
        if "<router>" in response and "</router>" in response:
            start = response.find("<router>") + len("<router>")
            end = response.find("</router>")
            symptoms_json = response[start:end].strip()
            try:
                result["router"] = json.loads(symptoms_json)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse symptoms JSON: {e}")
                result["router"] = {
                    "symptom_trigger": False,
                    "programme_trigger": False
                }

        # start = response.rfind("{")
        # if start != -1:
        #     try:
        #         json_str = response[start:]
        #         parsed = json.loads(json_str)
        #         # This is the key: we use the text BEFORE the JSON as the reply
        #         clean_message = response[:start].strip()
        #         logger.info("Successfully parsed JSON and separated message.")
        #     except json.JSONDecodeError as e:
        #         logger.warning(f"Failed to parse JSON, using full response as message: {e}")
        
        
        clean_message = result["response_text"]
        routers = result["router"]
        symptom_trigger = bool(routers.get("symptom_trigger"))
        programme_trigger = bool(routers.get("programme_trigger"))

        delta: Dict[str, Any] = {}
        

        if symptom_trigger:
            bridging_message = (
                "I hear you — thank you for telling me that. "
                "I'm going to pass this to our clinical intake specialist who will ask a few focused questions to understand your symptoms better. "
                "They will take it from here."
                "If this is an emergency, please say so or call your local emergency number immediately."
            )

            delta["messages"] = [
                AIMessage(
                    content=bridging_message,
                    additional_kwargs={"structured":parsed}
                )]
            delta["handoff_context"] = last_user_txt
            delta["current_agent"] = "symptom_agent"

            return delta

        if programme_trigger:
            bridging_message = (
                "I hear you — thank you for telling me that. "
                "I'm going to pass this to our programme specialist who will ask a few focused questions to understand your situation better. "
                "They will take it from here."
            )
            
            delta["messages"] = [
                AIMessage(
                    content=bridging_message,
                    additional_kwargs={"structured":parsed}
                )]
            delta["handoff_context"] = last_user_txt
            delta["current_agent"] = "programme_eligibility_agent"

            return delta

        delta["messages"] = [
            AIMessage(
                content=clean_message,
                additional_kwargs={"structured":parsed}
            )]
        delta["current_agent"] = "frontend_agent"

        return delta


