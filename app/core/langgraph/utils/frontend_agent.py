from typing import Dict, Any, Union, AsyncGenerator
import json
from langchain_core.messages import AIMessage

from core.langgraph.utils.base_node import Node
from core.langgraph.utils.state import MedicalAgentState
from core.langgraph.utils.helper import safe_str
from core.prompts.mcp_client_prompts import frontend_agent_prompt
from core.langgraph.utils.helper import (
    safe_str,
    infer_province_from_city
)



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

        system_prompt = frontend_agent_prompt()
       
        user_prompt = f"User's last 3 text\n{msgs_text}\nReturn the JSON described above for the LAST user message"
        
        try:
            response = await self.ainvoke(system_prompt, user_prompt) 
        except Exception as e:
            print(f"Frontend Error: {e}")
            response = ""


        parsed = {"is_symptom": False, "extracted": {"age": None, "gender": None, "city": None, "domicile_city": None}}
        if response:
            try:
                parsed_candidate = json.loads(response)
                parsed = parsed_candidate

            except Exception as e:
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed_candidate = json.loads(response[start:end+1])
                        parsed = parsed_candidate
                    except Exception:
                        # fall back: leave parsed as default
                        parsed = parsed
       
        is_symptom = bool(parsed.get("is_symptom"))
        extracted = parsed.get("extracted") or {}
        age = extracted.get("age")
        gender = extracted.get("gender")
        city = extracted.get("city")
        domicile_city = extracted.get("domicile_city")

        delta: Dict[str, Any] = {}
        # update profile fields only if not already present (prefer existing)
        if age is not None:
            try:
                age_int = int(age)
                if not state.get("user_age"):
                    delta["user_age"] = age_int
            except Exception:
                pass

        if gender:
            gender_norm = str(gender).strip().lower()
            if gender_norm in ("male", "female", "other") and not state.get("user_gender"):
                delta["user_gender"] = gender_norm

        # current location (city + province inference)
        if city:
            city_str = str(city).strip()
            existing_location = state.get("user_location") or {}
            # prefer existing explicit fields
            if not existing_location.get("city"):
                inferred_prov = infer_province_from_city(city_str)
                delta["user_location"] = {"city": city_str, "province": inferred_prov or existing_location.get("province")}

        # domicile location (separate field)
        if domicile_city:
            domicile_city_str = str(domicile_city).strip()
            # only write if not present
            if not state.get("domicile_location"):
                delta["domicile_location"] = {"city": domicile_city_str, "province": infer_province_from_city(domicile_city_str)}


        if is_symptom:
            # bridging reply from Ms Sehat (calm nurse persona)
            bridging_message = (
                "I hear you — thank you for telling me that. "
                "I'm going to pass this to our clinical intake specialist who will ask a few focused questions to understand your symptoms better. "
                "They will take it from here."
                "If this is an emergency, please say so or call your local emergency number immediately."
            )

            # append the bridging AI message and set handoff context so symptom agent can start immediately
            delta["messages"] = [AIMessage(content=bridging_message)]
            # handoff_context holds raw last user message so symptom agent can consume it immediately
            delta["handoff_context"] = last_user_txt
            # set the next agent explicitly
            delta["current_agent"] = "symptom_agent"

        else:
            # not symptom — generate normal friendly reply (Ms Sehat style) and continue conversation
            # compose a short system prompt for generation
            reply_system = """
                You are Ms Sehat, a calm and empathetic hospital triage nurse.
                Keep reply brief, kind, and helpful. Do NOT provide medical advice or diagnosis.
                You may ask clarifying non-clinical questions (e.g. 'Are you asking for yourself?')
                and offer assistance such as how to connect to clinical intake if user wants.\n
            """
            reply_user = f"User recent text:\n{last_user_txt}\n\nRespond as Ms Sehat in 1-2 short sentences."


            try:
                resp = await self.ainvoke(reply_system, reply_user)
            except Exception as e:
                print(f"Error from Frontend Node: {e}")
                resp = "Thanks for telling me - can you tell me a bit more or say 'start symptoms' if you want to describe your symptoms"

            delta["messages"] = [AIMessage(content=resp)]
            delta["current_agent"] = "frontend_agent"

        delta.setdefault("last_frontend_parse", {})  # non-user-facing, for logs
        delta["last_frontend_parse"].update({
            "is_symptom": is_symptom,
            "extracted": {
                "age": age,
                "gender": gender,
                "city": city,
                "domicile_city": domicile_city
            },
        })

        return delta


