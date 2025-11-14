from typing import AsyncIterator, Dict, Any, Optional, Union, AsyncGenerator, Tuple, List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI 
from api.mcp.client import Client
from langchain_core.messages import SystemMessage, AIMessage
from dotenv import load_dotenv
import json
import os

from core.langgraph.utils.state import MedicalAgentState
from core.prompts.mcp_client_prompts import (
    language_detector_prompt,
    triage_agent_prompt,
    urgency_detector_prompt,
    frontend_agent_prompt
)
from core.logging import get_logger


logger = get_logger("NODE LOGIC")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

NODE_STREAMING_MODE = {
    "frontend_agent": True, 
    # All agents that talk with user added here
}

def safe_str(x):
    if isinstance(x, str): 
        return x
    else: 
        return str(x)

    if hasattr(x, "content"):
        return str(x.content)
    if hasattr(x, "text"):
        return str(x.text)
    return str(x)

def safe_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_list(value, default=None):
    if value is None:
        return default or []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return default or []
    elif isinstance(value, list):
        return value
    else:
        return default or []

def safe_bool(value, default=False):
    if value is None:
        return default
    return bool(value)

# TODO: Add proper geolocation - i remember doing this before but now forgot
_CITY_TO_PROVINCE = {
    "karachi": "Sindh",
    "lahore": "Punjab",
    "faisalabad": "Punjab",
    "rawalpindi": "Punjab",
    "multan": "Punjab",
    "peshawar": "Khyber Pakhtunkhwa",
    "quetta": "Balochistan",
    "islamabad": "Islamabad Capital Territory",
}

def infer_province_from_city(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    key = city.strip().lower()
    return _CITY_TO_PROVINCE.get(key)



class Node:
    """
    Base Node Interface :) 
    
    Remarks:
        I did this solely because after re-writing this code 10 times did I realise that all nodes must be 
        uniform and I cant just create functions like in docs otherwise it will get extremely
        hard to scale when there are more than 5 agents
    """
    name: str

    def __init__(self, name: str, api_key: str = str(GEMINI_API_KEY), model: str = "gemini-2.5-flash", temperature: float = 0.7):
        self.name = name
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature,
            convert_system_message_to_human=True,
        )

    async def ainvoke(self, prompt: str, user_prompt: str = "") -> str:
        if user_prompt != "":
            message = [
                ("system", prompt),
                ("human", user_prompt)
            ]
            resp = await self.llm.ainvoke(message)

            return safe_str(resp.content)

        resp = await self.llm.ainvoke(prompt)
        return safe_str(resp.content)
    
    async def astream(self, messages: List[Tuple[str, str]]) -> AsyncIterator[str]:
       
        async for chunk in self.llm.astream(messages):
            yield safe_str(chunk.content)


    async def __call__(self, state: MedicalAgentState) -> Dict[str, Any]:
        raise NotImplementedError("Node Children must implement __call__")


class LanguageDetectorNode(Node):
    """
    Language Detector Node 
    Checks for languages and type of converstation style    
    """
    
    def __init__(self, name: str = "language_detector", temperature: float = 0.3):
       super().__init__(name=name, temperature=temperature) 

    async def __call__(self, state: MedicalAgentState) -> Dict[str, Any]:
        

        recent_messages = state["messages"][-5:]
        text = "\n".join(safe_str(m.content) for m in recent_messages)

        query_prompt = language_detector_prompt(text)

        response = await self.ainvoke(query_prompt)
        
        return {
            "detected_language": response,
            "current_agent": "language_detector"
        }


class UrgencyDetectorNode(Node):
    """
    Checks Urgency 
    
    State:
        detected_urgency: Literal["Emergency", "High", "Medium", "Low"]
    """
    def __init__(self, name: str = "urgency_detector", temperature: float = 0.6):
        super().__init__(name=name, temperature=temperature)

    async def __call__(self, state: MedicalAgentState) -> Dict[str, Any]:
        
        symptoms = state.get("symptoms_collected", [])

        if not symptoms:
            logger.info("UrgencyDetectorNode skipped - no symptoms collected yet")
            # Return state unchanged
            return {"current_agent": "urgency_detector"}        
        
        symptom_lines = []
        for s in state["symptoms_collected"]:
            line = f"""- Symptom: {s.get('symptom', 'unknown')},
                   Severity: {s.get('severity', 'unknown')},
                   Duration: {s.get('duration', 'unknown')},
                   Location: {s.get('location', 'unknown')}"""
            symptom_lines.append(line)

        symptom_text = "\n".join(symptom_lines)
        
        prompt = urgency_detector_prompt()
        
        message = f"{prompt} \nPatient symptoms: \n{symptom_text}"

        response = await self.ainvoke(message)

        return {
            "detected_urgency": response,
            "current_agent": "urgency_detector"
        }


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


class SymptomAgent(Node):
    """
    Main Symptom Agent that will record and summaarize symptoms as well as
    choose whether to call MCP Deep Research tool

    State:
        
    """
        

        
        

# class TriageAgentNode:
#     """
#     LangGraph based Triage Agent Node
#     """
#
#     def __init__(
#         self, mcp_client: Client, 
#         api_key: str, state: MedicalAgentState,): self.mcp_client = mcp_client self.llm = ChatGoogleGenerativeAI( api_key=api_key, model.5-flash",
#                 filter_dict={"type": "conversation"}
#             )
#             self.state["similar_cases"] = similar_cases
#
#             similar_context = "\n".join([
#                 f"- {case['date']}: {case['summary']} (Outcome: {case['outcome']})"
#                 for case in similar_cases
#             ])
#         else:
#             similar_context = "No similar cases found"
#
#         system_prompt = triage_agent_prompt(
#             state=self.state,
#             similar_context=similar_context
#         )
#
#         messages = [SystemMessage(content=system_prompt)] + self.state["messages"]
#         response = await self.llm.ainvoke(messages)
#
#         self.state["messages"].append(AIMessage(content=response.text))
#
#         # Trigger MCP deep research if sufficient data
#         if self.state["sufficient_symptom_data"] and not self.state["symptom_research_result"]:
#             self.state["requires_deep_research"] = True
#
#         return self.state
