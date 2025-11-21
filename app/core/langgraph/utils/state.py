from typing import Optional, Dict, List, Annotated, Literal, Any, Sequence, Union, Iterable
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AnyMessage
import operator

from core.langgraph.utils.tool_manager import MCPToolManager

def _normalize_val(v):
    """Treat empty strings and 'none' strings as missing."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() == "none":
            return None
        return s
    return v


def merge_symptoms(
    existing: Iterable[Dict[str, Union[str, None]]],
    incoming: Iterable[Dict[str, Union[str, None]]],
    *,
    prefer_new: bool = False
) -> List[Dict[str, Union[str, None]]]:
    """
    Merge two lists of symptom dicts (plain dicts), dedupe by 'symptom' (case-insensitive).
    - existing, incoming: iterables of dicts like {"symptom": "acne", "duration": ..., "location": ..., "additional_details": ...}
    - prefer_new: if True, when both existing and incoming have a non-empty value for the same field,
                 prefer the incoming value. Default False: keep existing value.
    Returns a list of merged dicts preserving the first-seen order of symptoms (existing first, then new ones).
    """
    merged_map = {}
    order = []

    def upsert(source, is_incoming=False):
        sym_raw = source.get("symptom")
        if not sym_raw:
            return
        key = str(sym_raw).strip().lower()
        if key == "":
            return

        # normalize fields
        dur = _normalize_val(source.get("duration"))
        loc = _normalize_val(source.get("location"))
        add = _normalize_val(source.get("additional_details"))

        if key not in merged_map:
            # Keep symptom as originally cased from source (prefer existing if present)
            merged_map[key] = {
                "symptom": source.get("symptom"),
                "duration": dur,
                "location": loc,
                "additional_details": add,
            }
            order.append(key)
            return

        cur = merged_map[key]

        # helper to decide whether to write incoming value
        def choose(field_cur, field_new):
            if field_new is None:
                return field_cur
            if field_cur is None:
                return field_new
            return field_new if prefer_new and is_incoming else field_cur

        # apply merges
        cur["duration"] = choose(cur.get("duration"), dur)
        cur["location"] = choose(cur.get("location"), loc)
        cur["additional_details"] = choose(cur.get("additional_details"), add)

    # insert existing first (preserve their casing for 'symptom')
    for e in existing or []:
        if isinstance(e, dict):
            upsert(e, is_incoming=False)

    # then merge incoming
    for inc in incoming or []:
        if isinstance(inc, dict):
            upsert(inc, is_incoming=True)

    # produce ordered list
    return [merged_map[k] for k in order]


class MedicalAgentState(TypedDict):
    """
    Complete state for medical consultation system.
    Redis checkpointer auto-saves this state.
    """
    # Core Conversation
    user_messages: Annotated[Sequence[BaseMessage] ,operator.add] 
    messages: Annotated[Sequence[BaseMessage] ,operator.add]
    bridge_messages: Annotated[List[AnyMessage], operator.add]
    
    tool_call_count: int
    error_count: int

    # User Context (loaded from FastAPI/Supabase)
    user_id: int
    user_name: str
    user_age: Optional[int]
    user_gender: Optional[str]
    user_location: str
    user_domicile_location: Optional[str]
    user_phone: Optional[str]
    # This is different from detected_language as this is
    # populated at the session end where LLM infers communication preference
    preferred_language: Optional[str]

    # Medical History (from FastAPI/Supabase)
    chronic_conditions: List[str]
    allergies: List[str]
    current_medications: List[str]
    
    # LLM-Detected Context (updated by detector nodes)
    detected_language: str  # LLM detection result
    detected_urgency: Literal["Emergency", "High", "Medium", "Low"]  # LLM detection
    detected_problem_type: str
    
    # Symptoms (LLM-extracted from conversation)
    symptom_trigger: bool
    symptom_init: bool
    symptoms_collected: Annotated[list, merge_symptoms]  # [{symptom, severity, duration, location}]
    symptoms_summary: str  # Natural language summary
    symptom_route: str

    # MCP Tool Results
    symptom_research_result: Optional[Dict]  # From MCP deep research tool
    
    # Program
    program_trigger: bool
    sehat_sahulat_program_eligibility: Optional[str]
    baitul_maal_program_eligibility: Optional[str]
    
    # Doctor
    required_specialty: Optional[str]
    doctor_collected: Annotated[list, operator.add]

    # Agent Coordination
    current_agent: str
    previous_agent: Optional[str]
    handoff_context: Optional[str]
    
    # Agent Flags
    triage_complete: bool
    sufficient_symptom_data: bool  # Triggers MCP research
    requires_deep_research: bool
    
    # Shared Knowledge
    shared_facts: Annotated[list, operator.add]
    shared_warnings: Annotated[list, operator.add]
    red_flags: Annotated[list, operator.add]  # Medical red flags detected

    # prescription
    prescription_data: Optional[Dict[str, Any]]

    # disease
    disease_name: Optional[str]
