from typing import Optional, Dict, List, Annotated, Literal, Any, Sequence, Union, Iterable
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AnyMessage
import operator



def merge_symptoms(
    existing: List[Dict[str, Any]], 
    incoming: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Reducer for LangGraph state.
    - Input: Lists of plain dictionaries (because response.dict() was called).
    - Logic: Deduplicates by symptom name. Updates specific fields. Appends details.
    """
    merged_map = {}
    order = []

    def upsert(source):
        if not source: 
            return
        
        # 1. Safe access (Handles if source is dict)
        sym_raw = source.get("symptom")
        if not sym_raw:
            return
            
        key = str(sym_raw).strip().lower()
        if not key or key == "none":
            return

        # 2. Normalize values
        sev = _normalize_val(source.get("severity"))
        dur = _normalize_val(source.get("duration"))
        loc = _normalize_val(source.get("location"))
        add = _normalize_val(source.get("additional_details"))

        if key not in merged_map:
            # New Entry
            merged_map[key] = {
                "symptom": source.get("symptom"), # Keep original casing
                "severity": sev,
                "duration": dur,
                "location": loc,
                "additional_details": add,
            }
            order.append(key)
        else:
            # Existing Entry - Smart Merge
            cur = merged_map[key]
            
            # For Scalar values (Severity, Duration, Location):
            # Overwrite if the new value is present (Patient update)
            if sev: cur["severity"] = sev
            if dur: cur["duration"] = dur
            if loc: cur["location"] = loc
            
            # For Details: Concatenate to preserve history
            if add:
                if cur["additional_details"] and add.lower() not in cur["additional_details"].lower():
                    cur["additional_details"] = f"{cur['additional_details']}; {add}"
                elif not cur["additional_details"]:
                    cur["additional_details"] = add

    # Process existing state first
    for e in existing or []:
        upsert(e)

    # Process new incoming data
    for inc in incoming or []:
        upsert(inc)

    return [merged_map[k] for k in order]

def _normalize_val(v):
    """Helper to clean empty strings."""
    if v is None: return None
    if isinstance(v, str):
        s = v.strip()
        return None if s == "" or s.lower() == "none" else s
    return v

def deduplicate_merge(current: Optional[List[str]], new: Optional[List[str]]) -> List[str]:
    """
    Merges new strings into the existing list, removing duplicates 
    while preserving the original insertion order.
    """
    # Handle None types safely
    current = current or []
    new = new or []
    
    # Combine lists
    combined = current + new
    
    # Deduplicate while preserving order
    seen = set()
    result = []
    for item in combined:
        # Clean whitespace just in case
        clean_item = item.strip()
        if clean_item and clean_item not in seen:
            seen.add(clean_item)
            result.append(clean_item)
            
    return result

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
    urgency_checked: bool  # Flag to ensure urgency is only checked once
    
    # Symptoms (LLM-extracted from conversation)
    symptom_trigger: bool
    symptom_init: bool
    symptoms_collected: Annotated[list, merge_symptoms]  # [{symptom, severity, duration, location}]
    symptom_route: str
    symptom_research_result: Optional[str]  # From MCP deep research tool
    
    # Program
    program_trigger: bool
    sehat_sahulat_program_eligibility: Optional[str]
    baitul_maal_program_eligibility: Optional[str]
    
    # Doctor
    required_specialty: Optional[str]
    doctor_collected: Annotated[list, operator.add]
    call_trigger: bool

    # Agent Coordination
    current_agent: str
    previous_agent: Optional[str]
    handoff_context: Optional[str]
    
    # Agent Flags
    triage_complete: bool
    sufficient_symptom_data: bool  # Triggers MCP research
    requires_deep_research: bool
    
    # Shared Knowledge
    shared_facts: Annotated[list, deduplicate_merge]
    shared_warnings: Annotated[list, deduplicate_merge]
    red_flags: Annotated[list, deduplicate_merge]

    # prescription
    prescription_data: Optional[Dict[str, Any]]
    prescription_processed: Optional[bool]

    # disease
    disease_name: Optional[str]
