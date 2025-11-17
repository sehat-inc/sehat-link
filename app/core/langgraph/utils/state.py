from typing import Optional, Dict, List, Annotated, Literal, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


def reducer(a: list, b: str | None) -> list:
    if b is not None:
        return a + [b]
    return a


class MedicalAgentState(TypedDict):
    """
    Complete state for medical consultation system.
    Redis checkpointer auto-saves this state.
    """
    # Core Conversation
    messages: Annotated[List[BaseMessage], operator.add]
    
    # User Context (loaded from FastAPI/Supabase)
    user_id: int
    user_name: Optional[str]
    user_age: Optional[int]
    user_gender: Optional[str]
    user_location: str
    user_domicile_location: Optional[str]
    user_phone: Optional[str]
    # This is different from detected_language as this is
    # populated at the session end where LLM infers communication preference
    preferred_language: str

    # Medical History (from FastAPI/Supabase)
    chronic_conditions: List[str]
    allergies: List[str]
    current_medications: List[str]
    
    # LLM-Detected Context (updated by detector nodes)
    detected_language: str  # LLM detection result
    detected_urgency: Literal["Emergency", "High", "Medium", "Low"]  # LLM detection
    detected_problem_type: str
    
    # Symptoms (LLM-extracted from conversation)
    program_trigger: bool
    symptom_trigger: bool
    symptoms_collected: Annotated[list, operator.add]  # [{symptom, severity, duration, location}]
    symptoms_summary: str  # Natural language summary
    
    # MCP Tool Results
    symptom_research_result: Optional[Dict]  # From MCP deep research tool
    similar_cases: List[Dict]  # From Pinecone vector search
    
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


class MedicalAgentSession:
    def __init__(self, state: Optional[MedicalAgentState] = None):
        self.state: Optional[MedicalAgentState] = state

    def update_state(self, updates: dict):
        if self.state is None:
            raise ValueError("State not initialized")
        for k, v in updates.items():
            if k in self.state:
                self.state[k] = v
