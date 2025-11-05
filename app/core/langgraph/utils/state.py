from typing import Optional, Dict, List, Annotated, Literal, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


class MedicalAgentState(TypedDict):
    """
    Complete state for medical consultation system.
    Redis checkpointer auto-saves this state.
    """
    # Core Conversation
    messages: Annotated[List[BaseMessage], operator.add]
    
    # User Context (loaded from FastAPI/Supabase)
    user_id: str
    user_name: Optional[str]
    user_age: Optional[int]
    user_gender: Optional[str]
    user_location: Dict[str, str]
    user_phone: Optional[str]
    preferred_language: str
    
    # Medical History (from FastAPI/Supabase)
    chronic_conditions: List[str]
    allergies: List[str]
    current_medications: List[str]
    
    # Session Metadata
    session_id: str
    started_at: str
    
    # LLM-Detected Context (updated by detector nodes)
    detected_language: str  # LLM detection result
    detected_urgency: Literal["Emergency", "High", "Medium", "Low"]  # LLM detection
    detected_problem_type: str
    
    # Symptoms (LLM-extracted from conversation)
    symptoms_collected: List[Dict[str, Any]]  # [{symptom, severity, duration, location}]
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
    shared_facts: List[str]
    shared_warnings: List[str]
    red_flags: List[str]  # Medical red flags detected
