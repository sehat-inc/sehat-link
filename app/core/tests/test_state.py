from typing import Optional, Any
from datetime import datetime
import uuid

from core.langgraph.utils.state import MedicalAgentState


def create_initial_state(user_id: str, session_id: Optional[str] = None) -> MedicalAgentState:
    """
    Create initial state with dummy/default values.
    In production, populate from database (Supabase).
    """
    return MedicalAgentState(
        # Core Conversation
        messages=[],
        
        # User Context
        user_id=user_id,
        user_name=None,
        user_age=None,
        user_gender=None,
        user_location={},
        user_domicile_location=None,
        user_phone=None,
        preferred_language="Urdu + English",
        
        # Medical History
        chronic_conditions=[],
        allergies=[],
        current_medications=[],
        
        # Session Metadata
        session_id=session_id or str(uuid.uuid4()),
        started_at=datetime.utcnow().isoformat(),
        
        # LLM-Detected Context
        detected_language="Urdu + English",
        detected_urgency="Low",
        detected_problem_type="",
        
        # Symptoms
        symptoms_collected=[],
        symptoms_summary="",
        
        # MCP Tool Results
        symptom_research_result=None,
        similar_cases=[],
        
        # Agent Coordination
        current_agent="frontend",
        previous_agent=None,
        handoff_context=None,
        
        # Agent Flags
        triage_complete=False,
        sufficient_symptom_data=False,
        requires_deep_research=False,
        
        # Shared Knowledge
        shared_facts=[],
        shared_warnings=[],
        red_flags=[]
    )

