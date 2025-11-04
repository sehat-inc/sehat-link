from typing import Optional, Dict, List, Literal
from typing_extensions import TypedDict

class UserProfile(TypedDict):
    """
    Persistant User Info synced with PostgreSQL and Redis
    """
    user_id: str # Primary Key from PostgreSQL/Redis
    name: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    location: Optional[Dict[str, str]] # {"city": "Karachi"}
    phone: Optional[str]
    email: Optional[str]

    # Medical History 
    last_hospital_visit: Optional[str] # Datetime
    chronic_conditions: List[str]
    allergies: List[str]
    current_medications: List[str]
    past_prescriptions: List[str]

    # Preferences
    preffered_language: str
    communication_style: str

    # System Metadata
    created_at: str
    last_active: str
    total_sessions: int


class SessionState(TypedDict):
    """
    Current Session Info - maintained across all agents in the current session
    """
    session_id: str
    started_at: str

    # Detected Context (Updated during the session)
    detected_language: str
    detected_urgency: Literal["Emergency", "High", "Medium", "Low"]

