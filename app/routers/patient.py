from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status, Depends
import json

from auth_utils import hash_password, verify_password, create_access_token, get_current_user_id
from database import get_supabase, fetch_longterm_by_user_id
from models import PatientSignUp, PatientLogin, Token

from core.langgraph.utils.state import MedicalAgentState, MedicalAgentSession

router = APIRouter(prefix="/patient", tags=["Patient"])

@router.post("/signup")
def signup(patient: PatientSignUp):
    supabase = get_supabase()
    
    existing = supabase.table("patients").select("email").eq("email", patient.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    password_hash = hash_password(patient.password)
    
    data = patient.model_dump(exclude={"password"})
    data["password_hash"] = password_hash

    if data.get("dob") and hasattr(data["dob"], "isoformat"):
        data["dob"] = data["dob"].isoformat()

    try:
        # Standard approach that often works for returning data
        response = supabase.table("patients").insert(data).select("id").execute() 
        
        if not response.data:
             # Fallback check if the insert was successful but didn't return data
             raise HTTPException(status_code=500, detail="Insertion successful but failed to retrieve patient ID.")

    except AttributeError:
        # If the chaining .select("id") failed, try the simpler execute()
        response = supabase.table("patients").insert(data).execute()
        if not response.data:
             raise HTTPException(status_code=500, detail="Failed to register patient and retrieve ID.")

    return {"message": "Patient registered successfully", "patient_id": response.data[0]["id"]}


@router.post("/login", response_model=Token)
def login_for_access_token(form_data: PatientLogin):
    supabase = get_supabase()

    response = supabase.table("patients").select("id, password_hash").eq("email", form_data.email).execute()
    
    if not response.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    patient_data = response.data[0]
    stored_hash = patient_data["password_hash"]
    patient_id = patient_data["id"]

    if not verify_password(form_data.password, stored_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"user_id": str(patient_id), "user_type": "patient"} 
    )

    return Token(access_token=access_token)


@router.get("/chat-start")
async def start_ai_chat_session(current_user_id: Annotated[str, Depends(get_current_user_id)]):
    """
    This is the key protected endpoint for your AI workflow. 
    The AI developer gets the verified user_id and patient context here.
    """
    supabase = get_supabase()
    
    patient_record = supabase.table("patients").select("*").eq("id", current_user_id).execute()
    
    if not patient_record.data:
        raise HTTPException(status_code=404, detail="Patient data not found")

    patient_data = patient_record.data[0]
    print(patient_data)
    
    try:
        user_idd = int(current_user_id)
    except ValueError:
        id_t = type(user_idd)
        raise HTTPException(status_code=400, detail=f"Incorrect format of UserID: {id_t} | should be int8")
    
    longterm_resp = supabase.table("longterm_session").select("*").eq("user_id", user_idd).execute()
    longterm_data = getattr(longterm_resp, "data", None)
    
    print(longterm_data)

    # If no data exists for this user, create a new row
    if not longterm_data:
        new_row = {
            "user_id": user_idd,
            "user_name": patient_data.get("name"),
            "user_age": patient_data.get("age"),
            "user_gender": patient_data.get("gender"),
            "user_location": patient_data.get("location"),
            "user_domicile_location": patient_data.get("domicile_location"),
            "user_phone": patient_data.get("phone"),
            "preferred_language": patient_data.get("preferred_language", "en"),
        }

        insert_resp = supabase.table("longterm_session").insert(new_row).execute()
        data = getattr(insert_resp, "data", [new_row])  # fallback if Supabase doesn’t return row
    else:
        data = longterm_data

    print(data)

    # Take the first row (Supabase always returns a list)
    row: Dict[str, Any] = data[0]

    # Build the state
    state: MedicalAgentState = {
        # Core Conversation
        "messages": [],

        # User Context
        "user_id": row.get("user_id", 0),
        "user_name": row.get("user_name"),
        "user_age": row.get("user_age"),
        "user_gender": row.get("user_gender"),
        "user_location": row.get("user_location"), 
        "user_domicile_location": row.get("user_domicile_location"), 
        "user_phone": row.get("user_phone"),
        "preferred_language": row.get("preferred_language") or "en",

        # Medical History
        "chronic_conditions": (
            json.loads(row["chronic_conditions"]) if isinstance(row.get("chronic_conditions"), str) else (row.get("chronic_conditions") or [])
        ),
        "allergies": (
            json.loads(row["allergies"]) if isinstance(row.get("allergies"), str) else (row.get("allergies") or [])
        ),
        "current_medications": (
            json.loads(row["current_medications"]) if isinstance(row.get("current_medications"), str) else (row.get("current_medications") or [])
        ),

        # LLM-Detected Context
        "detected_language": row.get("detected_language") or "en",
        "detected_urgency": row.get("detected_urgency", "Medium"),
        "detected_problem_type": row.get("detected_problem_type") or "",

        # Symptoms
        "symptoms_collected": row.get("symptoms_collected") or [],
        "symptoms_summary": row.get("symptoms_summary") or "",

        # MCP Tool Results
        "symptom_research_result": row.get("symptom_research_result"),
        "similar_cases": row.get("similar_cases") or [],

        # Agent Coordination
        "current_agent": row.get("current_agent") or "triage_agent",
        "previous_agent": row.get("previous_agent"),
        "handoff_context": row.get("handoff_context"),

        # Agent Flags
        "triage_complete": row.get("triage_complete", False),
        "sufficient_symptom_data": row.get("sufficient_symptom_data", False),
        "requires_deep_research": row.get("requires_deep_research", False),

        # Shared Knowledge
        "shared_facts": row.get("shared_facts") or [],
        "shared_warnings": row.get("shared_warnings") or [],
        "red_flags": row.get("red_flags") or [],
    }    

    print(f"State: \n{state}")

    session = MedicalAgentSession(state=state)
    
    print(f"Session: \n{session}")

    return {
        "message": "Authenticated. AI chat session initiated.", 
        "user_id": current_user_id,
        "session": session
    }
