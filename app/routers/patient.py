from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status, Depends
import json
import uuid

from auth_utils import hash_password, verify_password, create_access_token, get_current_user_id
from database import get_supabase, fetch_longterm_by_user_id
from models import PatientSignUp, PatientLogin, Token

from core.langgraph.utils.state import MedicalAgentState
from core.logging import get_logger

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

logger = get_logger("PATIENT ROUTER")

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


async def load_initial_state_from_db(user_id: int) -> MedicalAgentState:
    """Load initial state from database"""
    
    try:
        supabase = get_supabase()
        resp = supabase.table("longterm_session").select("*").eq("user_id", int(user_id)).execute()
    except Exception as e:
        logger.error(f"Error in getting supabase longterm session: {e}")
        

    if not resp.data:
        raise ValueError(f"No session data found for user {user_id}")
   
        
    row: Dict[str, Any] = resp.data[0]
    state: MedicalAgentState = {
        "messages": [],
        "user_id": safe_int(row.get("user_id"), 0),
        "user_name": safe_str(row.get("user_name")),
        "user_age": safe_int(row.get("user_age")),
        "user_gender": safe_str(row.get("user_gender")),
        "user_location": safe_str(row.get("user_location", "")), 
        "user_domicile_location": safe_str(row.get("user_domicile_location")),
        "user_phone": safe_str(row.get("user_phone")),
        "preferred_language": safe_str(row.get("preferred_language", "en")),

        # Medical History
        "chronic_conditions": safe_list(row.get("chronic_conditions"), []),
        "allergies": safe_list(row.get("allergies"), []),
        "current_medications": safe_list(row.get("current_medications"), []),
        
        # LLM-Detected Context
        "detected_language": safe_str(row.get("detected_language", "en")),
        "detected_urgency": row.get("detected_urgency", "Medium"),

        # Symptoms
        "symptoms_collected": safe_list(row.get("symptoms_collected"), []),

        # MCP Tool Results
        "symptom_research_result": safe_str(row.get("symptom_research_result")),

        
        # Shared Knowledge
        "shared_facts": safe_list(row.get("shared_facts"), []),
        "shared_warnings": safe_list(row.get("shared_warnings"), []),
        "red_flags": safe_list(row.get("red_flags"), []),

        # Prescription
        "prescription_data": safe_list(row.get("prescription_data"), []),
        
        # Programs
        "sehat_sahulat_program_eligibility": safe_str(row.get("sehat_sahulat_program_eligibility", "")),
        "baitul_maal_program_eligibility": safe_str(row.get("baitul_maal_program_eligibility", "")),

        # Disease
        "disease_name" : safe_str(row.get("disease_name", "")),
    }
    
    return state

async def ensure_user_session_exists(user_id: int, supabase):
    """
    Create and Initial session if it doesnt exist in db
    """
    
    try:
        longterm_resp = supabase.table("longterm_session").select("*").eq("user_id", user_id).execute()
        longterm_data = getattr(longterm_resp, "data", None)

        logger.info(longterm_data)
    except Exception as e:
        logger.error(f"Error getting longterm session: {e}")
        return

    if not longterm_data:
        try:
            patient_record = supabase.table("patients").select("*").eq("id", user_id).execute()
            patient_data = patient_record.data[0]
            new_row = {
                "user_id": user_id,
                "user_name": patient_data.get("name"),
                "user_age": patient_data.get("age"),
                "user_gender": patient_data.get("gender"),
                "user_location": patient_data.get("location"),
                "user_domicile_location": patient_data.get("domicile_location"),
                "user_phone": patient_data.get("phone"),
                "preferred_language": patient_data.get("preferred_language", "en"),
            }

            insert_resp = supabase.table("longterm_session").insert(new_row).execute()
            data = getattr(insert_resp, "data", [new_row])  
        except Exception as e:
            logger.error(f"Error in Setting new Patient Record in Fallback: {e}")
            return
    else:
        data = longterm_data

    logger.info(f"DATA AFTER ENSURANCE: {data}")


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
    
    
    await ensure_user_session_exists(int(current_user_id), supabase)
    
    return {
        "message": "Authenticated. AI chat session initiated.", 
        "user_id": int(current_user_id)
    }
