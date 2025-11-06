from typing import Annotated

from fastapi import APIRouter, HTTPException, status, Depends

from app.auth_utils import hash_password, verify_password, create_access_token, get_current_user_id
from app.database import get_supabase
from app.models import PatientSignUp, PatientLogin, Token

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
    
    # This is the context the AI agent needs to maintain state and provide accurate recommendations.
    ai_context = {
        "user_id": current_user_id,
        "name": patient_data.get("name"),
        "chronic_conditions": patient_data.get("chronic_conditions"),
        "allergies": patient_data.get("allergies"),
        "language_preferred": patient_data.get("language_preferred")
    }

    return {
        "message": "Authenticated. AI chat session initiated.", 
        "user_id": current_user_id,
        "ai_data_context": ai_context
    }