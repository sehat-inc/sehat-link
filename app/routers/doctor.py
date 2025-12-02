from auth_utils import hash_password, verify_password, create_access_token, get_current_doctor_id
from database import get_supabase 
from models import DoctorSignUpPayload, NewHospitalDetails, DoctorLogin, Token
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, Any, Annotated

router = APIRouter(prefix="/doctor", tags=["Doctor"])

@router.post("/signup")
def signup(doctor_signup_data: DoctorSignUpPayload):
    supabase = get_supabase()
    existing = supabase.table("doctors").select("id").eq("email", doctor_signup_data.email).execute()
    
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    plain_password = doctor_signup_data.password
    doctor_data: Dict[str, Any] = doctor_signup_data.model_dump(
        exclude_none=True,
        exclude={
            "password",
            "existing_hospital_id", 
            "new_hospital_details"
        }
    )
    doctor_data["password_hash"] = hash_password(plain_password)
    
    hospital_id_to_link = None
    
    if doctor_signup_data.existing_hospital_id is not None:
        hospital_id_to_link = doctor_signup_data.existing_hospital_id
    elif doctor_signup_data.new_hospital_details:
        new_hosp: NewHospitalDetails = doctor_signup_data.new_hospital_details
        hospital_data = new_hosp.model_dump(exclude_none=True)
        
        try:
            hospital_response = supabase.table("hospitals").insert(hospital_data).select("id").execute()
            
            if not hospital_response.data:
                raise HTTPException(status_code=500, detail="Failed to retrieve new hospital ID.")
            
            hospital_id_to_link = hospital_response.data[0]["id"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to register new hospital: {str(e)}")
    
    if hospital_id_to_link:
        doctor_data["hospital_id"] = hospital_id_to_link
        doctor_data["clinic_address"] = None 
    elif doctor_signup_data.affiliation_type in ["Private Clinic", "Independent"]:
        doctor_data["hospital_id"] = None
        pass
    
    try:
        response = supabase.table("doctors").insert(doctor_data).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register doctor: {str(e)}")
    
    return {"message": "Doctor registered successfully", "doctor_id": response.data[0]["id"]}


@router.post("/login", response_model=Token)
def login_for_access_token(form_data: DoctorLogin):
    supabase = get_supabase()

    response = supabase.table("doctors").select("id, password_hash").eq("email", form_data.email).execute()
    
    if not response.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    doctor_data = response.data[0]
    stored_hash = doctor_data["password_hash"]
    doctor_id = doctor_data["id"]

    if not verify_password(form_data.password, stored_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Key change: Include user_type in the token payload
    access_token = create_access_token(
        data={"user_id": str(doctor_id), "user_type": "doctor"} 
    )

    return Token(access_token=access_token)


@router.get("/profile")
async def get_doctor_profile(current_user_id: Annotated[str, Depends(get_current_doctor_id)]):
    """A protected route accessible only by authenticated Doctor users."""
    supabase = get_supabase()
    
    # Use the verified doctor ID to fetch specific data
    doctor_record = supabase.table("doctors").select("*").eq("id", current_user_id).execute()
    
    if not doctor_record.data:
        raise HTTPException(status_code=404, detail="Doctor data not found")

    return {
        "message": "Authenticated doctor profile data retrieved.", 
        "doctor_id": current_user_id,
        "profile": doctor_record.data[0]
    }
