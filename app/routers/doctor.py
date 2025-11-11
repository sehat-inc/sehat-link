from app.auth_utils import hash_password
from app.database import get_supabase 
from app.models import DoctorSignUpPayload, NewHospitalDetails
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

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
            "new_hospital_details", 
            "clinic_address"
        }
    )
    doctor_data["password_hash"] = hash_password(plain_password)
    
    hospital_id_to_link = None
    
    if doctor_signup_data.existing_hospital_id is not None:
        hospital_id_to_link = doctor_signup_data.existing_hospital_id
    elif doctor_signup_data.new_hospital_details:
        new_hosp: NewHospitalDetails = doctor_signup_data.new_hospital_details
        # Fix: Exclude 'id' field to let database auto-generate it
        hospital_data = new_hosp.model_dump(exclude_none=True, exclude={"id"})
        
        try:
            hospital_response = supabase.table("hospitals").insert(hospital_data).execute()
            
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
        doctor_data["clinic_address"] = doctor_signup_data.clinic_address
    
    try:
        response = supabase.table("doctors").insert(doctor_data).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register doctor: {str(e)}")
    
    return response.data
