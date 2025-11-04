from app.auth import hash_password
from app.database import supabase
from app.models import Patient
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/patient", tags=["Patient"])

@router.post("/signup")
def signup(patient: Patient):
    existing = supabase.table("patients").select("*").eq("email", patient.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    data = patient.model_dump()
    data["password_hash"] = hash_password(data["password_hash"])

    # Convert date fields to ISO string
    if data.get("dob"):
        data["dob"] = data["dob"].isoformat()

    response = supabase.table("patients").insert(data).execute()
    return response
