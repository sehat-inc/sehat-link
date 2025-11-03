from fastapi import APIRouter, HTTPException
from database import supabase
from auth import hash_password
from models import Doctor

router = APIRouter(prefix="/doctor", tags=["Doctor"])

@router.post("/signup")
def signup(doctor: Doctor):
    existing = supabase.table("doctors").select("*").eq("email", doctor.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    data = doctor.model_dump()
    data["password_hash"] = hash_password(data["password_hash"])
    response = supabase.table("doctors").insert(data).execute()
    return response
