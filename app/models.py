from datetime import datetime, date
from typing import Optional, List

from pydantic import BaseModel, EmailStr, Field

# --- Core Database Models (Used for DB retrieval and storage) ---

class Hospital(BaseModel):
    id: Optional[int] = None
    name: str
    address: Optional[str] = None
    city: Optional[str] = None
    contact_no: Optional[str] = None

class DoctorDB(BaseModel):
    id: Optional[int] = None
    name: str
    licence_no: str
    email: EmailStr
    password_hash: str
    specialization: Optional[str] = None
    affiliation_type: Optional[str] = None
    experience_years: Optional[int] = None
    city: Optional[str] = None
    hospital_id: Optional[int] = None
    clinic_address: Optional[str] = None 

class PatientDB(BaseModel):
    id: Optional[int] = None
    name: str
    phone_no: Optional[str] = None
    email: EmailStr
    password_hash: str
    dob: Optional[date] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    domicile_location: Optional[str] = None
    last_hospital_visit: Optional[str] = None
    chronic_conditions: Optional[List[str]] = Field(default_factory=list)
    allergies: Optional[List[str]] = Field(default_factory=list)
    current_medications: Optional[List[str]] = Field(default_factory=list)
    past_prescriptions: Optional[List[str]] = Field(default_factory=list)
    language_preferred: Optional[List[str]] = Field(default_factory=list)
    communication_style: Optional[str] = None

class Appointment(BaseModel):
    id: Optional[int] = None
    doctor_id: int
    patient_id: int
    appointment_time: datetime
    status: Optional[str] = None
    notes: Optional[str] = None

# --- Auth & Input Models ---

class PatientLogin(BaseModel):
    email: EmailStr
    password: str

class DoctorLogin(BaseModel):
    email: EmailStr
    password: str

class TokenData(BaseModel):
    user_id: Optional[str] = None
    user_type: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    
# --- Sign Up Payloads ---

class PatientSignUp(BaseModel):
    name: str
    phone_no: Optional[str] = None
    email: EmailStr
    password: str
    dob: Optional[date] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    domicile_location: Optional[str] = None  # NEW FIELD
    last_hospital_visit: Optional[str] = None
    chronic_conditions: Optional[List[str]] = Field(default_factory=list)
    allergies: Optional[List[str]] = Field(default_factory=list)
    current_medications: Optional[List[str]] = Field(default_factory=list)
    past_prescriptions: Optional[List[str]] = Field(default_factory=list)
    language_preferred: Optional[List[str]] = Field(default_factory=list)
    communication_style: Optional[str] = None

class NewHospitalDetails(BaseModel):
    name: str
    address: str
    city: str
    contact_no: Optional[str] = None

class DoctorSignUpPayload(BaseModel):
    name: str
    licence_no: str
    email: EmailStr
    password: str 
    specialization: Optional[str] = None
    affiliation_type: str
    experience_years: Optional[int] = None
    city: str
    existing_hospital_id: Optional[int] = None
    new_hospital_details: Optional[NewHospitalDetails] = None
    clinic_address: Optional[str] = None
