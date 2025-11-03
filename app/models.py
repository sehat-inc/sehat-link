from typing import Optional
from datetime import datetime, date
from pydantic import BaseModel, EmailStr, Field


class Hospital(BaseModel):
    id: Optional[int]
    name: str
    address: Optional[str]
    city: Optional[str]
    contact_no: Optional[str]


class Doctor(BaseModel):
    id: Optional[int]
    name: str
    licence_no: str
    email: EmailStr
    password_hash: str
    specialization: Optional[str]
    affiliation_type: Optional[str]
    experience_years: Optional[int]
    city: Optional[str]
    hospital_id: Optional[int]
    clinic_address: Optional[str] 


class Patient(BaseModel):
    id: Optional[int]
    name: str
    phone_no: Optional[str]
    email: EmailStr
    password_hash: str
    dob: Optional[date]
    gender: Optional[str]
    city: Optional[str]


class Appointment(BaseModel):
    id: Optional[int]
    doctor_id: int
    patient_id: int
    appointment_time: datetime
    status: Optional[str]
    notes: Optional[str]


class NewHospitalDetails(BaseModel):
    name: str
    address: str
    city: str
    contact_no: Optional[str]


class DoctorSignUpPayload(BaseModel):
    name: str
    licence_no: str
    email: EmailStr
    password: str 
    specialization: Optional[str]
    affiliation_type: str
    experience_years: Optional[int]
    city: str
    existing_hospital_id: Optional[int] = None
    new_hospital_details: Optional[NewHospitalDetails] = None
    clinic_address: Optional[str] = None