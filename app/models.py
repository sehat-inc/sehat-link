from typing import Optional, List
from datetime import datetime, date
from sqlmodel import SQLModel, Field, Relationship

class Hospital(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    address: Optional[str] = None
    city: Optional[str] = None
    contact_no: Optional[str] = None
    doctors: List["Doctor"] = Relationship(back_populates="hospital")

class Doctor(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    licence_no: str
    email: str
    password_hash: str
    specialization: Optional[str] = None
    affiliation_type: Optional[str] = None
    experience_years: Optional[int] = None
    city: Optional[str] = None
    hospital_id: Optional[int] = Field(default=None, foreign_key="hospital.id")
    hospital: Optional[Hospital] = Relationship(back_populates="doctors")

class Patient(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    phone_no: Optional[str] = None
    email: str
    password_hash: str
    dob: Optional[date] = None
    gender: Optional[str] = None
    city: Optional[str] = None

class Appointment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    doctor_id: int = Field(foreign_key="doctor.id")
    patient_id: int = Field(foreign_key="patient.id")
    appointment_time: datetime
    status: Optional[str] = None
    notes: Optional[str] = None
