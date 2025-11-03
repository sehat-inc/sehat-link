from sqlmodel import Session, select
from database import engine
from models import Patient
from auth import hash_password
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/patient", tags=["Patient"])

def get_session():
    with Session(engine) as session:
        yield session

@router.post("/signup")
def signup(patient: Patient, session: Session = Depends(get_session)):
    existing = session.exec(select(Patient).filter(Patient.email == patient.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    patient.password_hash = hash_password(patient.password_hash)
    session.add(patient)
    session.commit()
    session.refresh(patient)
    return {"id": patient.id, "email": patient.email}
