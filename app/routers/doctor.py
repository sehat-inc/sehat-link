from sqlmodel import Session, select
from database import engine
from models import Doctor
from auth import hash_password
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/doctor", tags=["Doctor"])

def get_session():
    with Session(engine) as session:
        yield session

@router.post("/signup")
def signup(doctor: Doctor, session: Session = Depends(get_session)):
    existing = session.exec(select(Doctor).filter(Doctor.email == doctor.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    doctor.password_hash = hash_password(doctor.password_hash)
    session.add(doctor)
    session.commit()
    session.refresh(doctor)
    return {"id": doctor.id, "email": doctor.email}
