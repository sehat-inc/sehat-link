from fastapi import (
    FastAPI,
    Request,
    status
    )
from database import init_db
from routers import doctor, patient

app = FastAPI()

init_db()

@app.get("/")
def read_root():
    return {"Hello from Sehat-Link API!"}


app.include_router(doctor.router)
app.include_router(patient.router)




