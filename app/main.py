from fastapi import (
    FastAPI,
    Request,
    status
    )


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello from Sehat-Link API!"}







