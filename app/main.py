from fastapi import (
    FastAPI,
    Request,
    status
    )
from database import init_db
from routers import doctor, patient

from api.mcp.server import mcp_app 

app = FastAPI()

init_db()

@app.get("/")
def read_root():
    return {"Hello from Sehat-Link API!"}


app.include_router(doctor.router)
app.include_router(patient.router)


#NOTE: This is where we merge both MCP and FastAPI app



combined_app = FastAPI(
    routes=[
        *mcp_app.routes,
        *app.routes,
    ],
    lifespan=mcp_app.lifespan
)






