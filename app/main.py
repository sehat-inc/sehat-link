from fastapi import (
    FastAPI,
    Request,
    status
    )
from routers import doctor, patient, hospital

# from api.mcp.server import mcp_app 

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello from Sehat-Link API!"}


app.include_router(doctor.router)
app.include_router(patient.router)
app.include_router(hospital.router)


#NOTE: This is where we merge both MCP and FastAPI app



# combined_app = FastAPI(
#     routes=[
#         *mcp_app.routes,
#         *app.routes,
#     ],
#     lifespan=mcp_app.lifespan
# )






