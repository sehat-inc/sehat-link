from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    status
    )
from routers import doctor, patient, hospital

from api.mcp.server import mcp_app 

app = FastAPI()

from api.mcp.server import mcp_app
from api.mcp.client import test_agent

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello from Sehat-Link API!"}


app.include_router(doctor.router)
app.include_router(patient.router)
app.include_router(hospital.router)


@app.websocket("/ws/chat")
async def chat_socket(socket: WebSocket):
    await socket.accept()
    while True:
        user_msg = await socket.receive_text()
        result = await test_agent(user_msg)
        await socket.send_text(result)


#NOTE: This is where we merge both MCP and FastAPI app

combined_app = FastAPI(
    routes=[
        *mcp_app.routes,
        *app.routes,
    ],
    lifespan=mcp_app.lifespan
)

