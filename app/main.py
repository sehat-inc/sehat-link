from typing import Optional, Any, List, Dict
from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status
    )
import json
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from pinecone.db_control.enums.clouds import CloudProvider

from routers import doctor, patient, hospital

from api.mcp.server import mcp_app
from api.mcp.client import test_agent
from core.langgraph.agent import build_triage_agent, run_graph_with_message
from core.langgraph.utils.state import MedicalAgentSession, MedicalAgentState
from core.tests.test_state import create_initial_state

from core.logging import get_logger


logger = get_logger("MAIN APP LOGIC")

app = FastAPI()

@app.middleware("http")
async def debug_request_middleware(request: Request, call_next):
    print("--- REQUEST RECEIVED ---")
    print(f"PATH: {request.url.path}")
    print("HEADERS:")
    for name, value in request.headers.items():
        print(f"  {name}: {value}")
    print("------------------------")
    
    response = await call_next(request)
    return response

@app.get("/")
def read_root():
    return {"Hello from Sehat-Link API!"}


app.include_router(doctor.router)
app.include_router(patient.router)
app.include_router(hospital.router)


@app.websocket("/ws/chat")
async def chat_socket(socket: WebSocket):
    """
    WebSocket endpoint for real-time chat with triage agent.
    """
    await socket.accept()
    
    graph = build_triage_agent()
    
    try:
        while True:
            # Receive message from client
            data = await socket.receive_text()
            logger.info("Recieve Message from WebSocket")
            
            try:
                # Parse incoming JSON
                payload = json.loads(data)
                user_msg = payload.get("message", "")
                user_id = payload.get("user_id", "")
                chat_start_session = payload.get("session", "")

                if not user_id:
                    await socket.send_text(json.dumps({
                        "response": "Error: Missing user_id",
                        "error": "user_id required"
                    }))
                    continue
               
                if not chat_start_session:
                    await socket.send_text(json.dumps({
                        "response": "Error: Missing initial session state from /chat-start.",
                        "error": "Missing session"
                    }))
                    continue
               
                session = MedicalAgentSession(chat_start_session)
                current_state = session.state

                logger.info(f"-------- SESSION STATE -----\n{current_state}")
                

                # Run graph with message
                result = await run_graph_with_message(graph, current_state, user_msg) 
                logger.info(f"RESULT: {result}")

                updated_state = result["state"]
                
                # Send response back to client
                response_payload = {
                    "response": result["response"],
                    "agent": result["agent"],
                    "urgency": result.get("urgency", "Low"),
                    "language": result.get("language", "english"),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await socket.send_text(json.dumps(response_payload))
                
            except Exception as e:
                logger.info(f"Message processing error: {e}")
                error_response = {
                    "response": "Sorry, I encountered an error processing your message.",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await socket.send_text(json.dumps(error_response))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user") 
    
    except Exception as e:
        logger.info(f"WebSocket error: {e}")
        await socket.close()


#NOTE: This is where we merge both MCP and FastAPI app

combined_app = app
combined_app.mount("/mcp", mcp_app)


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

combined_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
