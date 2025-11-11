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

from routers import doctor, patient, hospital

from api.mcp.server import mcp_app
from api.mcp.client import test_agent
from core.langgraph.agent import build_triage_agent, run_graph_with_message
from core.langgraph.utils.state import MedicalAgentState
from core.tests.test_state import create_initial_state


app = FastAPI()

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
    
    Protocol:
        Client sends: {"message": "user text", "user_id": "123", "session_id": "optional"}
        Server sends: {"response": "ai text", "agent": "frontend", "urgency": "Low"}
    """
    await socket.accept()
    
    graph = build_triage_agent()
    
    # Session state (persists across messages in this connection)
    session_state: Optional[MedicalAgentState] = None 
    user_id: Optional[str] = "user_1"
    session_id: Optional[str] = "1"
    
    try:
        while True:
            # Receive message from client
            data = await socket.receive_text()
            
            try:
                # Parse incoming JSON
                payload = json.loads(data)
                user_msg = payload.get("message", "")
                
                # Initialize state on first message
                session_state = create_initial_state(user_id, session_id)
                session_id = session_state["session_id"]
                
                # Run graph with message
                result = await run_graph_with_message(graph, session_state, user_msg)
                
                # Update session state
                session_state = result["state"]
                
                # Send response back to client
                response_payload = {
                    "response": result["response"],
                    "agent": result["agent"],
                    "urgency": result.get("urgency", "Low"),
                    "language": result.get("language", "english"),
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await socket.send_text(json.dumps(response_payload))
                
            except json.JSONDecodeError:
                # Handle plain text messages (backward compatibility)
                if session_state is None:
                    session_state = create_initial_state("anonymous")
                
                result = await run_graph_with_message(graph, session_state, data)
                session_state = result["state"]
                
                await socket.send_text(result["response"])
            
            except Exception as e:
                print(f"Message processing error: {e}")
                error_response = {
                    "response": "Sorry, I encountered an error processing your message.",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await socket.send_text(json.dumps(error_response))
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for user: {user_id}, session: {session_id}")
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        await socket.close()


#NOTE: This is where we merge both MCP and FastAPI app

combined_app = FastAPI(
    routes=[
        *mcp_app.routes,
        *app.routes,
    ],
    lifespan=mcp_app.lifespan
)

