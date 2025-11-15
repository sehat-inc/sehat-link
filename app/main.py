from typing import Optional, Any, List, Dict
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status
    )
import json
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from contextlib import asynccontextmanager
import os

from routers import doctor, patient, hospital, follow_up

from api.mcp.server import mcp_app
from core.langgraph.agent import build_triage_agent
from core.langgraph.utils.state import MedicalAgentSession, MedicalAgentState
from routers.patient import load_initial_state_from_db

from core.logging import get_logger


logger = get_logger("MAIN APP LOGIC")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
logger.info(f"REDIS_URL: {REDIS_URL}")


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
app.include_router(follow_up.router)


# Helper to run graph with proper state management
async def run_graph_for_user(builder, user_id: int, user_message: str):
    """
    Helper func for running the graph and returing the result
    """
    config = {
        "configurable": {
            "thread_id": str(user_id)
        }
    }

    async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
        await checkpointer.asetup()
        graph = builder.compile(checkpointer=checkpointer)
        
        try:
            current_state = await graph.aget_state(config)
            logger.info(f"Current State: {current_state}")

            if current_state and current_state.values:
                state_values = current_state.values
            else:
                state_values = await load_initial_state_from_db(user_id)
        except Exception as e:
            logger.error(f"Error in Getting Graph State: {e}")
            return
        
        logger.info(f"STATE VALUES: {state_values}")

        updated_state = {
            **state_values,
            "messages": [HumanMessage(content=user_message)]
        }

        result = await graph.ainvoke(updated_state, config=config)

        return result

@app.websocket("/ws/chat")
async def chat_socket(socket: WebSocket):
    """
    WebSocket endpoint for real-time chat with triage agent.
    """
    await socket.accept()
    
    graph = build_triage_agent()
    logger.info("Graph Built")

    try:
        while True:
            # Receive message from client
            data = await socket.receive_text()
            payload = json.loads(data)
            user_msg = payload.get("message", "")
            user_id = payload.get("user_id", "")

            logger.info("Recieve Message from WebSocket")
            
            if not user_id:
                await socket.send_text(json.dumps({
                    "response": "Error: Missing user_id",
                    "error": "user_id required"
                }))
                continue
            
            try:
                result = await run_graph_for_user(graph, int(user_id), user_msg)
                
                if result is not None:
                    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
                    final_response = ai_msgs[-1].content if ai_msgs else None
                    r_agent = result.get("current_agent", "Unknown")
                else:
                    logger.error(f"Result is None")
                    r_agent = "Unknown"
                    final_response = "Error 404"

                
                logger.info(f"RESULT: {result}")

                # Send response back to client
                response_payload = {
                    "response": final_response,
                    "agent": r_agent
                }
                
                await socket.send_text(json.dumps(response_payload))
                
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                error_response = {
                    "response": "Sorry, I encountered an error processing your message.",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await socket.send_text(json.dumps(error_response))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user") 
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
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
