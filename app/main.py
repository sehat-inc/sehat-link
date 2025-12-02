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
import uuid
import asyncio
from redis.asyncio import Redis

from routers import  doctor, follow_up, hospital, logout, patient, appointment, degraded

from api.mcp.server import mcp_app
from core.langgraph.agent import build_triage_agent
from core.langgraph.utils.state import MedicalAgentState
from routers.patient import load_initial_state_from_db
from core.langgraph.utils.tool_manager import MCPToolManager

from core.logging import get_logger


logger = get_logger("MAIN APP LOGIC")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
logger.info(f"REDIS_URL: {REDIS_URL}")

checkpointer = None

# Cached compiled graph for performance
_compiled_graph = None
_graph_lock = asyncio.Lock()

@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    """
    Handles startup/shutdown for both Redis and MCP.
    We nest the Context Managers to keep them both alive.
    """
    async with AsyncRedisSaver.from_conn_string(REDIS_URL) as cp:
        await cp.asetup()
        
        global checkpointer
        checkpointer = cp
        logger.info("Redis Checkpointer Initialized and Global Set")

        async with mcp_app.lifespan(app) as mcp_ctx:
            yield
        
        logger.info("Redis Connection Closed")
        checkpointer = None


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello from Sehat-Link API!"}


app.include_router(doctor.router)
app.include_router(patient.router)
app.include_router(hospital.router)
app.include_router(follow_up.router)
app.include_router(logout.router)
app.include_router(appointment.router)
app.include_router(degraded.router)  # Degraded mode - no auth required


async def get_compiled_graph():
    """
    Get or create the compiled graph with caching for performance.
    Saves ~10-50ms per request by avoiding recompilation.
    """
    global _compiled_graph
    
    if _compiled_graph is None:
        async with _graph_lock:
            # Double-check after acquiring lock
            if _compiled_graph is None:
                graph_builder = build_triage_agent()
                _compiled_graph = graph_builder.compile(checkpointer=checkpointer)
                logger.info("Graph compiled and cached")
    
    return _compiled_graph


async def run_graph_for_user_streaming(socket: WebSocket, user_id: int, user_message: Any):
    """
    Stream graph execution events to WebSocket for real-time updates.
    
    Streaming Strategy:
    - Each agent makes 2 LLM calls:
      1. First call (with tools): "thinking" process → stored in 'messages'
      2. Second call (structured output): parsed response → stored in 'user_messages'
    
    We stream:
    - 'thinking' type for the first LLM call (agent's internal reasoning) - NOT for chat display
    - 'token' type for the second LLM call (actual user-facing response) - FOR chat display
    """
    config = {
        "configurable": {
            "thread_id": f"user_{user_id}"
        }
    }
    
    graph = await get_compiled_graph()
    
    # Get initial state
    try:
        current_state = await graph.aget_state(config)
    except Exception as e:
        logger.error(f"Error in Getting Current State: {e}")
        return None

    try:
        if current_state and current_state.values:
            state_values = current_state.values
        else:
            logger.info("Getting State from DB")
            state_values = await load_initial_state_from_db(user_id)
    except Exception as e:
        logger.error(f"Error in Getting Graph State: {e}")
        return None
    
    updated_state = {
        **state_values,
        "user_messages": [HumanMessage(content=user_message)],
        "messages": [HumanMessage(content=user_message)]
    }
    
    # Track streaming state
    current_node = None
    final_result = None
    
    # Track which LLM call we're on per node (first = thinking, second = response)
    # The pattern is: each agent calls LLM twice
    # - First call: model_with_tools.ainvoke() → thinking/reasoning
    # - Second call: structured_llm.ainvoke() → user response
    llm_call_count_per_node = {}
    
    # Track if we've sent any response tokens for deduplication
    has_sent_response_tokens = False
    
    # Send initial acknowledgment
    await socket.send_text(json.dumps({
        "type": "stream_start",
        "message": "Processing your message..."
    }))
    
    try:
        # Stream events from the graph execution
        async for event in graph.astream_events(updated_state, config=config, version="v2"):
            event_type = event.get("event")
            
            # Node starting - inform user which agent is processing
            if event_type == "on_chain_start":
                node_name = event.get("name", "")
                if node_name in ["triage", "symptom", "program", "doctor", "prescription", "language", "urgency"]:
                    current_node = node_name
                    llm_call_count_per_node[node_name] = 0
                    await socket.send_text(json.dumps({
                        "type": "node_start",
                        "node": current_node,
                        "message": f"Agent '{current_node}' is processing..."
                    }))
            
            # Track LLM invocation starts to know which call we're on
            elif event_type == "on_chat_model_start":
                if current_node and current_node in llm_call_count_per_node:
                    llm_call_count_per_node[current_node] += 1
                    call_num = llm_call_count_per_node[current_node]
                    
                    if call_num == 1:
                        # First LLM call - thinking phase
                        await socket.send_text(json.dumps({
                            "type": "thinking_start",
                            "node": current_node,
                            "message": "Analyzing..."
                        }))
                    elif call_num == 2:
                        # Second LLM call - response generation
                        # Reset the response token flag for new response
                        has_sent_response_tokens = False
                        await socket.send_text(json.dumps({
                            "type": "response_start",
                            "node": current_node,
                            "message": "Generating response..."
                        }))
            
            # LLM streaming tokens
            elif event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content"):
                    content = chunk.content
                    if content and current_node:
                        # Determine if this is thinking (1st call) or response (2nd call)
                        call_num = llm_call_count_per_node.get(current_node, 1)
                        
                        # Only stream the first LLM call (thinking)
                        # The second call is structured output (JSON) and should not be streamed
                        # The actual response is extracted and sent in stream_end
                        if call_num == 1:
                            stream_type = "thinking"
                            
                            # Handle different content formats
                            if isinstance(content, str):
                                await socket.send_text(json.dumps({
                                    "type": stream_type,
                                    "content": content,
                                    "node": current_node
                                }))
                            elif isinstance(content, list) and len(content) > 0:
                                for item in content:
                                    if isinstance(item, dict) and "text" in item:
                                        text = item["text"]
                                        await socket.send_text(json.dumps({
                                            "type": stream_type,
                                            "content": text,
                                            "node": current_node
                                        }))
            
            # Tool calls - inform user about tool usage (part of thinking)
            elif event_type == "on_tool_start":
                tool_name = event.get("name", "unknown")
                await socket.send_text(json.dumps({
                    "type": "tool_start",
                    "tool": tool_name,
                    "message": f"Searching: {tool_name}..."
                }))
            
            elif event_type == "on_tool_end":
                tool_name = event.get("name", "unknown")
                await socket.send_text(json.dumps({
                    "type": "tool_end",
                    "tool": tool_name,
                    "message": f"Done: {tool_name}"
                }))
            
            # Chain/Node completion
            elif event_type == "on_chain_end":
                node_name = event.get("name", "")
                output = event.get("data", {}).get("output")
                
                # Capture final result from main graph completion
                if node_name == "LangGraph" and output:
                    final_result = output

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        await socket.send_text(json.dumps({
            "type": "error",
            "message": f"Streaming error: {str(e)}"
        }))
        # Try to get state directly as fallback
        try:
            final_state = await graph.aget_state(config)
            if final_state and final_state.values:
                return final_state.values
        except Exception:
            pass
        return None
    
    # If final_result is still None, try to get the latest state
    if final_result is None:
        logger.warning("No final result captured from stream, fetching state directly")
        try:
            final_state = await graph.aget_state(config)
            if final_state and final_state.values:
                final_result = final_state.values
        except Exception as e:
            logger.error(f"Failed to get final state: {e}")
    
    return final_result



@app.websocket("/ws/chat")
async def chat_socket(socket: WebSocket):
    """
    WebSocket endpoint for real-time chat with triage agent.
    Supports streaming responses for reduced perceived latency.
    """
    await socket.accept()
    logger.info("WebSocket connection accepted")

    def extract_text(message: AIMessage) -> str:
        c = message.content
        if isinstance(c, list) and len(c) > 0:
            first = c[0]
            if isinstance(first, dict):
                return first.get("text", "")
            if isinstance(first, str):
                return first
        return str(c)

    try:
        while True:
            # Receive message from client
            data = await socket.receive_text()
            payload = json.loads(data)
            user_msg = payload.get("message", "")
            user_id = payload.get("user_id", "")
            image_data = payload.get("image", "")
            # Allow client to opt-out of streaming for backwards compatibility
            enable_streaming = payload.get("streaming", True)

            logger.info(f"Received message from WebSocket, streaming={enable_streaming}")
            
            if not user_id:
                await socket.send_text(json.dumps({
                    "type": "error",
                    "response": "Error: Missing user_id",
                    "error": "user_id required"
                }))
                continue
            
            try:
                # Prepare message content
                if image_data:
                    # Ensure base64 image has proper data URL prefix
                    if not image_data.startswith("data:image/") and not image_data.startswith("http"):
                        # Detect image type from base64 header and add proper prefix
                        if image_data.startswith("/9j/"):
                            image_data = f"data:image/jpeg;base64,{image_data}"
                        elif image_data.startswith("iVBORw"):
                            image_data = f"data:image/png;base64,{image_data}"
                        elif image_data.startswith("R0lGOD"):
                            image_data = f"data:image/gif;base64,{image_data}"
                        elif image_data.startswith("UklGR"):
                            image_data = f"data:image/webp;base64,{image_data}"
                        else:
                            # Default to jpeg if unknown
                            image_data = f"data:image/jpeg;base64,{image_data}"
                    msg_content = [
                        {"type": "text", "text": user_msg or "Analyze this Prescription"},
                        {"type": "image_url", "image_url": image_data}
                    ]
                else:
                    msg_content = user_msg

                if enable_streaming:
                    # Use streaming for real-time updates
                    result = await run_graph_for_user_streaming(socket, int(user_id), msg_content)
                else:
                    # Fallback to non-streaming for legacy clients
                    result = await run_graph_for_user(int(user_id), msg_content)
                
                logger.info("Graph execution completed")
                
                # Extract final response data
                if result is not None:
                    ai_msgs = [m for m in result.get("user_messages", []) if isinstance(m, AIMessage)]
                    final_response = extract_text(ai_msgs[-1]) if ai_msgs else None
                    logger.info(f"Final Response: {final_response}")
                    r_agent = result.get("current_agent", "Unknown")
                    call_trigger = result.get("call_trigger", False)
                    prescription_data = result.get("prescription_data")
                else:
                    logger.error("Result is None")
                    r_agent = "Unknown"
                    final_response = "I apologize, but I encountered an issue processing your request."
                    call_trigger = False
                    prescription_data = None
                
                # Send final complete response
                response_payload = {
                    "type": "stream_end",
                    "call_trigger": call_trigger,
                    "response": final_response,
                    "agent": r_agent,
                    "prescription_data": prescription_data
                }
                
                await socket.send_text(json.dumps(response_payload))
                
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                error_response = {
                    "type": "error",
                    "response": "Sorry, I encountered an error processing your message.",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await socket.send_text(json.dumps(error_response))
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for user") 
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await socket.close()


async def run_graph_for_user(user_id: int, user_message: Any):
    """
    Non-streaming helper func for running the graph (backwards compatibility).
    """
    config = {
        "configurable": {
            "thread_id": f"user_{user_id}"
        }
    }
    
    graph = await get_compiled_graph()

    try:
        current_state = await graph.aget_state(config)
    except Exception as e:
        logger.error(f"Error in Getting Current State: {e}")
        return None

    try:
        if current_state and current_state.values:
            state_values = current_state.values
        else:
            logger.info("Getting State from DB")
            state_values = await load_initial_state_from_db(user_id)
    except Exception as e:
        logger.error(f"Error in Getting Graph State: {e}")
        return None
    
    updated_state = {
        **state_values,
        "user_messages": [HumanMessage(content=user_message)],
        "messages": [HumanMessage(content=user_message)]
    }

    result = await graph.ainvoke(updated_state, config=config)
    return result


#NOTE: This is where we merge both MCP and FastAPI app

# combined_app = app
# combined_app.mount("/mcp", mcp_app)

combined_app = FastAPI(
    title="Healthcare with MCP",
    routes=[
        *mcp_app.routes,
        *app.routes,
    ],
    lifespan=combined_lifespan,
)


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
