"""
Degraded Mode Router

Public endpoints for offline/low-bandwidth resilience.
No authentication required.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from degraded_mode.memory import degraded_memory
from degraded_mode.simple_agent import degraded_agent
from core.logging import get_logger


logger = get_logger("DEGRADED_MODE_ROUTER")

router = APIRouter(prefix="/degraded", tags=["degraded"])


# Request/Response Models
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: list
    timestamp: str


class NewSessionResponse(BaseModel):
    session_id: str
    message: str


class EndSessionRequest(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    mode: str
    active_sessions: int
    llm_provider: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if degraded mode is available and operational.
    """
    import os
    from datetime import datetime
    
    try:
        session_count = degraded_memory.get_session_count()
        llm_provider = os.getenv("DEGRADED_LLM_PROVIDER", "openai")
        
        return HealthResponse(
            status="operational",
            mode="degraded",
            active_sessions=session_count,
            llm_provider=llm_provider
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Degraded mode unavailable")


@router.post("/new-session", response_model=NewSessionResponse)
async def create_new_session():
    """
    Create a new chat session.
    Returns session_id to use for subsequent messages.
    """
    try:
        session_id = degraded_memory.create_session()
        logger.info(f"Created new degraded mode session: {session_id}")
        
        return NewSessionResponse(
            session_id=session_id,
            message="New session created successfully"
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send message and get response.
    
    If session_id is not provided, a new session will be created.
    """
    from datetime import datetime
    
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = degraded_memory.create_session()
            logger.info(f"Auto-created session: {session_id}")
        
        # Validate session exists
        session = degraded_memory.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired. Please create a new session."
            )
        
        # Add user message to history
        degraded_memory.add_message(session_id, "user", request.message)
        
        # Get chat history
        chat_history = degraded_memory.get_messages(session_id, limit=10)
        
        # Convert to LLM format (exclude timestamps)
        llm_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in chat_history[:-1]  # Exclude current message
        ]
        
        # Process message with agent
        logger.info(f"Processing message for session {session_id}")
        result = await degraded_agent.process_message(
            user_message=request.message,
            chat_history=llm_history
        )
        
        # Add assistant response to history
        degraded_memory.add_message(session_id, "assistant", result["response"])
        
        return ChatResponse(
            session_id=session_id,
            response=result["response"],
            sources=result.get("sources", []),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/end-session")
async def end_session(request: EndSessionRequest):
    """
    End chat session and cleanup memory.
    """
    try:
        success = degraded_memory.end_session(request.session_id)
        
        if success:
            logger.info(f"Ended session: {request.session_id}")
            return {
                "success": True,
                "message": "Session ended successfully",
                "session_id": request.session_id
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        raise HTTPException(status_code=500, detail="Failed to end session")


@router.get("/stats")
async def get_stats():
    """
    Get degraded mode statistics (for monitoring).
    """
    try:
        session_count = degraded_memory.get_session_count()
        
        return {
            "active_sessions": session_count,
            "max_age_minutes": degraded_memory.max_age_minutes,
            "status": "operational"
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")
