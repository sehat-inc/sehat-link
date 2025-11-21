"""
In-Memory Session Management for Degraded Mode

Completely isolated from main app's Redis memory.
Auto-cleanup of expired sessions.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uuid
from dataclasses import dataclass, field


@dataclass
class ChatSession:
    """Represents a single chat session"""
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class DegradedMemory:
    """
    In-memory session storage for degraded mode.
    Isolated from main app's Redis-based memory.
    """
    
    def __init__(self, max_age_minutes: int = 30):
        self.sessions: Dict[str, ChatSession] = {}
        self.max_age_minutes = max_age_minutes
    
    def create_session(self) -> str:
        """Create new chat session and return session_id"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ChatSession(session_id=session_id)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get session by ID"""
        self.cleanup_expired()  # Auto-cleanup on access
        return self.sessions.get(session_id)
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add message to session history"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        session.last_active = datetime.now()
        return True
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get message history for session"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = session.messages
        if limit:
            messages = messages[-limit:]
        return messages
    
    def end_session(self, session_id: str) -> bool:
        """Explicitly end and cleanup session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def cleanup_expired(self):
        """Remove sessions older than max_age_minutes"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=self.max_age_minutes)
        
        expired = [
            sid for sid, session in self.sessions.items()
            if session.last_active < cutoff
        ]
        
        for sid in expired:
            del self.sessions[sid]
        
        return len(expired)
    
    def get_session_count(self) -> int:
        """Get active session count"""
        self.cleanup_expired()
        return len(self.sessions)


# Global instance (isolated from main app)
degraded_memory = DegradedMemory(max_age_minutes=30)
