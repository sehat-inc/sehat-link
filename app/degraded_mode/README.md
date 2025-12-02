# Degraded Mode - Offline/Low-Bandwidth Resilience

## Overview

Degraded mode provides 10% offline resilience when main services are unavailable. It operates independently with:

- **No authentication required** - Public access
- **No Redis/Database dependencies** - Fully offline capable
- **Static JSON knowledge base** - Pre-loaded medical information
- **Isolated memory system** - Doesn't interfere with main app
- **Swappable LLM provider** - OpenAI or local models

## Architecture

```
degraded_mode/
├── __init__.py              # Module initialization
├── memory.py                # In-memory session management
├── llm_provider.py          # LLM abstraction (OpenAI/Local)
├── knowledge_base.py        # JSON search and loading
├── simple_agent.py          # Lightweight agent logic
└── knowledge/               # JSON knowledge files
    ├── medical_assistance.json
    ├── government_programs.json
    ├── hospitals_by_city.json
    └── emergency_info.json
```

## API Endpoints

### Health Check
```bash
GET /degraded/health
```

### Create Session
```bash
POST /degraded/new-session
Response: {"session_id": "uuid", "message": "..."}
```

### Chat
```bash
POST /degraded/chat
Body: {
  "session_id": "uuid",  # Optional, auto-created if missing
  "message": "I have a fever"
}
Response: {
  "session_id": "uuid",
  "response": "...",
  "sources": [...],
  "timestamp": "..."
}
```

### End Session
```bash
POST /degraded/end-session
Body: {"session_id": "uuid"}
```

## Configuration

### Environment Variables

```bash
# LLM Provider (default: openai)
DEGRADED_LLM_PROVIDER=openai  # or "local"

# OpenAI Configuration
OPENAI_API_KEY=your_key_here
DEGRADED_OPENAI_MODEL=gpt-4o-mini

# Local LLM Configuration (if using Ollama)
DEGRADED_LLM_PROVIDER=local
DEGRADED_LOCAL_MODEL=phi3:mini
OLLAMA_BASE_URL=http://localhost:11434
```

### Switching to Local LLM

1. Install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. Pull a model:
```bash
ollama pull phi3:mini
```

3. Install Python package:
```bash
pip install ollama
```

4. Set environment:
```bash
export DEGRADED_LLM_PROVIDER=local
export DEGRADED_LOCAL_MODEL=phi3:mini
```

## Memory Management

- **Auto-cleanup**: Sessions expire after 30 minutes of inactivity
- **Isolated**: Completely separate from main app's Redis memory
- **In-memory**: No persistence, cleared on server restart
- **Manual cleanup**: Use `/degraded/end-session` endpoint

## Knowledge Base

### JSON File Structure

Each JSON file follows a searchable format with keywords:

```json
{
  "symptoms": [
    {
      "name": "Fever",
      "keywords": ["fever", "temperature", "hot", "chills"],
      "description": "Elevated body temperature...",
      "severity": "mild",
      "advice": "Monitor temperature, stay hydrated...",
      "red_flags": ["temperature > 103°F", "persistent > 3 days"],
      "when_to_seek_help": "If fever persists or worsens"
    }
  ]
}
```

## Differences from Main App

| Feature | Main App | Degraded Mode |
|---------|----------|---------------|
| Authentication | Required (JWT) | None (public) |
| Memory | Redis + Supabase | In-memory dict |
| Agent | Complex LangGraph | Simple search + LLM |
| Personalization | Full medical history | Generic responses |
| Tools | MCP tools, DB queries | JSON search only |
| Persistence | Permanent | Session-based |

## Limitations

⚠️ **Cannot access**:
- Patient medical records
- Personalized recommendations
- Real-time doctor availability
- Appointment booking
- Prescription history

✅ **Can provide**:
- General medical information
- First aid guidance
- Government program info
- Hospital listings
- Emergency contacts

## Testing

```bash
# Health check
curl http://localhost:8000/degraded/health

# Create session
curl -X POST http://localhost:8000/degraded/new-session

# Send message
curl -X POST http://localhost:8000/degraded/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What should I do for a fever?"}'

# End session
curl -X POST http://localhost:8000/degraded/end-session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id"}'
```

## Monitoring

```bash
# Get statistics
curl http://localhost:8000/degraded/stats
```

## Safety

- Always includes medical disclaimers
- Recommends professional consultation
- Provides emergency contact information
- No prescription or diagnosis capabilities
