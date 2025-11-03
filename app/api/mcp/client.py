import asyncio
from fastmcp import Client

client = Client("http://localhost:8000/mcp")

async def test_agent(user_message: str):
    """
    Simulation of LangGraph Agent
    """

    await asyncio.sleep(0.3)
    return f"Agent Recieved: {user_message}"
