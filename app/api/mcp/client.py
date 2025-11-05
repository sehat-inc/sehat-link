import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


transport = StreamableHttpTransport(url="http://localhost:8080/mcp")
client = Client(transport)

async def test_agent(user_message: str):
    """
    Simulation of LangGraph Agent
    """

    await asyncio.sleep(0.3)
    return f"Agent Recieved: {user_message}"
