from fastmcp import FastMCP
from dotenv import load_dotenv
import os

from api.mcp.tools.vector_db import PineconeQuery

load_dotenv()

PINECONE_API = os.getenv("PINECONE_API")
PC_INDEX_NAME = os.getenv("PC_INDEX_NAME")

mcp = FastMCP("sehat-link")

pc_ctx_tool = PineconeQuery(PINECONE_API, PC_INDEX_NAME)

mcp.tool(
    pc_ctx_tool.smart_query,
    name="Pinecone Smart Query",
    description="""Intelligently query Pinecone by first breaking down complex questions
    into sub-queries, then aggregating results. Uses LLM to decompose questions.""",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True
    }
)

mcp.tool(
    pc_ctx_tool.direct_query,
    name="Pinecone Direct Query",
    description="Direct Pinecone query without decomposition for simple lookups"
)

mcp_app = mcp.http_app(path="/mcp")

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
