from fastmcp import FastMCP
from dotenv import load_dotenv
import os

from api.mcp.tools.vector_db import PineconeQuery

load_dotenv()

PINECONE_API = os.getenv("PINECONE_API")
PC_INDEX_NAME = os.getenv("PC_INDEX_NAME")
PC_INDEX_NAMEV2 = os.getenv("PC_INDEX_NAMEV2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""

mcp = FastMCP("sehat-link")

#NOTE: SYMPTOM AGENT
pc_ctx_tool = PineconeQuery(str(PINECONE_API), str(OPENAI_API_KEY), str(PC_INDEX_NAME))
#NOTE: ELIGIBILITY AGENT
pc_ctx_tool2 = PineconeQuery(str(PINECONE_API), str(OPENAI_API_KEY), str(PC_INDEX_NAMEV2))

mcp.tool(
    pc_ctx_tool.smart_query,
    name="Symptom Knowledge Base Smart Query",
    description="""Intelligently query Pinecone by first breaking down complex questions
    into sub-queries, then aggregating results. Uses LLM to decompose questions.""",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True
    }
)

mcp.tool(
    pc_ctx_tool.direct_query,
    name="Symptom Knowledge Base Direct Query",
    description="Direct Pinecone query without decomposition for simple lookups"
)

mcp.tool(
    pc_ctx_tool2.smart_query,
    name="Programme Eligibility KB Smart Query",
    description="""Intelligently query Pinecone by first breaking down complex questions
    into sub-queries, then aggregating results. Uses LLM to decompose questions.""",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True
    }
)

mcp.tool(
    pc_ctx_tool2.direct_query,
    name="Programme Eligibility KB Direct Query",
    description="Direct Pinecone query without decomposition for simple lookups"
)


mcp_app = mcp.http_app(path="/mcp")

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
