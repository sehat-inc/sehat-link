from fastmcp import FastMCP
from fastmcp.experimental.sampling.handlers.openai import OpenAISamplingHandler
from dotenv import load_dotenv
import os
from openai import OpenAI

from .tools.vector_db import PineconeQuery
from .tools.facility_finder import FacilityFinder 
from .tools.baba_qadeer_tool import ask_baba_qadeer

load_dotenv()

PINECONE_API = os.getenv("PINECONE_API")
PC_INDEX_NAME = os.getenv("PC_INDEX_NAME")
PC_INDEX_NAMEV2 = os.getenv("PC_INDEX_NAMEV2")
PC_INDEX_NAMEV3 = os.getenv("PC_INDEX_NAMEV3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

mcp = FastMCP(
    name="sehat-link",
    sampling_handler=OpenAISamplingHandler(
        default_model="gpt-4o-mini",
        client=OpenAI(
            api_key=OPENAI_API_KEY,
        ),
    ),
    sampling_handler_behavior="fallback",
)

#NOTE: SYMPTOM AGENT
symptom_tool = PineconeQuery(str(PINECONE_API), str(OPENAI_API_KEY), str(PC_INDEX_NAME))
#NOTE: ELIGIBILITY AGENT
program_tool = PineconeQuery(str(PINECONE_API), str(OPENAI_API_KEY), str(PC_INDEX_NAMEV2))

doctor_tool = PineconeQuery(str(PINECONE_API), str(OPENAI_API_KEY), str(PC_INDEX_NAMEV3))

#NOTE: Closest Facility Finder
facility_tool = FacilityFinder(
    gmaps_api_key=str(GOOGLE_MAPS_API_KEY), 
    supabase_url=str(SUPABASE_URL), 
    supabase_key=str(SUPABASE_KEY)
)



mcp.tool(
    ask_baba_qadeer,
    name="Baba_Qadeer_Tool",
    description="Returns Returns a random line of wisdom from Baba Qadeer."
)

mcp.tool(
    symptom_tool.smart_query,
    name="Symptom_Knowledge_Base_Smart_Query",
    description="""Intelligently query Pinecone by first breaking down complex questions
    into sub-queries, then aggregating results. Uses LLM to decompose questions.""",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True
    }
)

mcp.tool(
    symptom_tool.direct_query,
    name="Symptom_Knowledge_Base_Direct_Query",
    description="Direct Pinecone query without decomposition for simple lookups"
)

mcp.tool(
    program_tool.smart_query,
    name="Programme_Eligibility_KB_Smart_Query",
    description="""Intelligently query Pinecone by first breaking down complex questions
    into sub-queries, then aggregating results. Uses LLM to decompose questions.
    
    NAMESPACE SELECTION GUIDE:
    - USE '__default__' for symptom/medical retrieval
    - USE 'eligibility-namespace' for program related retrieval
    """,
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True
    }
)

mcp.tool(
    program_tool.direct_query,
    name="Programme_Eligibility_KB_Direct_Query",
    description="Direct Pinecone query without decomposition for simple lookups"
)

mcp.tool(
    facility_tool.find_nearby_facilities,
    name="Find_Nearest_Medical_Facility",
    description="""Locates the nearest hospital, clinic, or pharmacy.
    IMPORTANT: You must pass the current User's ID as 'user_id' to this tool.
    The tool will look up the user's location in the database automatically.""",
    annotations={
        "readOnlyHint": True
    }
)

    
mcp.tool(
    doctor_tool.smart_query_with_filters,
    name="Doctor_KB_Smart_Query",
    description="""Intelligently query Pinecone by first breaking down complex questions
    into sub-queries, then aggregating results. Uses LLM to decompose questions.
    
    NAMESPACE SELECTION GUIDE:
    - USE '__default__' for symptom/medical retrieval
    - USE 'dcotor-namespace' for doctor related retrieval

    FILTERING (All use OR logic):
    - Specialty filter: Matches ANY of the provided specialties
      Example: ["Cardiologist", "General Physician"] finds doctors who are EITHER cardiologists OR general physicians
    - City filter: Matches ANY of the provided cities
      Example: ["Lahore", "Karachi"] finds doctors in EITHER Lahore OR Karachi
    - Combined filters: Doctor must match at least one specialty AND at least one city
      Example: specialties=["Cardiologist"], cities=["Lahore", "Karachi"] 
      finds cardiologists in (Lahore OR Karachi)
    
    COMMON USE CASES:
    1. Find specific specialists in one city:
       specialties=["Cardiologist"], cities=["Lahore"]
    
    2. Find any of several specialists across multiple cities:
       specialties=["Cardiologist", "General Physician", "Dermatologist"], 
       cities=["Lahore", "Karachi", "Islamabad"]
    
    3. Find all doctors in specific cities (no specialty filter):
       cities=["Lahore", "Karachi"]
    
    4. Find specific specialty across all cities (no city filter):
       specialties=["Pediatrician"]
    """,
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True
    }
)

mcp.tool(
    doctor_tool.direct_query,
    name="Doctor_KB_Direct_Query",
    description="Direct Pinecone query without decomposition for simple lookups"
)

mcp_app = mcp.http_app(path="/mcp")

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
