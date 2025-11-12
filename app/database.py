import os
from dotenv import load_dotenv
from typing import List, Dict, TypedDict, TypeVar, Type, Any
from supabase import create_client, Client
from core.langgraph.utils.state import MedicalAgentState

load_dotenv()

supabase: Client | None = None

def get_supabase():
    global supabase
    if supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        supabase = create_client(str(url), str(key))
    return supabase


T = TypeVar("T", bound=dict[str, Any])

def fetch_longterm_by_user_id(supabase: Client, table_name: str, user_id: str, record_type: Type[T]) -> List[T]:

    response = supabase.table(table_name).select("*").eq("user_id", user_id).execute()
    
    error = getattr(response, "error", None)
    if error:
        message = getattr(error, "message", str(error))
        raise RuntimeError(f"Supabase error: {message}")

    rows = getattr(response, "data", None)
    if not isinstance(rows, list):
        raise TypeError(f"Unexpected response format: {type(rows)}")

    valid_keys = getattr(record_type, "__annotations__", {}).keys()

    typed_rows: List[T] = []
    for row in rows:
        if isinstance(row, dict):
            filtered = {k: v for k, v in row.items() if k in valid_keys}
            typed_rows.append(record_type(**filtered))

    return typed_rows

    
def save_disease_report(state: MedicalAgentState, supabase):
    """
    Simple function to save one disease report at session end.
    Uses data from longterm_session state.
    """
    patient_id = state["user_id"]
    reported_city = state.get("user_location", "Unknown")
    
    # Get disease name (from detected_problem_type)
    disease_name = state.get("detected_problem_type", "Unknown Disease")
    
    # Map detected_urgency to severity_level
    urgency_to_severity = {
        "Emergency": "Critical",
        "High": "Severe",
        "Medium": "Moderate",
        "Low": "Mild"
    }
    severity_level = urgency_to_severity.get(state.get("detected_urgency", "Medium"), "Unknown")

    try:
        supabase.table("disease_reports").insert({
            "patient_id": patient_id,
            "disease_name": disease_name,
            "reported_city": reported_city,
            "severity_level": severity_level
        }).execute()
        print(f"✅ Saved disease report: {disease_name} ({severity_level}) for patient {patient_id}")
    except Exception as e:
        print(f"❌ Failed to save disease report: {e}")