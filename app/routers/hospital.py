from database import supabase
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/hospitals", tags=["Hospitals"])

@router.get("/list")
def get_hospitals_list():
    try:
        response = supabase.table("hospitals").select("id, name").execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch hospitals list.")
