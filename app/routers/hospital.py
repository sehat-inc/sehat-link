from database import get_supabase 
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/hospitals", tags=["Hospitals"])

@router.get("/list")
def get_hospitals_list():
    try:
        supabase = get_supabase()
        response = supabase.table("hospitals").select("id, name").execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch hospitals list.")
