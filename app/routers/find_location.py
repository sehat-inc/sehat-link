import os
import logging
import googlemaps
from supabase import create_client, Client
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Optional

from auth_utils import get_current_patient_id
from core.logging import get_logger

# --- Configuration ---
router = APIRouter(prefix="/facilities", tags=["Medical Facilities"])
logger = get_logger("NEARBY FACILITIES ENDPOINT")

GMAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Clients Globaly (or use a dependency injection pattern)
gmaps = googlemaps.Client(key=GMAPS_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Pydantic Models for Response ---
class FacilityLocation(BaseModel):
    lat: float
    lng: float

class FacilityItem(BaseModel):
    name: str
    address: str | None = None
    rating: float | str | None = None
    status: str | None = None
    map_link: str

class FacilityResponse(BaseModel):
    search_location: FacilityLocation
    facility_type: str
    count: int
    facilities: List[FacilityItem]

# --- Service Logic ---
class FacilityService:
    def __init__(self, table_name: str = "patient_locations"): 
        self.table_name = table_name

    def get_user_coordinates(self, user_id: str) -> tuple[float, float]:
        """Fetches lat/lng from Supabase."""
        try:
            response = supabase.table(self.table_name)\
                .select("latitude, longitude")\
                .eq("patient_id", user_id)\
                .execute()

            if not response.data:
                return None

            user_data = response.data[0]
            
            # Handle potential key variations
            lat = user_data.get('latitude') or user_data.get('lat')
            lng = user_data.get('longitude') or user_data.get('long') or user_data.get('lng')
            
            if lat is not None and lng is not None:
                return float(lat), float(lng)
            return None
            
        except Exception as e:
            logger.error(f"Supabase DB Error: {e}")
            raise HTTPException(status_code=500, detail="Database error while fetching location")

    def find_places(self, lat: float, lng: float, facility_type: str, keyword: str = None):
        """Queries Google Maps."""
        try:
            places_result = gmaps.places_nearby(
                location=(lat, lng),
                rank_by='distance',  # strict distance sorting
                type=facility_type,
                keyword=keyword
            )
            
            results = []
            if 'results' in places_result:
                for place in places_result['results'][:5]: # Top 5
                    p_lat = place['geometry']['location']['lat']
                    p_lng = place['geometry']['location']['lng']
                    
                    results.append({
                        "name": place.get('name'),
                        "address": place.get('vicinity'),
                        "rating": place.get('rating', 'N/A'),
                        "status": place.get('business_status'),
                        "map_link": f"https://www.google.com/maps/search/?api=1&query={p_lat},{p_lng}"
                    })
            return results
        except Exception as e:
            logger.error(f"Google Maps API Error: {e}")
            raise HTTPException(status_code=502, detail="Failed to fetch data from Maps provider")

# --- The Endpoint ---

@router.get("/nearby", response_model=FacilityResponse)
async def get_nearby_facilities(
    facility_type: str = Query("hospital", description="Type of facility (hospital, pharmacy, doctor)"),
    keyword: Optional[str] = Query(None, description="Specialist keyword e.g. cardiologist"),
    user_id: str = Depends(get_current_patient_id) # <--- Authenticated User ID
):
    """
    Finds medical facilities near the authenticated patient's location.
    """
    logger.info(f"DEBUG: Extracted user_id = {user_id}")  # <-- Add this line
    service = FacilityService(table_name="patient_locations") 
    
    
    # 1. Get Location
    coords = service.get_user_coordinates(user_id)
    
    if not coords:
        raise HTTPException(
            status_code=404, 
            detail="User location not found. Please update your profile with coordinates."
        )
    
    lat, lng = coords
    
    # 2. Search Maps
    facilities = service.find_places(lat, lng, facility_type, keyword)
    
    # 3. Return formatted response
    return {
        "search_location": {"lat": lat, "lng": lng},
        "facility_type": facility_type,
        "count": len(facilities),
        "facilities": facilities
    }