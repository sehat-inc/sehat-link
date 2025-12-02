import googlemaps
from supabase import create_client, Client
from typing import Annotated, Dict, Optional
from pydantic import Field
from fastmcp import Context

class FacilityFinder:
    """
    A class to find nearby medical facilities using user location from Supabase
    and the Google Maps Places API.
    """
    def __init__(self, gmaps_api_key: str, supabase_url: str, supabase_key: str, table_name: str = "patient_locations"):
        self.gmaps = googlemaps.Client(key=gmaps_api_key)
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.table_name = "patient_locations"

    def _get_coordinates_from_supabase(self, user_id: str) -> Optional[tuple[float, float]]:
        """
        Fetches latitude and longitude from the Supabase table for a specific user.
        """
        table_name = "patient_locations"
        try:
            response = self.supabase.table(table_name)\
                .select("latitude, longitude")\
                .eq("patient_id", user_id)\
                .execute()

            if response.data and len(response.data) > 0:
                user_data = response.data[0]
                # Ensure keys match your Supabase columns (e.g., 'lat', 'latitude')
                lat = user_data.get('latitude') or user_data.get('lat')
                lng = user_data.get('longitude') or user_data.get('long') or user_data.get('lng')
                
                if lat is not None and lng is not None:
                    return float(lat), float(lng)
            return None
        except Exception as e:
            print(f"Supabase Error: {e}")
            return None

    async def find_nearby_facilities(
        self,
        ctx: Context,
        user_id: Annotated[str, Field(description="The ID of the user in Supabase. Do not call without user id")],
        facility_type: Annotated[str, Field(description="Type of facility (hospital, pharmacy, doctor)")] = "hospital",
        keyword: Annotated[str, Field(description="Specific keyword (e.g., 'cardiologist', 'emergency')")] = None,
    ) -> Dict:
        """
        Finds the nearest medical facilities based on the user's location stored in Supabase.
        """
        if ctx:
            await ctx.info(f"Fetching location for User ID: {user_id}")

        # 1. Get Location from Supabase
        coords = self._get_coordinates_from_supabase(user_id)
        
        if not coords:

            error_msg = f"Could not find coordinates for user {user_id} in table 'patient_locations'."
            if ctx: await ctx.warning(error_msg)
            return {"error": error_msg, "status": "location_not_found"}

        lat, lng = coords
        if ctx:
            await ctx.info(f"Location found: {lat}, {lng}. Searching for {facility_type}...")

        # 2. Query Google Maps
        try:
            # Note: When using rank_by='distance', 'radius' must not be included.
            places_result = self.gmaps.places_nearby(
                location=(lat, lng),
                rank_by='distance',
                type=facility_type,
                keyword=keyword
            )

            results = []
            if 'results' in places_result:
                # Limit to top 5 nearest results
                for place in places_result['results'][:5]:
                    place_lat = place['geometry']['location']['lat']
                    place_lng = place['geometry']['location']['lng']
                    
                    results.append({
                        "name": place.get('name'),
                        "address": place.get('vicinity'),
                        "rating": place.get('rating', 'N/A'),
                        "status": place.get('business_status'),
                        "map_link": f"https://www.google.com/maps/search/?api=1&query={place_lat},{place_lng}"
                    })

            if ctx:
                await ctx.info(f"Found {len(results)} facilities nearby.")

            return {
                "user_id": user_id,
                "search_location": {"lat": lat, "lng": lng},
                "facility_type": facility_type,
                "count": len(results),
                "facilities": results
            }

        except Exception as e:
            error_msg = f"Google Maps API Error: {str(e)}"
            if ctx: await ctx.error(error_msg)
            return {"error": error_msg}
