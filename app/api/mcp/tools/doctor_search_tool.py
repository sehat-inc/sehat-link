import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import requests

class DoctorSearchInput(BaseModel):
    location: str = Field(..., description="City or area in Pakistan (e.g., 'Lahore', 'Karachi')")
    specialty: str = Field(..., description="Medical specialty or symptom-based query (e.g., 'cardiologist', 'fever')")
    max_results: int = Field(5, ge=1, le=10, description="Number of doctor profiles to return")

class DoctorSearchTool(BaseTool):
    name: str = "search_pakistani_doctors"
    description: str = (
        "Searches for doctors in Pakistan using trusted online booking platforms: "
        "Marham.pk, Healthwire.pk, MeriSehat.pk, and SehatKahani.com. "
        "Returns structured results: title, snippet, link, and source domain."
    )
    args_schema: type[BaseModel] = DoctorSearchInput

    # Declare as Pydantic fields — they’ll be populated from env or kwargs
    api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_CSE_API_KEY"))
    search_engine_id: str = Field(default_factory=lambda: os.getenv("GOOGLE_CSE_ENGINE_ID"))

    def _run(self, location: str, specialty: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.api_key or not self.search_engine_id:
            raise ValueError("GOOGLE_CSE_API_KEY and GOOGLE_CSE_ENGINE_ID must be set in environment.")

        query = (
            f'doctor "{specialty}" in "{location}" '
            f'site:marham.pk OR site:healthwire.pk OR site:merisehat.pk OR site:sehatkahani.com'
        )

        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(max_results, 10),
            "lr": "lang_en",
            "gl": "pk",
        }

        try:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", "").strip(),
                    "snippet": item.get("snippet", "").strip(),
                    "link": item.get("link", ""),
                    "source": next(
                        (domain for domain in ["marham.pk", "healthwire.pk", "merisehat.pk", "sehatkahani.com"]
                         if domain in item.get("link", "")),
                        "unknown"
                    )
                })
            return results or [{"message": "No doctors found matching your criteria."}]

        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]

# ✅ Now safe to instantiate at module level
doctor_search_tool = DoctorSearchTool()