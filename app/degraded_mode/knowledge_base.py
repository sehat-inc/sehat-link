"""
JSON Knowledge Base Loader and Search

Loads static JSON files and provides search functionality.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class KnowledgeBase:
    """Manages loading and searching JSON knowledge files"""
    
    def __init__(self, knowledge_dir: str = None):
        if knowledge_dir is None:
            # Get path relative to this file
            knowledge_dir = Path(__file__).parent / "knowledge"
        self.knowledge_dir = Path(knowledge_dir)
        self.data: Dict[str, Any] = {}
        self.load_all()
    
    def load_all(self):
        """Load all JSON files from knowledge directory"""
        if not self.knowledge_dir.exists():
            raise FileNotFoundError(f"Knowledge directory not found: {self.knowledge_dir}")
        
        json_files = {
            "medical": "medical_assistance.json",
            "programs": "government_programs.json",
            "hospitals": "hospitals_by_city.json",
            "emergency": "emergency_info.json"
        }
        
        for key, filename in json_files.items():
            filepath = self.knowledge_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.data[key] = json.load(f)
            else:
                self.data[key] = {}
    
    def search_medical(self, query: str) -> List[Dict]:
        """Search medical assistance data"""
        results = []
        medical_data = self.data.get("medical", {})
        
        query_lower = query.lower()
        
        # Search symptoms
        for symptom in medical_data.get("symptoms", []):
            if self._matches(query_lower, symptom):
                results.append({
                    "type": "symptom",
                    "data": symptom
                })
        
        # Search conditions
        for condition in medical_data.get("conditions", []):
            if self._matches(query_lower, condition):
                results.append({
                    "type": "condition",
                    "data": condition
                })
        
        # Search first aid
        for aid in medical_data.get("first_aid", []):
            if self._matches(query_lower, aid):
                results.append({
                    "type": "first_aid",
                    "data": aid
                })
        
        return results
    
    def search_programs(self, query: str) -> List[Dict]:
        """Search government programs"""
        results = []
        programs_data = self.data.get("programs", {})
        
        query_lower = query.lower()
        
        for program in programs_data.get("programs", []):
            if self._matches(query_lower, program):
                results.append({
                    "type": "program",
                    "data": program
                })
        
        return results
    
    def search_hospitals(self, city: Optional[str] = None) -> List[Dict]:
        """Search hospitals by city"""
        hospitals_data = self.data.get("hospitals", {})
        
        if city:
            city_lower = city.lower()
            return hospitals_data.get("cities", {}).get(city_lower, [])
        
        # Return all hospitals if no city specified
        all_hospitals = []
        for city_hospitals in hospitals_data.get("cities", {}).values():
            all_hospitals.extend(city_hospitals)
        return all_hospitals
    
    def get_emergency_info(self) -> Dict:
        """Get emergency contact information"""
        return self.data.get("emergency", {})
    
    def search_all(self, query: str) -> Dict[str, List]:
        """Search across all knowledge bases"""
        return {
            "medical": self.search_medical(query),
            "programs": self.search_programs(query),
            "emergency": self.get_emergency_info()
        }
    
    def _matches(self, query: str, item: Dict) -> bool:
        """Check if query matches item using keywords or text fields"""
        # Check keywords
        keywords = item.get("keywords", [])
        if any(keyword.lower() in query for keyword in keywords):
            return True
        
        # Check name/title
        name = item.get("name", "").lower()
        if query in name or name in query:
            return True
        
        # Check description
        description = item.get("description", "").lower()
        if query in description:
            return True
        
        return False
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all loaded knowledge data"""
        return self.data


# Global instance
knowledge_base = KnowledgeBase()
