"""
Simple Agent for Degraded Mode

Lightweight agent that:
1. Searches JSON knowledge base
2. Uses LLM to generate natural responses
3. No complex LangGraph workflow
"""

from typing import List, Dict, Optional
from .knowledge_base import knowledge_base
from .llm_provider import get_llm_provider
from core.logging import get_logger


logger = get_logger("DEGRADED_AGENT")


class DegradedAgent:
    """Simple agent for offline/degraded mode"""
    
    def __init__(self):
        self.kb = knowledge_base
        self.llm = get_llm_provider()
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with knowledge base context"""
        return """You are a helpful medical assistant operating in degraded/offline mode.

Your capabilities:
- Provide general medical information and first aid guidance
- Share information about government health programs in Pakistan
- Recommend hospitals in major cities
- Offer emergency contact information

IMPORTANT LIMITATIONS:
- You cannot access patient medical records
- You cannot provide personalized medical advice
- You cannot prescribe medications
- Always recommend consulting a healthcare professional for serious concerns

When answering:
1. Search the knowledge base for relevant information
2. Provide clear, concise responses in 2-3 lines maximum.
3. Include disclaimers when appropriate
4. Suggest emergency services for urgent situations
5. Always output a string. No markdown no extra characters.

Be helpful, empathetic, and safety-conscious."""
    
    async def process_message(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]]
    ) -> Dict[str, any]:
        """
        Process user message and generate response.
        
        Returns:
        {
            "response": str,
            "sources": List[Dict],
            "search_results": Dict
        }
        """
        try:
            # Step 1: Search knowledge base
            search_results = self._search_knowledge(user_message)
            logger.info(f"Search results: {search_results}")
            
            # Step 2: Build context from search results
            context = self._build_context(search_results)
            logger.info(f"Context built: {context[:200]}...")
            
            # Step 3: Generate LLM response
            messages = self._prepare_messages(
                user_message,
                context,
                chat_history
            )
            
            response = await self.llm.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            # Step 4: Extract sources
            sources = self._extract_sources(search_results)
            
            return {
                "response": response,
                "sources": sources,
                "search_results": search_results
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": self._get_fallback_response(),
                "sources": [],
                "search_results": {},
                "error": str(e)
            }
    
    def _search_knowledge(self, query: str) -> Dict:
        """Search all knowledge bases"""
        # Detect query intent
        query_lower = query.lower()
        
        results = {}
        
        # Medical queries
        if any(word in query_lower for word in [
            "symptom", "pain", "fever", "cough", "headache", 
            "treatment", "medicine", "diagnosis", "sick", "hurt"
        ]):
            results["medical"] = self.kb.search_medical(query)
        
        # Program queries
        if any(word in query_lower for word in [
            "program", "sehat", "sahulat", "bait", "maal",
            "eligibility", "government", "scheme", "card"
        ]):
            results["programs"] = self.kb.search_programs(query)
        
        # Hospital queries
        if any(word in query_lower for word in [
            "hospital", "clinic", "doctor", "facility", "medical center"
        ]):
            # Try to extract city name
            city = self._extract_city(query)
            results["hospitals"] = self.kb.search_hospitals(city)
        
        # Emergency queries
        if any(word in query_lower for word in [
            "emergency", "urgent", "ambulance", "rescue", "helpline"
        ]):
            results["emergency"] = self.kb.get_emergency_info()
        
        # If no specific category, search all
        if not results:
            results = self.kb.search_all(query)
        
        return results
    
    def _extract_city(self, query: str) -> Optional[str]:
        """Extract city name from query"""
        cities = [
            "karachi", "lahore", "islamabad", "rawalpindi",
            "faisalabad", "multan", "peshawar", "quetta",
            "sialkot", "gujranwala", "hyderabad"
        ]
        
        query_lower = query.lower()
        for city in cities:
            if city in query_lower:
                return city
        
        return None
    
    def _build_context(self, search_results: Dict) -> str:
        """Build context string from search results"""
        context_parts = []
        
        # Medical context
        if "medical" in search_results and search_results["medical"]:
            context_parts.append("MEDICAL INFORMATION:")
            for item in search_results["medical"][:3]:  # Limit to top 3
                data = item["data"]
                context_parts.append(f"- {data.get('name', 'Unknown')}: {data.get('description', '')}")
        
        # Programs context
        if "programs" in search_results and search_results["programs"]:
            context_parts.append("\nGOVERNMENT PROGRAMS:")
            for item in search_results["programs"][:3]:
                data = item["data"]
                context_parts.append(f"- {data.get('name', 'Unknown')}: {data.get('description', '')}")
        
        # Hospitals context
        if "hospitals" in search_results and search_results["hospitals"]:
            context_parts.append("\nHOSPITALS:")
            for hospital in search_results["hospitals"][:5]:
                context_parts.append(
                    f"- {hospital.get('name', 'Unknown')} ({hospital.get('city', '')}): "
                    f"{hospital.get('address', '')}, Contact: {hospital.get('contact', 'N/A')}"
                )
        
        # Emergency context
        if "emergency" in search_results and search_results["emergency"]:
            emergency = search_results["emergency"]
            context_parts.append("\nEMERGENCY CONTACTS:")
            for service, number in emergency.get("hotlines", {}).items():
                context_parts.append(f"- {service}: {number}")
        
        return "\n".join(context_parts) if context_parts else "No relevant information found in knowledge base."
    
    def _prepare_messages(
        self,
        user_message: str,
        context: str,
        chat_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepare messages for LLM"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add recent chat history (last 4 messages)
        recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
        messages.extend(recent_history)
        
        # Add context and current message
        user_content = f"""KNOWLEDGE BASE CONTEXT:
{context}

USER QUESTION:
{user_message}

Please provide a helpful response based on the available information. If the knowledge base doesn't contain relevant information, provide general guidance and recommend consulting a healthcare professional."""
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def _extract_sources(self, search_results: Dict) -> List[Dict]:
        """Extract source references from search results"""
        sources = []
        
        for category, items in search_results.items():
            if isinstance(items, list):
                for item in items[:3]:  # Top 3 per category
                    if isinstance(item, dict):
                        if "data" in item:
                            data = item["data"]
                            sources.append({
                                "category": category,
                                "name": data.get("name", "Unknown"),
                                "type": item.get("type", category)
                            })
                        elif "name" in item:
                            sources.append({
                                "category": category,
                                "name": item.get("name", "Unknown"),
                                "type": category
                            })
            elif isinstance(items, dict) and category == "emergency":
                sources.append({
                    "category": "emergency",
                    "name": "Emergency Services",
                    "type": "emergency_info"
                })
        
        return sources
    
    def _get_fallback_response(self) -> str:
        """Fallback response when agent fails"""
        return """I apologize, but I'm experiencing technical difficulties. 

For immediate medical concerns:
- Call emergency services: 1122 (Pakistan)
- Visit the nearest hospital emergency department

For general health information, please try again or consult a healthcare professional."""


# Global instance
degraded_agent = DegradedAgent()
