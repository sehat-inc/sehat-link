from typing import Dict, List, Optional
from datetime import datetime
from pinecone import Pinecone, SearchQuery
import json


class PineconeMemory:
    """
    Pinecone for semantic search of past cases and conversations.
    Stores embeddings + metadata.
    """
    
    def __init__(self, api_key: str, index_name: str):
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)
    
    async def store_conversation(
        self,
        conversation_id: str,
        user_id: str,
        summary: str,
        metadata: Dict
    ):
        """
        Store conversation summary with embedding + metadata.
        
        Metadata includes:
        - user_id, session_id, date
        - symptoms, diagnosis, urgency
        - medications, outcome
        """
        
       
        async with self.pc.IndexAsyncio(host=self.index_name) as idx:

            await idx.upsert_records(
                namespace=f"user_{user_id}",  # User-specific namespace
                records=[
                    {
                        "_id":conversation_id,
                        "chunk_text":summary,
                        "user_id": user_id,
                        "date": metadata.get("date", datetime.now().isoformat()),
                        "urgency": metadata.get("urgency", "Medium"),
                        "symptoms": json.dumps(metadata.get("symptoms", [])),  # JSON string
                        "diagnosis": metadata.get("diagnosis", ""),
                        "outcome": metadata.get("outcome", ""),
                        "type": "conversation"
                    }
                ]
            )
    
    async def search_similar_cases(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar past cases using semantic search.
        
        Example filter: {"urgency": "High", "type": "conversation"}
        """
        
        # Search in user's namespace
        async with self.pc.IndexAsyncio(host=self.index_name) as idx:
            results = await idx.search_records(
                namespace=f"user_{user_id}",
                query=SearchQuery(
                    inputs={"text": query},
                    top_k=top_k,
                    filter=filter_dict
                )
            )

        parsed = []
        for r in results:     # results is your list of dicts
            case = {
                "id": r["id"],
                "score": r["score"],
                "summary": r["metadata"].get("summary", ""),
                "date": r["metadata"].get("date", ""),
                "urgency": r["metadata"].get("urgency", ""),
                "symptoms": json.loads(r["metadata"].get("symptoms", "[]")),
                "diagnosis": r["metadata"].get("diagnosis", ""),
                "outcome": r["metadata"].get("outcome", ""),
                "text": r.get("text", "")     # since this result object has text (pinecone matches sometimes don’t)
            }
            parsed.append(case)
        
        return parsed
