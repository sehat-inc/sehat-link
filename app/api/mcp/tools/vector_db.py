import json
from typing import Annotated, List, Dict, Optional, Any
from pydantic import Field
from openai import OpenAI
from pinecone import Pinecone
from ..prompts.decompose import decompose_prompt
from ..tools.helper import SPECIALTIES, CITIES
from fastmcp import Context


class PineconeQuery:
    """
    A class to interact with a Pinecone index, using OpenAI for embeddings.
    """
    def __init__(self, pinecone_api_key: str, openai_api_key: str, index_name: str):
        self.index_name = index_name
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.index = self.pc.Index(self.index_name)
        self.embed_model = "text-embedding-3-large"

    def _embed_text(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using OpenAI.
        """
        response = self.openai_client.embeddings.create(
            input=[text],
            model=self.embed_model,
            dimensions=2048
        )
        return response.data[0].embedding

    def _query_pinecone(
        self,
        query_vector: List[float],
        top_k: int = 2,
        namespace: str = "eligibility-namespace",
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query the Pinecone index with a vector and return the results.
        """

        query_params = {
            "vector": query_vector,
            "top_k": top_k,
            "namespace": namespace,
            "include_metadata": True
        }

        if filter:
            query_params["filter"] = filter

        results = self.index.query(**query_params)

        
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
                "text": match.metadata.get("text", "")
            }
            for match in results.get("matches", [])
        ]
    
    def _build_pinecone_filter(self, metadata_filters: Dict[str, Any]) -> Dict:
        """
        Build Pinecone filter with OR logic for list values.
        
        Args:
            metadata_filters: Dict where list values are treated as OR conditions
            
        Returns:
            Pinecone-compatible filter dict
            
        Examples:
            Input: {"category": ["symptom", "treatment"], "severity": "high"}
            Output: {"$and": [
                {"category": {"$in": ["symptom", "treatment"]}},
                {"severity": {"$eq": "high"}}
            ]}
        """
        if not metadata_filters:
            return {}
        
        filter_conditions = []
        
        for key, value in metadata_filters.items():
            if isinstance(value, list):
                # OR logic for list values using $in operator
                if len(value) == 1:
                    filter_conditions.append({key: {"$eq": value[0]}})
                else:
                    filter_conditions.append({key: {"$in": value}})
            else:
                # Exact match for single values
                filter_conditions.append({key: {"$eq": value}})
        
        # Combine all conditions with AND
        if len(filter_conditions) == 0:
            return {}
        elif len(filter_conditions) == 1:
            return filter_conditions[0]
        else:
            return {"$and": filter_conditions}


    async def smart_query(
        self,
        ctx: Context,
        question: Annotated[str, Field(description="The question or query to answer")],
        top_k_per_query: Annotated[int, Field(description="Amount of chunks to retrieve per query", ge=1, le=10)] = 2,
        namespace: Annotated[str,
            Field(
                description="""
                Pinecone namespace to query. Choose based on topic.
                - '__default__' for symptom/medical related queries
                - 'eligibility-namespace' for program/eligibility related queries
                - 'dcotor-namespace' for doctor related queries
                """
            )
        ] = "__default__",
        decompose: Annotated[bool, Field(description="Whether to decompose the question into multiple queries")] = True,
        metadata_filters: Annotated[Optional[Dict[str, Any]], 
            Field(description="""
                Metadata filters to apply. Supports OR logic for sub-filters.
                Example: {"category": ["symptom", "treatment"], "severity": "high"}
                List values are treated as OR conditions, single values as exact match.
                """
            )
        ] = None
    ) -> Dict:
        """
        Decomposes a complex question into sub-queries or executes a direct query.
        """
        if ctx:
            await ctx.info(f"Processing question: {question}")
            if metadata_filters:
                await ctx.info(f"Applying metadata filters: {metadata_filters}")
        
        pinecone_filter = self._build_pinecone_filter(metadata_filters) if metadata_filters else None

        if not decompose:
            if ctx:
                await ctx.info("Direct Query Mode (no decomposition)")
            query_vector = self._embed_text(question)
            results = self._query_pinecone(query_vector, top_k_per_query, namespace, filter=pinecone_filter)
            return {
                "strategy": "direct",
                "original_question": question,
                "results": results,
                "count": len(results)
            }

        if ctx:
            await ctx.info("Decomposing question into sub-queries")

        try:
            decomposition_prompt = decompose_prompt(question)
            decomposition_response = await ctx.sample(
                messages=decomposition_prompt,
                model_preferences=["gemini-2.5-flash", "gpt4o-mini"],
                temperature=0.5
            )
            decomposition_text = decomposition_response.text.strip()
            
            if "```json" in decomposition_text:
                json_str = decomposition_text.split("```json")[1].split("```").strip()
            elif "```" in decomposition_text:
                json_str = decomposition_text.split("```").split("```")[0].strip()
            else:
                json_str = decomposition_text

            sub_queries_info = json.loads(json_str)
            if ctx:
                await ctx.info(f"Decomposed into {len(sub_queries_info)} queries")
        except Exception as e:
            if ctx:
                await ctx.warning(f"Decomposition Failed: {e} - using direct query method")
            query_vector = self._embed_text(question)
            results = self._query_pinecone(query_vector, top_k_per_query, namespace, filter=pinecone_filter)
            return {
                "strategy": "direct_fallback",
                "original_question": question,
                "metadata_filters": metadata_filters,
                "error": str(e),
                "results": results,
                "count": len(results)
            }

        all_results = {}
        total_queries = len(sub_queries_info)
        for i, info in enumerate(sub_queries_info):
            sub_query = info.get("query", "")
            purpose = info.get("purpose", "")
            if ctx:
                await ctx.report_progress(progress=i, total=total_queries)
                await ctx.debug(f"Executing Query: {sub_query}")
            
            query_vector = self._embed_text(sub_query)
            results = self._query_pinecone(query_vector, top_k_per_query, namespace, filter=pinecone_filter)
            all_results[sub_query] = {
                "purpose": purpose,
                "results": results,
                "result_count": len(results)
            }
        
        if ctx:
            await ctx.report_progress(progress=total_queries, total=total_queries)
            await ctx.info("All sub-queries completed")

        return {
            "strategy": "decomposed",
            "original_question": question,
            "sub_queries": list(all_results.keys()),
            "results": all_results,
            "total_sub_queries": len(all_results),
            "synthesis_needed": True
        }

    async def direct_query(
        self,
        ctx: Context,
        query: Annotated[str, Field(description="Search query")],
        top_k: Annotated[int, Field(description="Number of results", ge=1, le=15)] = 2,
        namespace: Annotated[str, Field(description="Pinecone namespace")] = "__default__",
    ) -> List[Dict]:
        """
        Direct, simple query to Pinecone without any decomposition.
        """
        if ctx:
            await ctx.info(f"Direct query: {query}")
        
        query_vector = self._embed_text(query)
        results = self._query_pinecone(query_vector, top_k, namespace)
        
        return results

    async def smart_query_with_filters(
        self,
        ctx: Context,
        question: Annotated[str, Field(description="The question or query to answer about doctors")],
        top_k_per_query: Annotated[int, Field(description="Amount of chunks to retrieve per query", ge=1, le=10)] = 5,
        namespace: Annotated[str,
            Field(
                description="""
                Pinecone namespace to query. Choose based on topic.
                - '__default__' for symptom/medical related queries
                - 'eligibility-namespace' for program/eligibility related queries
                - 'dcotor-namespace' for doctor related queries
                """
            )
        ] = "dcotor-namespace",
        decompose: Annotated[bool, Field(description="Whether to decompose the question into multiple queries")] = True,
        specialties: Annotated[
            Optional[List[str]], 
            Field(
                description=f"""Filter by doctor specialties (OR logic - matches ANY of the provided specialties).
                
                AVAILABLE SPECIALTIES (choose from these ONLY):
                {', '.join(sorted(set(SPECIALTIES)))}
                
                Examples: 
                - ["Cardiologist"]
                - ["General Physician", "Dermatologist"]
                - ["Cardiac Surgeon", "Neuro Surgeon"]
                
                Leave empty or null to search all specialties."""
            )
        ] = None,
        cities: Annotated[
            Optional[List[str]], 
            Field(
                description=f"""Filter by cities (OR logic - matches ANY of the provided cities).
                
                AVAILABLE CITIES (choose from these ONLY):
                {', '.join(sorted(CITIES))}
                
                Examples: 
                - ["Lahore"]
                - ["Karachi", "Islamabad"]
                - ["Lahore", "Karachi", "Rawalpindi"]
                
                Leave empty or null to search all cities."""
            )
        ] = None,
    ) -> Dict[str, Any]:
        """
        Smart query for doctors with specialty and city filtering.
        
        This function intelligently queries the doctor database by:
        1. Breaking down complex questions into sub-queries (if decompose=True)
        2. Applying specialty and city filters with OR logic
        3. Aggregating and returning relevant results
        
        Filter Logic:
        - specialties: Matches ANY of the provided specialties (OR within specialties)
        - cities: Matches ANY of the provided cities (OR within cities)
        - Combined: Must match (specialty1 OR specialty2 OR...) AND (city1 OR city2 OR...)
        
        Returns:
        - Dict containing query results, strategy used, and metadata
        """
        # Build metadata filters
        metadata_filters = {}
        
        if ctx:
            await ctx.info(f"Running Doctor Tool")

        if specialties and len(specialties) > 0:
            metadata_filters["specialty"] = specialties
            if ctx:
                await ctx.info(f"Filtering by specialties: {', '.join(specialties)}")
        
        
        if cities and len(cities) > 0:
            metadata_filters["city"] = cities
            if ctx:
                await ctx.info(f"Filtering by cities: {', '.join(cities)}")
        
        # Call the underlying smart_query method with metadata filters
        return await self.smart_query(
            ctx=ctx,
            question=question,
            top_k_per_query=top_k_per_query,
            namespace=namespace,
            decompose=decompose,
            metadata_filters=metadata_filters if metadata_filters else None
        )

