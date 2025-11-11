import os
import json
from typing import Annotated, List, Dict
from pydantic import Field
from openai import OpenAI
from pinecone import Pinecone
from app.api.mcp.prompts.decompose import decompose_prompt
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
        top_k: int = 5,
        namespace: str = "eligibility-namespace"
    ) -> List[Dict]:
        """
        Query the Pinecone index with a vector and return the results.
        """
        results = self.index.query(
            namespace=namespace,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
                "text": match.metadata.get("text", "")
            }
            for match in results.get("matches", [])
        ]

    async def smart_query(
        self,
        question: Annotated[str, Field(description="The question or query to answer")],
        top_k_per_query: Annotated[int, Field(description="Amount of chunks to retrieve per query", ge=1, le=10)] = 5,
        namespace: Annotated[str, Field(description="Pinecone namespace to query")] = "eligibility-namespace",
        decompose: Annotated[bool, Field(description="Whether to decompose the question into multiple queries")] = True,
        ctx: Context = None
    ) -> Dict:
        """
        Decomposes a complex question into sub-queries or executes a direct query.
        """
        if ctx:
            await ctx.info(f"Processing question: {question}")

        if not decompose:
            if ctx:
                await ctx.info("Direct Query Mode (no decomposition)")
            query_vector = self._embed_text(question)
            results = self._query_pinecone(query_vector, top_k_per_query, namespace)
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
            decomposition_response = await ctx.sample(decomposition_prompt)
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
            results = self._query_pinecone(query_vector, top_k_per_query, namespace)
            return {
                "strategy": "direct_fallback",
                "original_question": question,
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
            results = self._query_pinecone(query_vector, top_k_per_query, namespace)
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
        query: Annotated[str, Field(description="Search query")],
        top_k: Annotated[int, Field(description="Number of results", ge=1, le=15)] = 5,
        namespace: Annotated[str, Field(description="Pinecone namespace")] = "eligibility-namespace",
        ctx: Context = None
    ) -> List[Dict]:
        """
        Direct, simple query to Pinecone without any decomposition.
        """
        if ctx:
            await ctx.info(f"Direct query: {query}")
        
        query_vector = self._embed_text(query)
        results = self._query_pinecone(query_vector, top_k, namespace)
        
        return results