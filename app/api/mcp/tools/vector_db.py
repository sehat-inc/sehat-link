from fastmcp import Context
from pinecone import Pinecone, SearchQuery
from typing import Annotated, Optional, Dict
from pydantic import Field
from mcp.types import ImageContent, AudioContent, TextContent
import json

from app.api.mcp.prompts.decompose import decompose_prompt


class PineconeQuery():
    def __init__(self, api_key: str, index_name: str):
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)

    async def _query_pinecone(self, query_text: str, top_k: int = 5, namespace: str = "__default__", filter: Optional[Dict] = None) -> list[dict]:
        """
        Query Pinecone and return results

        Args:
            query_text: Text to query
            top_k: Number of results to return
            namespace: Optional namespaces to query
        
        Return:
            List of matches with metadata
        """

        async with self.pc.IndexAsyncio(self.index_name) as index:
            results = await index.search_records(
                namespace=namespace,
                query=SearchQuery(
                    inputs={"text": query_text},
                    top_k=top_k,
                    filter=filter,
                )
            )

        hits = results["result"]["hits"]

        return [
            {
                "id": h["_id"],
                "score": h["_score"],
                "metadata": h.get("fields", {}),
                "text": h.get("fields", {}).get("chunk_text", "")
            }
            for h in hits
        ]

    async def smart_query(
        self,
        ctx: Context,
        question: Annotated[str, Field(description="The question or query to answer")],
        top_k_per_query: Annotated[int, Field(description="Amount of chunks to retrieve per query", ge=1, le=10)] = 5,
        namespace: Annotated[str , Field(description="Pinecone namespace to query")] = "__default__",
        decompose: Annotated[bool, Field(description="Whether to decompose the question into multiple queries")] = True,
    ) -> dict:

        if ctx:
            await ctx.info(f"Processing question: {question}")

        if not decompose:
            if ctx:
                await ctx.info("Direct Query Mode (no decomposition)")

            results = await self._query_pinecone(query_text=question,
                                      top_k=top_k_per_query,
                                      namespace=namespace)

            return {
                "strategy": "direct",
                "original_question": question,
                "results": results,
                "count": len(results)
            }

        if ctx:
            await ctx.info(f"Decomposing questions into sub-queries")

        try: 
            decomposition_prompt = decompose_prompt(question)
            decomposition_response = await ctx.sample(decomposition_prompt, model_preferences="gemini-2.5-flash")
            if isinstance(decomposition_response, (ImageContent, AudioContent)):
                await ctx.error("Decomposition model returned non-text content.")
                return {"error": "Decomposition model returned non-text content."}
            decomposition_text = decomposition_response.text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in decomposition_text:
                decomposition_text = decomposition_text.split("```json")[1].split("```")[0].strip()
            elif "```" in decomposition_text:
                decomposition_text = decomposition_text.split("```")[1].split("```")[0].strip()
                
            sub_queries = json.loads(decomposition_text)

            if ctx:
                await ctx.info(f"Decomposed into {len(sub_queries)} queries")

        except Exception as e:
            if ctx:
                await ctx.warning(f"Decomposition Failed: {e} - using direct query method")

            results = await self._query_pinecone(query_text=question,
                                      top_k=top_k_per_query,
                                      namespace=namespace)
            return {
                "strategy": "direct",
                "original_question": question,
                "error": str(e),
                "results": results,
                "count": len(results)
            }

        
        all_results = {}
        total_queries = len(sub_queries)

        for i, sub_query_info in enumerate(sub_queries):
            sub_query = sub_query_info.get("query", "")
            purpose = sub_query_info.get("purpose", "")

            if ctx:
                await ctx.report_progress(progress=i, total=total_queries)
                await ctx.debug(f"Executing Query: {sub_query}")

            results = await self._query_pinecone(query_text=sub_query,
                                      top_k=top_k_per_query,
                                      namespace=namespace)

            all_results[sub_query] = {
                "purpose": purpose,
                "results": results,
                "result_count": len(results)
            }

        if ctx:
            await ctx.report_progress(progress=total_queries, total=total_queries)
            await ctx.info(f"All sub-queries completed")

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
        top_k: Annotated[int, Field(description="Number of results", ge=1, le=15)] = 5,
        namespace: Annotated[str, Field(description="Pinecone namespace")] = "__default__",
        filter_metadata: Annotated[dict | None, Field(description="Metadata filter")] = None,
    ) -> list[dict]:
        """
        Direct, simple query to Pinecone without any decomposition.
        Use this for straightforward searches.
        """
        if ctx:
            await ctx.info(f"Direct query: {query}")
        
        results = await self._query_pinecone(query, top_k, namespace, filter_metadata)

        return [
            {
                "id": r["id"],
                "score": r["score"],
                "metadata": r["metadata"],
                "text": r["text"]
            }
            for r in results
        ]
