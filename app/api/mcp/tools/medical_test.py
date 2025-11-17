import os
import json
import asyncio
from app.api.mcp.tools.vector_db import PineconeQuery 

# ---------- Mock Dependencies ----------
# This section creates mock objects to simulate the behavior of 
# fastmcp.Context and your decompose prompt for local testing.

class MockContext:
    """A mock context to simulate fastmcp.Context for local testing."""
    async def info(self, msg: str):
        print(f"[INFO] {msg}")

    async def debug(self, msg: str):
        print(f"[DEBUG] {msg}")

    async def warning(self, msg: str):
        print(f"[WARNING] {msg}")
    
    async def report_progress(self, progress, total):
        print(f"[PROGRESS] {progress}/{total}")

    async def sample(self, prompt: str):
        """Simulates the LLM call for decomposition."""
        print(f"[SAMPLING] Simulating LLM response for decomposition...")
        # A mock response tailored for a complex medical query
        mock_response_text = """
        ```json
        [
            {
                "query": "What are the common symptoms of a stroke?",
                "purpose": "To identify the primary signs and symptoms of a stroke for initial diagnosis."
            },
            {
                "query": "What diagnostic tests are used to confirm a stroke?",
                "purpose": "To find information about the medical procedures and imaging used to diagnose a stroke, such as CT scans or MRIs."
            }
        ]
        ```
        """
        class MockResponse:
            def __init__(self, text):
                self.text = text
        
        return MockResponse(mock_response_text)

def decompose_prompt(question: str) -> str:
    """Mock decompose prompt function."""
    return f"Decompose this question: {question}"

# Monkey-patch the original module to use our mocks during the test
try:
    import app.api.mcp.prompts.decompose as decompose_module
    decompose_module.decompose_prompt = decompose_prompt
except (ImportError, ModuleNotFoundError):
    print("[NOTE] Mocking setup for 'app' module skipped as it's not in the path. Continuing...")


# ---------- Main Test Logic ----------

async def main():
    """Main function to run the tests for the medical knowledge base."""
    pinecone_api_key = os.getenv("PINECONE_API")
    # Fetch the OPENAI_API_KEY as it is required by the PineconeQuery constructor
    openai_api_key = os.getenv("OPENAI_API_KEY") 
    
    # --- Configuration for the Medical Index ---
    index_name = "analytics-agent-kb"
    namespace = "__default__" 

    # Check that BOTH keys are present
    if not all([pinecone_api_key, openai_api_key]):
        print("Error: Make sure the PINECONE_API_KEY and OPENAI_API_KEY environment variables are set.")
        return

    # --- FIX APPLIED HERE: Using the correct required keyword arguments ---
    # The PineconeQuery.__init__ method requires pinecone_api_key and openai_api_key.
    pinecone_client = PineconeQuery(
        pinecone_api_key=pinecone_api_key, 
        openai_api_key=openai_api_key,
        index_name=index_name
    )
    
    # Initialize the mock context
    ctx = MockContext()

    # Define queries relevant to the medical dataset
    simple_query = "What is a heart attack?"
    complex_query = "What are the symptoms of a stroke and how is it diagnosed?"

    print("=============================================")
    print("  Running Pinecone Tests on Medical KB")
    print("=============================================\n")

    # --- Test 1: Direct Query ---
    print("--- Test 1: Running direct_query() ---")
    try:
        direct_results = await pinecone_client.direct_query(
            query=simple_query,
            top_k=3,
            namespace=namespace,
            ctx=ctx
        )
        print("✅ Direct Query Successful. Results:")
        print(json.dumps(direct_results, indent=2))
    except Exception as e:
        print(f"❌ Direct Query Failed: {e}")

    print("\n" + "="*45 + "\n")

    # --- Test 2: Smart Query with Decomposition ---
    print("--- Test 2: Running smart_query() with decompose=True ---")
    try:
        smart_results_decomposed = await pinecone_client.smart_query(
            question=complex_query,
            top_k_per_query=2,
            namespace=namespace,
            decompose=True,
            ctx=ctx
        )
        print("✅ Smart Query (Decomposed) Successful. Results:")
        print(json.dumps(smart_results_decomposed, indent=2))
    except Exception as e:
        print(f"❌ Smart Query (Decomposed) Failed: {e}")
        
    print("\n" + "="*45 + "\n")
    
    # --- Test 3: Smart Query without Decomposition ---
    print("--- Test 3: Running smart_query() with decompose=False ---")
    try:
        smart_results_direct = await pinecone_client.smart_query(
            question=simple_query,
            top_k_per_query=3,
            namespace=namespace,
            decompose=False,
            ctx=ctx
        )
        print("✅ Smart Query (Direct) Successful. Results:")
        print(json.dumps(smart_results_direct, indent=2))
    except Exception as e:
        print(f"❌ Smart Query (Direct) Failed: {e}")


if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock
    
    # This setup helps the script find the 'app' module if it's not in the Python path
    if 'app' not in sys.modules:
        # Create a fake 'app' structure for the imports to resolve correctly
        mock_app = MagicMock()
        sys.modules['app'] = mock_app
        sys.modules['app.api'] = MagicMock()
        sys.modules['app.api.mcp'] = MagicMock()
        sys.modules['app.api.mcp.tools'] = MagicMock()
        # This assumes your PineconeQuery class is in the vector_db.py file
        # If not, you must create a mock for it here as well.
        sys.modules['app.api.mcp.prompts'] = MagicMock()
        sys.modules['app.api.mcp.prompts.decompose'] = MagicMock()

    asyncio.run(main())