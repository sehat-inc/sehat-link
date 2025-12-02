def decompose_prompt(question: str) -> str:

    return f"""
    Analyze this question and break it down into 2 simple, focused sub-queries that would help answer it comprehensively.

    Question: {question}

    For each sub-query, provide:
    1. A clear, focused search query (5-10 words)
    2. Why this sub-query is needed

    Return ONLY a JSON array of objects with 'query' and 'purpose' fields. Example:
    [
      {{"query": "features of product X", "purpose": "understand X capabilities"}},
      {{"query": "features of product Y", "purpose": "understand Y capabilities"}}
    ]

    Return ONLY the JSON array, no other text.
    """
