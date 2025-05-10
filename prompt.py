def get_prompt(text: str, context_to_keep: str, doc_type: str = "Financial Accounting Standard") -> str:
    return f"""
You are an expert in Islamic finance and legal document structuring.

Given the following {doc_type} content, extract **independent, semantically meaningful text chunks** suitable for retrieval and indexing. Each chunk must retain clear context and boundaries (e.g., a full paragraph, a clause, or a grouped rule). Do not summarize or interpret — preserve the original wording exactly. 

As the chunking process is being done in a batch mode, you have the Context to keep from the last batc: {context_to_keep}. You can use this context to help you chunk the text better. and follow it up with the next batch.

For each chunk, output a flat JSON entry with:
- "text": the exact original text of the chunk, feel free to append enough context to make it meaningful / standalone
- "type": one of ["rule", "definition", "treatment", "disclosure", "juristic", "historical", "misc"]
- "keywords": a list of relevant keywords or phrases that capture the essence of the chunk

Guidelines:
- The chunks that you return should be semantically meaningful and contextually complete.
- These chunks will be used for Retreival-Augmented Generation (RAG) tasks, so they should be suitable for that purpose.
- Feel free to remove unnecessary whitespace, points, new lines, and other formatting issues.
- Please append to each chunk enough context to make it meaningful / standalone. 

Format:
{{
    "chunks": [
        {{
            "text": "...",
            "type": "...",
            "keywords": ["..."]
        }},
        ...
    ],
    "context_to_keep": "..."
}}

Here is the text:
\"\"\"
{text}
\"\"\"
"""
