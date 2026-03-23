LEGALBENCH_QA_BASE = """### Legal Document:
{context_text}

### Question:
{question}

### Instructions:
You are a legal document analyzer. Answer the question based on the provided legal document.

**Please answer directly following these rules:**
1. Think through the question internally, but **do not output your reasoning process**
2. Be concise and accurate. Cite relevant clauses or sections when possible.
3. Output **only** the final answer in this exact format:
   ### Final Answer: <your answer>
4. If information is not available in the document, output:
   ### Final Answer: Not specified in the document

**Important:** Do not include any explanation, step-by-step analysis, or intermediate thoughts. Just output the final answer line.

### Response:
"""
