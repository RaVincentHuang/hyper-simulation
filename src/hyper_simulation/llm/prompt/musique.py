MUSIQUE_QA_BASE = """### Context:
{context_text}

### Question:
{question}

### Instructions:
This is a multi-hop reasoning question. 

**Please answer directly following these rules:**
1. Think through the question internally, but **do not output your reasoning process**
2. Combine information from multiple paragraphs silently
3. Output **only** the final answer in this exact format:
   ### Final Answer: <your answer>
4. If the answer cannot be determined from the context, output:
   ### Final Answer: unanswerable

**Important:** Do not include any explanation, step-by-step analysis, or intermediate thoughts. Just output the final answer line.

### Response:
"""