LEGALBENCH_CORPORATE_LOBBYING_BASE = """### Bill:
{context_text}

### Question:
{question}

### Instructions:
You are analyzing whether a proposed bill is relevant to a specific company's business interests.
Determine if the company would likely take an interest in (lobby for/against) this bill.

**Please answer directly following these rules:**
1. Think through the question internally, but **do not output your reasoning process**
2. Consider the bill's subject matter and potential impact, the company's likely business interests and sector, and whether the bill could affect the company's operations, revenue, or obligations.
3. Output **only** the final answer ("Yes" or "No") in this exact format:
   ### Final Answer: <your answer>
4. If unanswerable, output:
   ### Final Answer: unanswerable

**Important:** Do not include any explanation, step-by-step analysis, or intermediate thoughts. Just output the final answer line.

### Response:
"""
