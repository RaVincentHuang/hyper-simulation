LEGALBENCH_CORPORATE_LOBBYING_BASE = """### Bill:
{context_text}

### Question:
{question}

### Instructions:
You are analyzing whether a proposed bill is relevant to a specific company's business interests.
Determine if the company would likely take an interest in (lobby for/against) this bill.

Consider:
1) The bill's subject matter and potential impact
2) The company's likely business interests and sector
3) Whether the bill could affect the company's operations, revenue, or obligations

Output "Yes" if the bill is relevant to the company's interests, or "No" if it is not.
Provide only the answer, no explanation.

### Answer:
"""
