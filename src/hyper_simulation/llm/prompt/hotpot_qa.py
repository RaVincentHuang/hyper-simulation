HOTPOT_QA_BASE = """### Background:
{context_text}

### Question:
{question}

### Instructions:
Answer the question using only the background information.

**Output Format (STRICT):**
- Output exactly ONE line: ### Final Answer: <your answer>
- Do NOT include any reasoning, explanations, or extra text
- If unanswerable, output: ### Final Answer: unanswerable

### Response:
"""

HOTPOT_QA_HYPER = """### Background (Ranked by Relevance, Highest to Lowest):
{context_text}

### Question:
{question}

### Instructions:
Answer the question using only the background information.

**Output Format (STRICT):**
- Output exactly ONE line: ### Final Answer: <your answer>
- Do NOT include any reasoning, explanations, or extra text
- If unanswerable, output: ### Final Answer: unanswerable

### Response:
"""