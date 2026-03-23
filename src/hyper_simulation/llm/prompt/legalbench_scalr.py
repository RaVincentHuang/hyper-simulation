LEGALBENCH_SCALR_BASE = """### Supreme Court Case and Legal Principle:
{context_text}

### Question:
{question}

### Instructions:
You are a legal analyst trained on U.S. Supreme Court precedents.

**Output Format (STRICT):**
- If options are provided (A, B, C, etc.), output ONLY the option letter
- Output exactly ONE line: ### Final Answer: <your answer>
- Do NOT include any reasoning, explanations, or the option text
- Example: ### Final Answer: A
- If unanswerable, output: ### Final Answer: unanswerable

### Response:
"""