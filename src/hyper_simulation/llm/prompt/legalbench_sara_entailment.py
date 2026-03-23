LEGALBENCH_SARA_ENTAILMENT_BASE = """### Legal Text:
{context_text}

### Statement to Evaluate:
{question}

### Instructions:
You are a legal reasoning system. Determine the relationship between the legal text and the given statement.

**Output Format (STRICT):**
- Output exactly ONE line: ### Final Answer: <your answer>
- Choose ONLY one of these three labels:
  • ### Final Answer: Entails
  • ### Final Answer: Contradicts  
  • ### Final Answer: Neutral
- Do NOT include any reasoning, explanations, or intermediate steps

### Response:
"""