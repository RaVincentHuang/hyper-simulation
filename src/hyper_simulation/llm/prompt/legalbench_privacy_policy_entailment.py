LEGALBENCH_PRIVACY_POLICY_ENTAILMENT_BASE = """### Privacy Policy:
{context_text}

### Statement to Evaluate:
{question}

### Instructions:
You are a privacy policy analyst. Determine if the statement is consistent with the privacy policy.

**Output Format (STRICT):**
- Output exactly ONE line: ### Final Answer: <your answer>
- Choose ONLY one of these two labels:
  • ### Final Answer: Correct      (if the statement is supported by the policy)
  • ### Final Answer: Incorrect    (if the statement contradicts or is not supported by the policy)
- Do NOT include any reasoning, explanations, or intermediate steps

**Examples:**
✓ Correct: ### Final Answer: Correct
✓ Correct: ### Final Answer: Incorrect
✗ Wrong: The statement is correct because...
✗ Wrong: ### Final Answer: Entails  # 注意：用 Correct 不是 Entails

### Response:
"""