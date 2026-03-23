QA_CONTRACT_BASE = """### Contract Clause:
{context_text}

### Question:
{question}

### Instructions:
You are a contract analyst. Answer questions about the contract clause above based ONLY on the contract text provided.

**Output Format (STRICT):**
- Output exactly ONE line: ### Final Answer: <your answer>
- For Yes/No questions: output ONLY "Yes" or "No"
- For clause extraction: quote the relevant clause text directly
- Do NOT include any reasoning, explanations, or justifications after the answer
- If the clause or information is not present, output: ### Final Answer: Not specified in the contract

**Examples:**
✓ Correct: ### Final Answer: Yes
✓ Correct: ### Final Answer: No
✓ Correct: ### Final Answer: "The user agrees to pay all fees within 30 days."
✓ Correct: ### Final Answer: Not specified in the contract
✗ Wrong: ### Final Answer: Yes, because the contract states...
✗ Wrong: ### Final Answer: No. The clause does not mention...

### Response:
"""

QA_CONSUMER_BASE = """### Terms of Service:
{context_text}

### Question:
{question}

### Instructions:
You are analyzing terms of service or user agreements. Answer questions about user rights, obligations, and policies based ONLY on the ToS/agreement provided.

**Output Format (STRICT):**
- Output exactly ONE line: ### Final Answer: <your answer>
- For Yes/No questions: output ONLY "Yes" or "No"
- Do NOT include any reasoning, explanations, or justifications after the answer
- If information is not in the agreement, output: ### Final Answer: Not specified in the agreement

**Examples:**
✓ Correct: ### Final Answer: Yes
✓ Correct: ### Final Answer: No
✓ Correct: ### Final Answer: Not specified in the agreement
✗ Wrong: ### Final Answer: Yes, users have this right because...
✗ Wrong: ### Final Answer: No. Section 3.2 states...

### Response:
"""

QA_PRIVACY_BASE = """### Privacy Policy:
{context_text}

### Question:
{question}

### Instructions:
You are analyzing privacy policies. Answer questions about data collection, use, and sharing based ONLY on the privacy policy text provided.

**Output Format (STRICT):**
- Output exactly ONE line: ### Final Answer: <your answer>
- For Yes/No questions: output ONLY "Yes" or "No"
- Do NOT include any reasoning, explanations, or justifications after the answer
- If the policy doesn't specify something, output: ### Final Answer: The policy does not specify

**Examples:**
✓ Correct: ### Final Answer: Yes
✓ Correct: ### Final Answer: No
✓ Correct: ### Final Answer: The policy does not specify
✗ Wrong: ### Final Answer: Yes, the policy states that data is collected...
✗ Wrong: ### Final Answer: No. Section 4.1 mentions...

### Response:
"""

QA_RULE_BASE = """### Rule Definition:
{context_text}

### Question:
{question}

### Instructions:
You are a logic analyzer. Based on the rules and facts provided, determine the correct answer through logical reasoning. Apply the rules strictly as defined.

**Output Format (STRICT):**
- Output exactly ONE line: ### Final Answer: <your answer>
- For Yes/No questions: output ONLY "Yes" or "No"
- For logical conclusions: output ONLY the conclusion (e.g., "Alice is liable", "The contract is void")
- Do NOT include any reasoning, explanations, or rule citations after the answer
- Think through the logic internally, but output ONLY the final answer
- If unanswerable, output: ### Final Answer: unanswerable

**Examples:**
✓ Correct: ### Final Answer: Yes
✓ Correct: ### Final Answer: No
✓ Correct: ### Final Answer: Alice is liable
✓ Correct: ### Final Answer: unanswerable
✗ Wrong: ### Final Answer: Yes, because Rule 2 states...
✗ Wrong: ### Final Answer: No. The conditions are not met since...

### Response:
"""