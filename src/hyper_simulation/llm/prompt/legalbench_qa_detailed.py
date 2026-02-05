QA_CONTRACT_BASE = """### Contract Clause:
{context_text}

### Question:
{question}

### Instructions:
You are a contract analyst. Answer questions about the contract clause above.

Guidelines:
- Answer based ONLY on the contract text provided
- For questions about specific clauses: identify and quote the relevant clause
- For Yes/No questions about clause presence: answer "Yes" or "No"
- If the clause or information is not present, answer "Not specified in the contract"
- Be precise and avoid interpretation not directly supported by text

Answer format:
- Keep answer concise (1-2 sentences)
- Quote specific clauses when relevant

### Answer:
"""

QA_CONSUMER_BASE = """### Terms of Service:
{context_text}

### Question:
{question}

### Instructions:
You are analyzing terms of service or user agreements. Answer questions about user rights, obligations, and policies.

Guidelines:
- Answer based ONLY on the ToS/agreement provided
- Identify relevant sections that support your answer
- For Yes/No questions about user rights or obligations: answer clearly
- If information is not in the agreement, state "Not specified in the agreement"
- Focus on what the company requires or promises, not on general knowledge

Answer format:
- Keep answer concise and direct (1-2 sentences)
- Mention specific rights or obligations when relevant

### Answer:
"""

QA_PRIVACY_BASE = """### Privacy Policy:
{context_text}

### Question:
{question}

### Instructions:
You are analyzing privacy policies. Answer questions about data collection, use, and sharing.

Guidelines:
- Answer based ONLY on the privacy policy text provided
- Identify what data is collected and how it's used/shared
- For Yes/No questions: answer clearly with supporting evidence
- If the policy doesn't specify something, state "The policy does not specify"
- Be literal and avoid assumptions about what companies might do

Answer format:
- Keep answer concise (1-2 sentences)
- Quote specific data practices when relevant

### Answer:
"""

QA_RULE_BASE = """### Rule Definition:
{context_text}

### Question:
{question}

### Instructions:
You are a logic analyzer. Based on the rules and facts provided, determine the correct answer through logical reasoning.

Guidelines:
- Apply the rules strictly as defined
- Follow the logical chain: if condition A and B are met, then conclusion C applies
- For Yes/No questions: determine the answer by checking if conditions are met
- Be explicit about which rules and facts lead to your conclusion
- Do not add assumptions beyond what the rules state

Answer format:
- Provide a clear Yes/No answer or the logical conclusion
- Show the reasoning: which rules/facts led to this answer

### Answer:
"""
