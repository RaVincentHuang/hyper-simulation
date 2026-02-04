LEGALBENCH_INSURANCE_BASE = """### Insurance Policy:
{context_text}

### Claim Scenario:
{question}

### Instructions:
You are an insurance underwriter. Determine if the insurance policy would cover the claim in the given scenario.

Consider:
1) Coverage scope and applicability
2) Exclusions and limitations
3) Conditions that must be met

Output "Yes" if the policy would likely cover the claim, or "No" if it would likely deny the claim.
Provide only the answer, no explanation.

### Answer:
"""
