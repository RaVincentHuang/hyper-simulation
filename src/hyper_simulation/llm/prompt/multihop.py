MULTIHOP_QA_BASE = """### Context:
{context_text}

### Question:
{question}

### Instructions:
You are given multiple evidence passages. The question may require multi-hop reasoning.
Follow these steps internally:
1) Identify relevant evidence pieces.
2) Link facts across evidence to reach the final answer.
3) Ensure the final answer is supported by the given evidence.

Answering rules:
- Output a concise final answer only (no explanation).
- If the answer is a short phrase or entity, output only that.
- If the answer is a number or date, output only the value.
- If the question is unanswerable from the context, output "unanswerable".

### Answer:
"""
