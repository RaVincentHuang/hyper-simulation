MUSIQUE_QA_BASE = """### Context:
{context_text}

### Question:
{question}

### Instructions:
You are given multiple paragraphs. The question may require multi-hop reasoning.
Follow these steps internally:
1) Identify which paragraphs contain relevant clues.
2) Combine information across paragraphs to derive the final answer.
3) If multiple facts are needed, ensure the chain of reasoning is consistent.

Answering rules:
- Output a concise final answer only (no explanation).
- If the answer is a short phrase or entity, output only that.
- If the answer is a number or date, output only the value.
- If the question is unanswerable from the context, output "unanswerable".

### Answer:
"""
