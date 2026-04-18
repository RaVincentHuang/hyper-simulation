MUSIQUE_QA_BASE = """### Context:
{context_text}

### Question:
{question}

### Instructions:
This is a multi-hop reasoning question. 

**Please answer directly following these rules:**
1. Combine information from multiple paragraphs
2. Output **only** the final answer in this exact format:
   ### Final Answer: <your answer>
3. If the answer cannot be determined from the context, output:
   ### Final Answer: unanswerable

**Please answer directly following these rules:**
Output **only** the final answer in this exact format:
### Final Answer: <your answer>
"""


MUSIQUE_QA_HYPER = """### Context (Priority Ordered):
{context_text}

### Question:
{question}

### Instructions:
You are solving a MuSiQue multi-hop QA step with ranked contexts.

Rules:
1. Context blocks are priority-ordered from top to bottom.
2. Prefer earlier context blocks when evidence conflicts.
3. Use later context only as supplemental evidence.
4. If evidence is insufficient, answer conservatively.
5. Beware of Distractors: Some context blocks may share keywords with the question but are logically irrelevant. Do not fall for lexical overlap.
6. The provided contexts ensure that the question can be answered, meaning the answer can be derived from the information in the context blocks. However, not all context blocks may be relevant, and some may even contain misleading information. 
7. DO NOT answer as `unanswerable` or 'the provided context is insufficient', etc., since the question is guaranteed to be answerable based on the provided contexts. Instead, if you find the information insufficient or conflicting, make your best effort to infer the answer based on the available evidence, while adhering to the priority order of the contexts.

Output **only** the final answer in this exact format:
### Final Answer: <your answer>
"""
