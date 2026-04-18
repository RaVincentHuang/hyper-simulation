MULTIHOP_QA_BASE = """### Context:
{context_text}

### Question:
{question}

### Instructions:
You are given multiple evidence passages. The question may require multi-hop reasoning.
Think silently and output ONLY the final answer in this exact format:
### Final Answer: <your answer>
If unanswerable, output: ### Final Answer: unanswerable

### Response:
"""

MULTIHOP_QA_HYPER = """### Context (Priority Ordered):
{context_text}

### Question:
{question} 

### Sub-question that needs to be answered NOW:
{sub_question}

### Instructions:
You are solving a multi-hop QA step with ranked contexts.
Rules:
1. Answer the sub-question based on the provided contexts ONLY!
2. Context blocks are priority-ordered from top to bottom.
3. Prefer earlier context blocks when evidence conflicts.
4. Use later context only as supplemental evidence.
5. If evidence is insufficient, answer conservatively.
6. The provided contexts ensure that the question can be answered, meaning the answer can be derived from the information in the context blocks. However, not all context blocks may be relevant, and some may even contain misleading information. 
7. DO NOT answer as `unanswerable` or 'the provided context is insufficient', etc., since the question is guaranteed to be answerable based on the provided contexts. Instead, if you find the information insufficient or conflicting, make your best effort to infer the answer based on the available evidence, while adhering to the
priority order of the contexts. 

Output **only** the final answer in this exact format:
### Final Answer: <your answer>

### Response:
"""

MULTIHOP_QA_HYPER_SUB = """### Context (Priority Ordered):
{context_text}

### Question:
{question}

### Sub-questions Answered before:
{sub_questions_answered}

### Sub-question that needs to be answered NOW:
{sub_question}


### Instructions:
You are solving a multi-hop QA step with ranked contexts.
Rules:
1. Answer the current sub-question by combining the provided contexts AND the information from 'Sub-questions Answered before'.
2. Context blocks are priority-ordered from top to bottom.
3. Prefer earlier context blocks when evidence conflicts.
4. Use later context only as supplemental evidence.
5. If evidence is insufficient, answer conservatively.
6. The provided contexts ensure that the question can be answered, meaning the answer can be derived from the information in the context blocks. However, not all context blocks may be relevant, and some may even contain misleading information. 
7. DO NOT answer as `unanswerable` or 'the provided context is insufficient', etc., since the question is guaranteed to be answerable based on the provided contexts. Instead, if you find the information insufficient or conflicting, make your best effort to infer the answer based on the available evidence, while adhering to the
priority order of the contexts. 

Output **only** the final answer in this exact format:
### Final Answer: <your answer>

### Response:
"""

MULTIHOP_QA_HYPER_FINAL = """### Context (Priority Ordered):
{context_text}

### Sub-questions Answered before:
{sub_questions}

### Question:
{question}

### Instructions:
You are solving a multi-hop QA step with ranked contexts.
Rules:
1. Deduce the answer to the main Question based on the provided contexts and the logical chain in 'Sub-questions Answered before'.
2. Context blocks are priority-ordered from top to bottom.
3. Prefer earlier context blocks when evidence conflicts.
4. Use later context only as supplemental evidence.
5. If evidence is insufficient, answer conservatively.
6. The provided contexts ensure that the question can be answered, meaning the answer can be derived from the information in the context blocks. However, not all context blocks may be relevant, and some may even contain misleading information. 
7. DO NOT answer as `unanswerable` or 'the provided context is insufficient', etc., since the question is guaranteed to be answerable based on the provided contexts. Instead, if you find the information insufficient or conflicting, make your best effort to infer the answer based on the available evidence, while adhering to the
priority order of the contexts. 

Output **only** the final answer in this exact format:
### Final Answer: <your answer>

### Response:
"""
