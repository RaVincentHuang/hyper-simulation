contradoc_prompt = """
You are an expert at checking consistency between a user question and retrieved context in a QA pipeline.
Document A is always a question. Document B is RAG context that may support, contradict, or be irrelevant to the question.

Treat every explicit premise or implied comparison in the question as a fact to be checked (entities, numbers, dates, superlatives, comparisons, options mentioned, etc.).

**When to call contradiction (Yes):**
- Document B states a fact that is mutually exclusive with a premise in Document A (e.g., different entity attribute, number, date, location, role, or comparison outcome).
- Document B asserts an answer to the question that conflicts or inconsistent with the premise or implied answer in Document A.

**When to call no contradiction (No):**
- Document B is neutral/irrelevant or lacks the information needed to confirm/deny the question.
- Document B only partially overlaps but does not conflict.

**Instructions:**
1. Read the question (Document A) and context (Document B).
2. Decide if a factual contradiction exists (Yes/No) under the above rules.
3. If **Yes**: Provide the minimal evidence as a Python list of exactly two strings:
   - The specific clause/sentence from Document A (the question) that contains the contradictory premise or implied answer.
   - The contradictory sentence from Document B.
4. If **No**: Provide an empty list.

--- Document A ---
{doc_a}

--- Document B ---
{doc_b}
------------------

**Response Format:**
Please strictly follow the format below (do not provide explanations, only the format):

Judgment: yes OR no
Evidence: [["sentence_from_doc_A", "sentence_from_doc_B"], ...] OR []
"""
