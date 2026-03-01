MUSIQUE_QA_BASE = """### Context:
{context_text}

### Question:
{question}

### Instructions:
This is a multi-hop reasoning question. You need to combine information from multiple paragraphs.

Please follow these steps:

**Step 1: Identify relevant paragraphs**
- List which paragraphs contain information related to the question
- Quote the key sentences

**Step 2: Break down the question**
- Identify sub-questions that need to be answered first
- Solve each sub-question using the context

**Step 3: Combine the information**
- Connect the answers from sub-questions to derive the final answer
- Check for consistency across paragraphs

**Step 4: Output the final answer**
- After your reasoning, output the final answer in this exact format:
  ### Final Answer: <your answer>
- If the answer cannot be determined from the context, output:
  ### Final Answer: unanswerable

### Response:
"""