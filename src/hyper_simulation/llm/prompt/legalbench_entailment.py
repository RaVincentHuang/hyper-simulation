LEGALBENCH_ENTAILMENT_BASE = """### Legal Text:
{context_text}

### Statement:
{question}

### Instructions:
You are a legal reasoning system. Determine the relationship between the legal text and the given statement.

Output one of three labels:
- "Entails": The legal text logically entails (supports/implies) the statement
- "Contradicts": The legal text contradicts the statement
- "Neutral": The legal text neither entails nor contradicts the statement

Provide only the label, no explanation.

### Answer:
"""
