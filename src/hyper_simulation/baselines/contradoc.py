import re
from ast import mod
from hyper_simulation.llm.prompt.contradoc import contradoc_prompt
from hyper_simulation.query_instance import QueryInstance

from langchain_ollama import ChatOllama

from hyper_simulation.llm.chat_completion import get_generate

import json

def judge_contradiction_batch(doc_a_list: list[str], doc_b_list: list[str], model: ChatOllama) -> list[tuple[bool, str]]:
    # input two documents and judge whether there is a contradiction
    # output: list[tuple[bool, str]]
    prompts = [contradoc_prompt.format(doc_a=doc_a, doc_b=doc_b) for doc_a, doc_b in zip(doc_a_list, doc_b_list)]
    responses = get_generate(prompts=prompts, model=model)
    
    results = []
    for response in responses:
        has_contradiction = False
        evidence_str = ""
        
        # Parse judgment (yes/no)
        judgment_match = re.search(r'Judgment:\s*(yes|no)', response, re.IGNORECASE)
        if judgment_match:
            has_contradiction = judgment_match.group(1).lower() == 'yes'
        
        # Parse evidence
        evidence_match = re.search(r'Evidence:\s*(\[.*?\])', response, re.DOTALL)
        if evidence_match:
            evidence_text = evidence_match.group(1)
            try:
                evidence_list = json.loads(evidence_text)
                if evidence_list:
                    # Convert each evidence pair to string and join with newline
                    evidence_str = "\n".join([" | ".join(pair) if isinstance(pair, list) else str(pair) for pair in evidence_list])
            except (json.JSONDecodeError, ValueError):
                evidence_str = evidence_text
        
        results.append((has_contradiction, evidence_str))
    
    return results

def query_fixup(query: QueryInstance, model: ChatOllama) -> QueryInstance:
    
    # use judge_contradiction_batch to add fixed_data to query
    # for all data in query.data, judge contradictions using judge_contradiction_batch
    # if contradiction found, add string "[INCONSISTENT DETECTED]" and the evidence to fixed_data
    fixed_data = []
    judgments = judge_contradiction_batch([query.query]*len(query.data), query.data, model=model)
    for doc, (has_contradiction, evidence) in zip(query.data, judgments):
        if has_contradiction:
            fixed_doc = f"{doc}\n[INCONSISTENT DETECTED!]\nEvidence:\n{evidence}"
        else:
            fixed_doc = doc
        fixed_data.append(fixed_doc)
    query.fixed_data = fixed_data
    return query
