import re
from hyper_simulation.llm.prompt.contradoc import contradoc_prompt, contradoc_entailment_prompt
from hyper_simulation.query_instance import QueryInstance

from langchain_ollama import ChatOllama

from hyper_simulation.llm.chat_completion import get_generate

import json


def _parse_evidence(response_text: str) -> str:
    evidence_match = re.search(r'Evidence:\s*(\[.*?\])', response_text, re.DOTALL)
    if not evidence_match:
        return ""
    evidence_text = evidence_match.group(1)
    try:
        evidence_list = json.loads(evidence_text)
        if evidence_list:
            return "\n".join([" | ".join(pair) if isinstance(pair, list) else str(pair) for pair in evidence_list])
    except (json.JSONDecodeError, ValueError):
        return evidence_text
    return ""


def _parse_yes_no_judgment(response_text: str) -> bool | None:
    match = re.search(r'judg(?:e)?ment\s*:\s*(yes|no)', response_text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower().strip() == 'yes'

def judge_contradiction_batch(doc_a_list: list[str], doc_b_list: list[str], model: ChatOllama) -> list[tuple[bool, str]]:
    # input two documents and judge whether there is a contradiction
    # output: list[tuple[bool, str]]
    prompts = [contradoc_prompt.format(doc_a=doc_a, doc_b=doc_b) for doc_a, doc_b in zip(doc_a_list, doc_b_list)]
    responses = get_generate(prompts=prompts, model=model)
    
    results = []
    for response in responses:
        has_contradiction = False
        response_text = str(response or "")
        response_lower = response_text.lower()
        
        # Parse judgment with tolerant formats (yes/no and contradiction labels).
        judgment_match = re.search(
            r'judg(?:e)?ment\s*:\s*(yes|no|contradiction|non-contradiction|entailment|neutral)',
            response_text,
            re.IGNORECASE,
        )
        if judgment_match:
            token = judgment_match.group(1).lower().strip()
            if token in {"yes", "contradiction"}:
                has_contradiction = True
            elif token in {"no", "non-contradiction", "entailment", "neutral"}:
                has_contradiction = False
        else:
            # Fallback parsing for loosely formatted outputs.
            if "non-contradiction" in response_lower or "no contradiction" in response_lower:
                has_contradiction = False
            elif "contradiction" in response_lower:
                has_contradiction = True
        evidence_str = _parse_evidence(response_text)
        
        results.append((has_contradiction, evidence_str))
    
    return results


def judge_entailment_batch(doc_a_list: list[str], doc_b_list: list[str], model: ChatOllama) -> list[tuple[bool, str]]:
    # input two documents and judge whether doc_b entails doc_a
    # output: list[tuple[bool, str]]
    prompts = [contradoc_entailment_prompt.format(doc_a=doc_a, doc_b=doc_b) for doc_a, doc_b in zip(doc_a_list, doc_b_list)]
    responses = get_generate(prompts=prompts, model=model)

    results = []
    for response in responses:
        response_text = str(response or "")
        response_lower = response_text.lower()

        judgment = _parse_yes_no_judgment(response_text)
        if judgment is None:
            if "non-entailment" in response_lower or "not entail" in response_lower:
                is_entailment = False
            elif "entailment" in response_lower:
                is_entailment = True
            else:
                is_entailment = False
        else:
            is_entailment = judgment

        evidence_str = _parse_evidence(response_text)
        results.append((is_entailment, evidence_str))

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
