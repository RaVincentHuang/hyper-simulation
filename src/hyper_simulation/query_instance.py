
from dataclasses import dataclass
from typing import Any, List, Union, Dict, Optional

@dataclass
class QueryInstance:
    query: str
    data: list[str]# top-k, H1, ..., Hk => H
    # H_Q, H
    # (_, C2) in HC -> HC2, v in reach(u), (1) exists a C sequence C1, ..., Cn, n >= 1, u in C1, v in Cn, Ci iter Ci+1 != empty  
    answers: list[str]
    ground_truth: list[tuple[bool, Any]]  # list of (has_contradiction, evidence)
    fixed_data: Union[List[str], None] = None
    query_decomposition: Union[List[str], None] = None
    
    simulation_logs: Union[List[str], None] = None
    denial_logs: Union[List[str], None] = None
    semantic_cluster_logs: Union[List[str], None] = None
    d_match_logs: Union[List[str], None] = None
    context_type: str = None

    def add_simulation_log(self, log: str, data_id: int) -> None:
        if self.simulation_logs is None:
            self.simulation_logs = []
        self.simulation_logs.append(f"[{data_id}] {log}\n")
    
    def add_denial_log(self, log: str, data_id: int) -> None:
        if self.denial_logs is None:
            self.denial_logs = []
        self.denial_logs.append(f"[{data_id}] {log}\n")
        
    def add_semantic_cluster_log(self, log: str, data_id: int) -> None:
        if self.semantic_cluster_logs is None:
            self.semantic_cluster_logs = []
        self.semantic_cluster_logs.append(f"[{data_id}] {log}\n")
    
    def add_d_match_log(self, log: str, data_id: int) -> None:
        if self.d_match_logs is None:
            self.d_match_logs = []
        self.d_match_logs.append(f"[{data_id}] {log}\n")


def build_query_instance_for_task(item: Dict[str, Any], task: str) -> QueryInstance:
    """
    根据任务类型构建 QueryInstance
    
    Args:
        item: 原始数据项（从 load_data 加载的 dict）
        task: 任务类型 (hotpotqa, musique, multihop, legalbench, qa/*, ARC)
    
    Returns:
        QueryInstance 对象
    """
    if task == "hotpotqa":
        supporting_facts = item.get('supporting_facts', {})
        ground_truths = []
        titles_set = set(supporting_facts.get('title', []))
        
        for title, sentences in item['context']:
            if title in titles_set:
                has_contradiction = True
                sent_ids = supporting_facts.get('sent_id', [])
                evidence_sentences = [sentences[i] for i in sent_ids if i < len(sentences)]
                evidence = "\n".join(evidence_sentences)
                ground_truths.append((has_contradiction, evidence))
            else:
                ground_truths.append((False, ""))
        
        return QueryInstance(
            query=item['question'],
            data=["\n".join(sentences) for title, sentences in item['context']],
            fixed_data=[],
            answers=item['answer'],
            ground_truth=ground_truths
        )
    
    elif task == "musique":
        ground_truths = []
        supporting_flags = item.get("supporting_flags", []) or []
        
        for idx, (title, sentences) in enumerate(item["context"]):
            is_supporting = supporting_flags[idx] if idx < len(supporting_flags) else False
            if is_supporting:
                has_contradiction = True
                evidence = "\n".join(sentences)
                ground_truths.append((has_contradiction, evidence))
            else:
                ground_truths.append((False, ""))
        
        answer = item.get("answer", "")
        aliases = item.get("answer_alias", []) or []
        answers = [answer] + [a for a in aliases if a != answer]
        
        # 处理 question_decomposition
        raw_decomposition = item.get("question_decomposition", []) or []
        query_decomposition: Optional[List[str]] = None
        if isinstance(raw_decomposition, list) and raw_decomposition:
            if all(isinstance(d, dict) and "id" in d for d in raw_decomposition):
                sorted_decomposition = sorted(raw_decomposition, key=lambda d: d.get("id"))
            else:
                sorted_decomposition = raw_decomposition
            query_decomposition = [
                (d.get("question") or "").strip() 
                for d in sorted_decomposition 
                if isinstance(d, dict)
            ]
        
        return QueryInstance(
            query=item["question"],
            data=["\n".join(sentences) for title, sentences in item["context"]],
            fixed_data=[],
            answers=answers,
            ground_truth=ground_truths,
            query_decomposition=query_decomposition
        )
    
    elif task == "multihop":
        ground_truths = []
        supporting_flags = item.get("supporting_flags", []) or []
        
        for idx, (title, sentences) in enumerate(item["context"]):
            is_supporting = supporting_flags[idx] if idx < len(supporting_flags) else False
            if is_supporting:
                has_contradiction = True
                evidence = "\n".join(sentences)
                ground_truths.append((has_contradiction, evidence))
            else:
                ground_truths.append((False, ""))
        
        answer = item.get("answer", "")
        answers = [answer] if answer else []
        
        return QueryInstance(
            query=item["question"],
            data=["\n".join(sentences) for title, sentences in item["context"]],
            fixed_data=[],
            answers=answers,
            ground_truth=ground_truths
        )
    
    elif task == "legalbench" or task.startswith("legalbench/"):
        ground_truths = []
        context_type = item.get("context_type", "legal_document")
        supporting_flags = item.get("supporting_flags", []) or []
        
        for idx, (title, sentences) in enumerate(item["context"]):
            is_supporting = supporting_flags[idx] if idx < len(supporting_flags) else True
            if is_supporting:
                has_contradiction = True
                evidence = "\n".join(sentences)
                ground_truths.append((has_contradiction, evidence))
            else:
                ground_truths.append((False, ""))
        
        answer = item.get("answer", "")
        answers = [answer] if answer else []
        
        return QueryInstance(
            query=item["question"],
            data=["\n".join(sentences) for title, sentences in item["context"]],
            fixed_data=[],
            answers=answers,
            ground_truth=ground_truths,
            context_type=context_type,
        )
    
    elif task in ("qa/contract", "qa/consumer", "qa/privacy", "qa/rule"):
        ground_truths = []
        context_type = item.get("context_type", "legal_document")
        supporting_flags = item.get("supporting_flags", []) or [True]
        
        for idx, (title, sentences) in enumerate(item["context"]):
            is_supporting = supporting_flags[idx] if idx < len(supporting_flags) else True
            if is_supporting:
                has_contradiction = True
                evidence = "\n".join(sentences)
                ground_truths.append((has_contradiction, evidence))
            else:
                ground_truths.append((False, ""))
        
        answer = item.get("answer", "")
        answers = [answer] if answer else []
        
        return QueryInstance(
            query=item["question"],
            data=["\n".join(sentences) for title, sentences in item["context"]],
            fixed_data=[],
            answers=answers,
            ground_truth=ground_truths
        )
    
    elif task == "ARC":
        return QueryInstance(
            query=item['question'],
            data=[],
            fixed_data=[],
            answers=[item['answer_label']],
            ground_truth=[]
        )
    
    else:
        raise ValueError(f"Unsupported task: {task}")
