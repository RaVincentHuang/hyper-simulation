import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from hyper_simulation.query_instance import QueryInstance
from hyper_simulation.component.build_hypergraph import test_build_hypergraph_for_query_instance
from hyper_simulation.utils.clean import clean_text_for_spacy
from pathlib import Path
import json
import jsonlines

def load_hotpotqa_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载HotpotQA数据集
    
    支持的schema:
    - id: 问题ID
    - question: 问题文本
    - answer: 正确答案
    - type: 问题类型 (comparison, bridge)
    - level: 难度级别 (easy, medium, hard)
    - supporting_facts: {title: [...], sent_id: [...]}
    - context: {title: [...], sentences: [[...], ...]}
    """
    data = []
    path = Path(file_path)
    
    # collect all hotpot_*.jsonl in `file_path` if it's a directory
    paths = []
    if path.is_dir():
        paths = list(path.glob('hotpot_*.jsonl'))
    else:
        paths = [path]
    
    for path in paths:
        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                for item in raw_data:
                    formatted_item = {
                        '_id': item.get('id', ''),
                        'question': item.get('question', '').strip(),
                        'answer': item.get('answer', '').strip(),
                        'type': item.get('type', 'unknown'),
                        'level': item.get('level', 'unknown'),
                        'context': [
                            (title, sentences)
                            for title, sentences in zip(
                                item['context']['title'],
                                item['context']['sentences']
                            )
                        ],
                        'supporting_facts': item.get('supporting_facts', {}),
                    }
                    data.append(formatted_item)
        elif path.suffix == '.jsonl':
            with jsonlines.open(path, 'r') as reader:
                for item in reader:
                    formatted_item = {
                        '_id': item.get('id', ''),
                        'question': item.get('question', '').strip(),
                        'answer': item.get('answer', '').strip(),
                        'type': item.get('type', 'unknown'),
                        'level': item.get('level', 'unknown'),
                        'context': [
                            (title, sentences)
                            for title, sentences in zip(
                                item['context']['title'],
                                item['context']['sentences']
                            )
                        ],
                        'supporting_facts': item.get('supporting_facts', {}),
                    }
                    data.append(formatted_item)
        else:
            raise ValueError("Unsupported file format. Please use .json or .jsonl")
    
    return data


def load_musique_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载MuSiQue数据集（jsonl）

    支持的schema:
    - id: 问题ID
    - question: 问题文本
    - answer: 正确答案
    - answer_alias: 答案别名
    - answerable: 是否可回答
    - paragraphs: [{idx, title, paragraph_text, is_supporting}]
    - question_decomposition: 多跳子问题
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    # collect all *.jsonl in `file_path` if it's a directory
    paths = []
    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            raise ValueError("Unsupported file format. Please use .jsonl")

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                paragraphs = item.get("paragraphs", []) or []

                context = []
                supporting_titles = []
                supporting_sent_ids = []
                supporting_flags = []

                for p in paragraphs:
                    title = (p.get("title") or "").strip()
                    paragraph_text = (p.get("paragraph_text") or "").strip()

                    if title or paragraph_text:
                        context.append((title, [paragraph_text] if paragraph_text else []))
                        is_supporting = bool(p.get("is_supporting", False))
                        supporting_flags.append(is_supporting)

                        if is_supporting and title:
                            supporting_titles.append(title)
                            supporting_sent_ids.append(0)

                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": (item.get("question") or "").strip(),
                    "answer": (item.get("answer") or "").strip(),
                    "answer_alias": item.get("answer_alias", []) or [],
                    "answerable": item.get("answerable", True),
                    "question_decomposition": item.get("question_decomposition", []) or [],
                    "context": context,
                    "supporting_flags": supporting_flags,
                }
                data.append(formatted_item)

    return data


def load_multihop_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载多跳推理数据集（jsonl）

    支持的schema:
    - query: 问题文本
    - answer: 正确答案
    - question_type: 问题类型
    - evidence_list: [{title, text, published_at, source}]
    - metadata: 其他元数据
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    # collect all *.jsonl in `file_path` if it's a directory
    paths = []
    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            raise ValueError("Unsupported file format. Please use .jsonl")

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                evidence_list = item.get("evidence_list", []) or []

                context = []
                supporting_flags = []
                for evidence in evidence_list:
                    title = (evidence.get("title") or "").strip()
                    text = (evidence.get("text") or "").strip()
                    if title or text:
                        context.append((title, [text] if text else []))
                        supporting_flags.append(True)

                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": (item.get("query") or "").strip(),
                    "answer": (item.get("answer") or "").strip(),
                    "question_type": item.get("question_type", ""),
                    "metadata": item.get("metadata", []) or [],
                    "context": context,
                    "supporting_flags": supporting_flags,
                }
                data.append(formatted_item)

    return data


def load_contract_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 Contract QA 数据（合同条款问答）
    询问合同中是否包含特定条款，或条款的具体内容。
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    
    for filepath in paths:
        if filepath.suffix != ".jsonl":
            continue
        
        with jsonlines.open(filepath, "r") as reader:
            for idx, item in enumerate(reader):
                text = (item.get("text") or "").strip()
                question = (item.get("question") or "").strip()
                answer = (item.get("answer") or "").strip()
                
                if not text or not question:
                    continue
                
                formatted_item = {
                    "_id": item.get("id", f"contract_qa_{idx}"),
                    "question": question,
                    "answer": answer,
                    "context": [("contract", [text])],
                    "context_type": "contract",
                    "task_type": "clause_extraction",
                    "text_length": len(text),
                }
                data.append(formatted_item)
    
    return data


def load_consumer_contracts_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 Consumer Contracts QA 数据（服务条款/用户协议问答）
    针对 ToS 或用户协议的问答，通常涉及用户权利和义务。
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    
    for filepath in paths:
        if filepath.suffix != ".jsonl":
            continue
        
        with jsonlines.open(filepath, "r") as reader:
            for idx, item in enumerate(reader):
                text = (item.get("text") or "").strip()
                question = (item.get("question") or "").strip()
                answer = (item.get("answer") or "").strip()
                
                if not text or not question:
                    continue
                
                formatted_item = {
                    "_id": item.get("id", f"consumer_qa_{idx}"),
                    "question": question,
                    "answer": answer,
                    "context": [("terms_of_service", [text])],
                    "context_type": "tos",
                    "task_type": "user_agreement_qa",
                    "text_length": len(text),
                }
                data.append(formatted_item)
    
    return data


def load_privacy_policy_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 Privacy Policy QA 数据（隐私政策问答）
    用户针对隐私政策提问，涉及数据收集、使用和共享。
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    
    for filepath in paths:
        if filepath.suffix != ".jsonl":
            continue
        
        with jsonlines.open(filepath, "r") as reader:
            for idx, item in enumerate(reader):
                text = (item.get("text") or "").strip()
                question = (item.get("question") or "").strip()
                answer = (item.get("answer") or "").strip()
                
                if not text or not question:
                    continue
                
                formatted_item = {
                    "_id": item.get("id", f"privacy_qa_{idx}"),
                    "question": question,
                    "answer": answer,
                    "context": [("privacy_policy", [text])],
                    "context_type": "privacy_policy",
                    "task_type": "data_handling_qa",
                    "text_length": len(text),
                }
                data.append(formatted_item)
    
    return data


def load_rule_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 Rule QA 数据（规则推导问答）
    基于明确给出的规则进行逻辑推导。
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    
    for filepath in paths:
        if filepath.suffix != ".jsonl":
            continue
        
        with jsonlines.open(filepath, "r") as reader:
            for idx, item in enumerate(reader):
                text = (item.get("text") or "").strip()
                question = (item.get("question") or "").strip()
                answer = (item.get("answer") or "").strip()
                
                if not text or not question:
                    continue
                
                formatted_item = {
                    "_id": item.get("id", f"rule_qa_{idx}"),
                    "question": question,
                    "answer": answer,
                    "context": [("rule_definition", [text])],
                    "context_type": "rules",
                    "task_type": "logical_reasoning",
                    "text_length": len(text),
                }
                data.append(formatted_item)
    
    return data


def load_legalbench_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 LegalBench QA 类数据（contract_qa, consumer_contracts_qa, privacy_policy_qa, rule_qa）

    支持的schema:
    - text: 上下文文本（合同、隐私政策等）
    - question: 问题
    - answer: 答案
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    paths = []
    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            continue

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": (item.get("question") or "").strip(),
                    "answer": (item.get("answer") or "").strip(),
                    "context": [(path.stem, [item.get("text", "")])],
                    "context_type": "legal_document",
                }
                data.append(formatted_item)

    return data


def load_legalbench_entailment_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 LegalBench Entailment 类数据（privacy_policy_entailment, sara_entailment）

    支持的schema:
    - text: 上下文文本（法律条文、隐私政策等）
    - hypothesis: 要判断的陈述句
    - answer: Entails / Contradicts / Neutral
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    paths = []
    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            continue

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": (item.get("hypothesis") or "").strip(),
                    "answer": (item.get("answer") or "").strip(),
                    "context": [(path.stem, [item.get("text", "")])],
                    "context_type": "legal_entailment",
                }
                data.append(formatted_item)

    return data


def load_legalbench_insurance_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 LegalBench 保险相关数据（insurance_policy_interpretation）

    支持的schema:
    - text: 保险条款原文
    - question: 场景问题
    - answer: 是否赔付
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            continue

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": (item.get("question") or "").strip(),
                    "answer": (item.get("answer") or "").strip(),
                    "context": [("insurance_policy", [item.get("text", "")])],
                    "context_type": "legal_insurance",
                }
                data.append(formatted_item)

    return data


def load_legalbench_corporate_lobbying_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 LegalBench 企业游说数据（corporate_lobbying）

    支持的schema:
    - text: 法案标题或摘要
    - company_name: 公司名称
    - answer: Yes / No（是否相关）
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            continue

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                company_name = item.get("company_name", "Unknown Company")
                question = f"Is this bill relevant to {company_name}?"
                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": question,
                    "answer": (item.get("answer") or "").strip(),
                    "company_name": company_name,
                    "context": [("bill", [item.get("text", "")])],
                    "context_type": "legal_corporate_lobbying",
                }
                data.append(formatted_item)

    return data


def load_legalbench_scalr_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 LegalBench 案例法推理数据（scalr）

    支持的schema:
    - text: 案件事实和法律原则
    - question: 问题
    - options: 选项列表
    - answer: 正确答案
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

    if path.is_dir():
        paths = list(path.glob("*.jsonl"))
    else:
        paths = [path]

    for path in paths:
        if path.suffix != ".jsonl":
            continue

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                options = item.get("options", []) or []
                options_str = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
                question = (item.get("question") or "").strip()
                if options_str:
                    question = f"{question}\n\nOptions:\n{options_str}"

                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": question,
                    "answer": (item.get("answer") or "").strip(),
                    "context": [("supreme_court_case", [item.get("text", "")])],
                    "context_type": "legal_case",
                }
                data.append(formatted_item)

    return data


def load_data(file_path: str, task: str = "hotpotqa") -> List[Dict[str, Any]]:
    """
    通用数据加载接口，支持不同任务的数据集
    """
    if task == "hotpotqa":
        return load_hotpotqa_data(file_path)
    elif task == "musique":
        return load_musique_data(file_path)
    elif task == "multihop":
        return load_multihop_data(file_path)
    elif task == "qa/contract":
        return load_contract_qa_data(file_path)
    elif task == "qa/consumer":
        return load_consumer_contracts_qa_data(file_path)
    elif task == "qa/privacy":
        return load_privacy_policy_qa_data(file_path)
    elif task == "qa/rule":
        return load_rule_qa_data(file_path)
    elif task.startswith("legalbench/qa"):
        return load_legalbench_qa_data(file_path)
    elif task.startswith("legalbench/entailment"):
        return load_legalbench_entailment_data(file_path)
    elif task.startswith("legalbench/insurance"):
        return load_legalbench_insurance_data(file_path)
    elif task.startswith("legalbench/corporate_lobbying"):
        return load_legalbench_corporate_lobbying_data(file_path)
    elif task.startswith("legalbench/scalr"):
        return load_legalbench_scalr_data(file_path)
    else:
        raise ValueError(f"Unsupported task: {task}")


def run_hypergraph_build(
    data_path: str,
    batch_size: int = 5,
    task: str = "hotpotqa",
    output_dir: str = "data/hypergraph"
):
    """
    将数据转为QueryInstance并构建超图
    
    Args:
        data_path: 数据文件路径
        batch_size: 批处理大小
        task: 任务类型
        output_dir: 输出目录
    """
    print(f"Loading data from {data_path}...")
    data: List[Dict[str, Any]] = load_data(data_path, task)
    
    print(f"Loaded {len(data)} samples")
    print(f"Building hypergraph with batch_size={batch_size}...")
    
    
    instance_dir = f"{output_dir}/{task}"
    import os
    os.makedirs(instance_dir, exist_ok=True)
    # 按批次处理
    for batch_start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        assert batch_start + batch_size <= len(data) 
        batch = data[batch_start:(batch_start + batch_size)]
        
        # build data as `QueryInstance` by task
        if task == "hotpotqa":
            query_instances: list[QueryInstance] = []
            for item in batch:
                supporting_facts = item.get('supporting_facts', {})
                ground_truths = []
                titles_set = set(supporting_facts.get('title', []))
                for title, sentences in item['context']:
                    if title in titles_set:
                        has_contradiction = True
                        sent_ids = supporting_facts.get('sent_id', [])
                        evidence_sentences = [
                            sentences[i] for i in sent_ids 
                            if i < len(sentences)
                        ]
                        evidence = "\n".join(evidence_sentences)
                        ground_truths.append((has_contradiction, evidence))
                    else:
                        ground_truths.append((False, ""))
                query_instance = QueryInstance(
                    query=item['question'],
                    data=[
                        f"{title}.\n" + "\n".join(sentences)
                        for title, sentences in item['context']
                    ],
                    fixed_data=[],
                    answers=item['answer'],
                    ground_truth=ground_truths
                )
                query_instances.append(query_instance)
        elif task == "musique":
            query_instances: list[QueryInstance] = []
            for item in batch:
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

                raw_decomposition = item.get("question_decomposition", []) or []
                if isinstance(raw_decomposition, list) and raw_decomposition:
                    if all(isinstance(d, dict) and "id" in d for d in raw_decomposition):
                        sorted_decomposition = sorted(raw_decomposition, key=lambda d: d.get("id"))
                    else:
                        sorted_decomposition = raw_decomposition
                    query_decomposition = [
                        (d.get("question") or "").strip() for d in sorted_decomposition
                        if isinstance(d, dict)
                    ]
                else:
                    query_decomposition = None

                query_instance = QueryInstance(
                    query=item["question"],
                    data=[
                        f"{title}.\n" + "\n".join(sentences)
                        for title, sentences in item["context"]
                    ],
                    fixed_data=[],
                    answers=answers,
                    ground_truth=ground_truths,
                    query_decomposition=query_decomposition
                )
                query_instances.append(query_instance)
        elif task == "multihop":
            query_instances: list[QueryInstance] = []
            for item in batch:
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

                query_instance = QueryInstance(
                    query=item["question"],
                    data=[
                        f"{title}.\n" + "\n".join(sentences)
                        for title, sentences in item["context"]
                    ],
                    fixed_data=[],
                    answers=answers,
                    ground_truth=ground_truths
                )
                query_instances.append(query_instance)
        elif task.startswith("legalbench/"):
            query_instances: list[QueryInstance] = []
            for item in batch:
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

                query_instance = QueryInstance(
                    query=item["question"],
                    data=[
                        f"{title}.\n" + "\n".join(sentences)
                        for title, sentences in item["context"]
                    ],
                    fixed_data=[],
                    answers=answers,
                    ground_truth=ground_truths
                )
                query_instances.append(query_instance)
        elif task in ("qa/contract", "qa/consumer", "qa/privacy", "qa/rule"):
            query_instances: list[QueryInstance] = []
            for item in batch:
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
                
                query_instance = QueryInstance(
                    query=item["question"],
                    data=[
                        f"{title}.\n" + "\n".join(sentences)
                        for title, sentences in item["context"]
                    ],
                    fixed_data=[],
                    answers=answers,
                    ground_truth=ground_truths
                )
                query_instances.append(query_instance)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        # Clean the QueryInstance for spacy processing
        for qi in query_instances:
            qi.query = clean_text_for_spacy(qi.query)
            qi.data = [clean_text_for_spacy(d) for d in qi.data]

        for idx, qi in enumerate(query_instances):
            q, d_list = test_build_hypergraph_for_query_instance(qi)
            # save an instance at output_dir/task/<instance_id>/ as a text file
            instance_id = idx + batch_start
            # the file is instance_dir/instance_id.txt
            instance_path = f"{instance_dir}/{instance_id}.txt"
            with open(instance_path, "w", encoding="utf-8") as f:
                # write query
                f.write(f"Query:\n{qi.query}\n\n")
                # write query hypergraph
                f.write(f"Query Hypergraph:\n")
                f.write(str(q))
                f.write("\n\n")
                
                # Write data hypergraphs
                for d_idx, d_hg in enumerate(d_list):
                    # First write data document, then hypergraph
                    f.write(f"Data Document #{d_idx}:\n{qi.data[d_idx]}\n\n")
                    f.write(f"Data Hypergraph #{d_idx}:\n")
                    f.write(str(d_hg))
                    f.write("\n\n")
            


def main():
    parser = argparse.ArgumentParser(description="Build hypergraph from datasets")
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to HotpotQA data file (json or jsonl)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5,
        help='Batch size for LLM inference'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='hotpotqa',
        help='Task type (default: hotpotqa)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Output directory for hypergraphs'
    )

    args = parser.parse_args()
    # 构建超图
    run_hypergraph_build(
        data_path=args.data_path,
        batch_size=args.batch_size,
        task=args.task,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
