import json
import re
import jsonlines
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path
import logging,time
from hyper_simulation.question_answer.vmdit.metrics import (
    exact_match_score, 
    metric_max_over_ground_truths,
    qa_f1_score,
    match
)
from hyper_simulation.query_instance import QueryInstance
from hyper_simulation.utils.log import current_task, current_query_id
from hyper_simulation.utils.log import getLogger
from hyper_simulation.llm.prompt.hotpot_qa import HOTPOT_QA_BASE
from hyper_simulation.llm.prompt.musique import MUSIQUE_QA_BASE
from hyper_simulation.llm.prompt.multihop import MULTIHOP_QA_BASE
from hyper_simulation.llm.prompt.legalbench_qa import LEGALBENCH_QA_BASE
from hyper_simulation.llm.prompt.legalbench_qa_detailed import QA_CONTRACT_BASE, QA_CONSUMER_BASE, QA_PRIVACY_BASE, QA_RULE_BASE
from hyper_simulation.llm.prompt.legalbench_sara_entailment import LEGALBENCH_SARA_ENTAILMENT_BASE
from hyper_simulation.llm.prompt.legalbench_privacy_policy_entailment import LEGALBENCH_PRIVACY_POLICY_ENTAILMENT_BASE
from hyper_simulation.llm.prompt.legalbench_insurance import LEGALBENCH_INSURANCE_BASE
from hyper_simulation.llm.prompt.legalbench_corporate_lobbying import LEGALBENCH_CORPORATE_LOBBYING_BASE
from hyper_simulation.llm.prompt.legalbench_scalr import LEGALBENCH_SCALR_BASE
from hyper_simulation.llm.prompt.arc import ARC_BASE
from hyper_simulation.llm.prompt.vmdit import PROMPT_DICT

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

def load_musique_data(file_path: str, use_supporting_only: bool = True) -> List[Dict[str, Any]]:
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
                    is_supporting = bool(p.get("is_supporting", False))
                    if use_supporting_only and not is_supporting:
                        continue  # 跳过非支持段落
                    title = (p.get("title") or "").strip()
                    paragraph_text = (p.get("paragraph_text") or "").strip()

                    if title or paragraph_text:
                        context.append((title, [paragraph_text] if paragraph_text else []))
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
                    # ✅ 关键修复: 兼容 "text" 或 "fact" 字段（你的数据用 "fact"）
                    text = (evidence.get("text") or evidence.get("fact") or "").strip()
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

def load_arc_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 ARC 数据集（选择题）"""
    data: List[Dict[str, Any]] = []
    path = Path(file_path)

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
                choices = item.get("choices", {})
                options_text = choices.get("text", [])
                options_label = choices.get("label", [])
                
                # 格式化选项
                options_str = "\n".join([
                    f"{label}) {text}" 
                    for label, text in zip(options_label, options_text)
                ]) if options_label and options_text else ""
                
                question_with_options = f"{item.get('question', '').strip()}\n\nOptions:\n{options_str}"
                
                answer_label = item.get("answerKey", "")  # ✅ 保留标签
                answer_text = ""
                if answer_label and options_label and options_text:
                    try:
                        idx = options_label.index(answer_label)
                        answer_text = options_text[idx]
                    except (ValueError, IndexError):
                        answer_text = answer_label
                
                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": question_with_options,
                    "answer": answer_text,           # 答案文本（用于参考）
                    "answer_label": answer_label,    # ✅ 答案标签（用于评估）
                    "options": options_text,
                    "option_labels": options_label,
                    "context": [],
                    "supporting_flags": [],
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
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    for filepath in paths:
        if filepath.suffix != ".jsonl":
            continue
        with jsonlines.open(filepath, "r") as reader:
            for idx, item in enumerate(reader):
                text = (item.get("text") or item.get("contract") or "").strip()
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
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    for filepath in paths:
        if filepath.suffix != ".jsonl":
            continue
        with jsonlines.open(filepath, "r") as reader:
            for idx, item in enumerate(reader):
                question = (item.get("question") or "").strip()
                text = (item.get("text") or "").strip()
                answer = (item.get("answer") or "").strip()
                
                if not question and text and text.endswith("?"):
                    question = text
                    text = ""  # 清空 text，避免作为上下文重复
                
                if not question:
                    continue
                
                formatted_item = {
                    "_id": item.get("id", f"rule_qa_{idx}"),
                    "question": question,
                    "answer": answer,
                    "context": [("rule_definition", [text])] if text else [],
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

def load_sara_entailment_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 SARA Entailment 数据
    格式: statute + description + question + answer(Entailment/Contradiction/Neutral)
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    
    for path in paths:
        if path.suffix != ".jsonl":
            continue
        with jsonlines.open(path, "r") as reader:
            for item in reader:
                statute = (item.get("statute") or "").strip()
                description = (item.get("description") or "").strip()
                hypothesis = (item.get("question") or item.get("hypothesis") or "").strip()
                
                context_parts = [p for p in [statute, description] if p]
                context_text = "\n\n".join(context_parts) if context_parts else ""
                
                # 统一标签: Entailment -> Entails
                answer_raw = (item.get("answer") or "").strip()
                answer_map = {
                    "Entailment": "Entails",
                    "Entails": "Entails",
                    "Contradiction": "Contradicts",
                    "Contradicts": "Contradicts",
                    "Neutral": "Neutral",
                }
                answer = answer_map.get(answer_raw, answer_raw)
                
                formatted_item = {
                    "_id": item.get("id", item.get("case id", "")),
                    "question": hypothesis,
                    "answer": answer,
                    "context": [("legal_text", [context_text])],
                    "context_type": "legal_sara_entailment"
                }
                data.append(formatted_item)
    return data

def load_privacy_policy_entailment_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 Privacy Policy Entailment 数据
    格式: text + description + answer(Correct/Incorrect)
    """
    data: List[Dict[str, Any]] = []
    path = Path(file_path)
    paths = [path] if not path.is_dir() else list(path.glob("*.jsonl"))
    
    for path in paths:
        if path.suffix != ".jsonl":
            continue
        with jsonlines.open(path, "r") as reader:
            for item in reader:
                policy_text = (item.get("text") or "").strip()
                description = (item.get("description") or "").strip()
                
                # ✅ 正确解析: text 是政策原文，description 是要判断的陈述
                context_text = policy_text
                
                # ✅ 答案标签: Correct/Incorrect (保持原样)
                answer = (item.get("answer") or "").strip()
                
                formatted_item = {
                    "_id": item.get("id", f"ppe_{item.get('index', '')}"),
                    "question": description,  # ✅ description 作为 hypothesis
                    "answer": answer,         # ✅ Correct/Incorrect
                    "context": [("privacy_policy", [context_text])],
                    "context_type": "privacy_policy_entailment",
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
    """加载 LegalBench 案例法推理数据（scalr）"""
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
                # ✅ 修复 1: 解析 choice_0, choice_1, ... 格式
                options = []
                i = 0
                while f"choice_{i}" in item:
                    opt_text = item.get(f"choice_{i}", "")
                    if opt_text:
                        options.append(opt_text)
                    i += 1
                
                # 格式化选项: "(A) text\n(B) text\n..."
                options_str = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
                question = (item.get("question") or "").strip()
                if options_str:
                    question = f"{question}\n\nOptions:\n{options_str}"
                
                # ✅ 修复 2: answer 是整数索引，转换为选项标签
                answer_idx = item.get("answer")
                if answer_idx is None:
                    answer_label = ""
                elif isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
                    answer_label = chr(65 + answer_idx)  # 0->A, 1->B, 2->C...
                else:
                    answer_label = str(answer_idx).strip()
                
                formatted_item = {
                    "_id": item.get("id", item.get("index", "")),
                    "question": question,
                    "answer": answer_label,        # 用于评估的标签 (A/B/C...)
                    "answer_index": answer_idx,    # 保留原始索引
                    "options": options,
                    "context": [("supreme_court_case", [item.get("text", "")])],
                    "context_type": "legal_case",
                }
                data.append(formatted_item)
    return data

def load_data(file_path: str, task: str = "hotpotqa", use_supporting_only: bool = False) -> List[Dict[str, Any]]:
    """通用数据加载接口"""
    
    # ========== 顶层统一入口：加载所有 LegalBench 子任务 ==========
    if task == "legalbench":  # ✅ 新增：直接加载所有子任务
        # 定义子任务配置：(loader 函数，相对子路径，context_type)
        legalbench_tasks = [
            (load_contract_qa_data, "QA/contract_qa.jsonl", "contract"),
            (load_consumer_contracts_qa_data, "QA/consumer_contracts_qa.jsonl", "tos"),
            (load_privacy_policy_qa_data, "QA/privacy_policy_qa.jsonl", "privacy_policy"),
            (load_rule_qa_data, "QA/rule_qa.jsonl", "rules"),
            (load_privacy_policy_entailment_data, "privacy_policy_entailment.jsonl", "legal_privacy_policy_entailment"),
            (load_sara_entailment_data, "sara_entailment.jsonl", "legal_sora_entailment"),
            (load_legalbench_insurance_data, "insurance_policy_interpretation.jsonl", "legal_insurance"),
            (load_legalbench_corporate_lobbying_data, "corporate_lobbying.jsonl", "legal_corporate_lobbying"),
            (load_legalbench_scalr_data, "scalr.jsonl", "legal_case"),
        ]
        
        # 合并所有子任务数据
        all_items = []
        base_path = Path(file_path)
        for loader, subpath, ctx_type in legalbench_tasks:
            task_path = base_path / subpath
            if task_path.exists():
                try:
                    sub_data = loader(str(task_path))
                    # 标记 context_type 供 prompt 选择
                    for item in sub_data:
                        item["context_type"] = ctx_type
                    all_items.extend(sub_data)
                    print(f"  ✓ Loaded {len(sub_data)} from {subpath}")
                except Exception as e:
                    print(f"  ⚠️ Failed to load {subpath}: {e}")
        
        if not all_items:
            raise ValueError(f"No LegalBench data found at {file_path}")
        
        print(f"✓ Total LegalBench samples: {len(all_items)}")
        
        # ✅ 修复关键：直接返回原始数据列表，不要在 load_data 里构建 QueryInstance
        # run_rag_evaluation 会根据 task 类型再次构建 QueryInstance
        return all_items
    
    # ========== 原子任务接口（向后兼容）==========
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
    elif task.startswith("legalbench/sara_entailment"):
        return load_sara_entailment_data(file_path)
    elif task.startswith("legalbench/privacy_policy_entailment"):
        return load_privacy_policy_entailment_data(file_path)
    elif task.startswith("legalbench/insurance"):
        return load_legalbench_insurance_data(file_path)
    elif task.startswith("legalbench/corporate_lobbying"):
        return load_legalbench_corporate_lobbying_data(file_path)
    elif task.startswith("legalbench/scalr"):
        return load_legalbench_scalr_data(file_path)
    
    # ========== 其他任务 ==========
    elif task == "hotpotqa":
        return load_hotpotqa_data(file_path)
    elif task == "musique":
        return load_musique_data(file_path, use_supporting_only)
    elif task == "multihop":
        return load_multihop_data(file_path)
    elif task == "ARC":
        return load_arc_data(file_path)
    
    else:
        raise ValueError(f"Unsupported task: {task}")

def build_prompt(question: str, context_text: str, task: str = "hotpotqa", context_type: str = None) -> str:
    """
    构建用于LLM的prompt，根据不同任务选择相应的模板
    
    Args:
        question: 问题文本
        context_text: 格式化后的context文本
        task: 任务类型 (hotpotqa, musique, multihop, qa/*, legalbench/*)
    
    Returns:
        完整的prompt
    """
    if task == "hotpotqa":
        prompt = HOTPOT_QA_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "musique":
        prompt = MUSIQUE_QA_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "multihop":
        prompt = MULTIHOP_QA_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "legalbench":
        if context_type == "contract":
            prompt = QA_CONTRACT_BASE.format(context_text=context_text, question=question)
        elif context_type == "tos":
            prompt = QA_CONSUMER_BASE.format(context_text=context_text, question=question)
        elif context_type == "privacy_policy":
            prompt = QA_PRIVACY_BASE.format(context_text=context_text, question=question)
        elif context_type == "rules":
            prompt = QA_RULE_BASE.format(context_text=context_text, question=question)
        elif context_type == "legal_sara_entailment":
            prompt = LEGALBENCH_SARA_ENTAILMENT_BASE.format(context_text=context_text, question=question)
        elif context_type == "legal_privacy_policy_entailment":
            prompt = LEGALBENCH_PRIVACY_POLICY_ENTAILMENT_BASE.format(context_text=context_text, question=question)
        elif context_type == "legal_insurance":
            prompt = LEGALBENCH_INSURANCE_BASE.format(context_text=context_text, question=question)
        elif context_type == "legal_corporate_lobbying":
            prompt = LEGALBENCH_CORPORATE_LOBBYING_BASE.format(context_text=context_text, question=question)
        elif context_type == "legal_case":
            prompt = LEGALBENCH_SCALR_BASE.format(context_text=context_text, question=question)
        else:
            # 兜底：用通用 legalbench prompt
            prompt = LEGALBENCH_QA_BASE.format(context_text=context_text, question=question)
    elif task == "qa/contract":
        prompt = QA_CONTRACT_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "qa/consumer":
        prompt = QA_CONSUMER_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "qa/privacy":
        prompt = QA_PRIVACY_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "qa/rule":
        prompt = QA_RULE_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task.startswith("legalbench/qa"):
        prompt = LEGALBENCH_QA_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task.startswith("legalbench/sara_entailment"):
        prompt = LEGALBENCH_SARA_ENTAILMENT_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task.startswith("legalbench/privacy_policy_entailment"):
        prompt = LEGALBENCH_PRIVACY_POLICY_ENTAILMENT_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task.startswith("legalbench/insurance"):
        prompt = LEGALBENCH_INSURANCE_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task.startswith("legalbench/corporate_lobbying"):
        prompt = LEGALBENCH_CORPORATE_LOBBYING_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task.startswith("legalbench/scalr"):
        prompt = LEGALBENCH_SCALR_BASE.format(
            context_text=context_text,
            question=question
        )
    elif task == "ARC":
        prompt = ARC_BASE.format(
            question=question
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    return prompt

def evaluate_answer(prediction: str, ground_truth: list | str) -> Dict[str, float]:
    """
    评估生成的答案
    
    Args:
        prediction: 模型生成的答案
        ground_truth: 正确答案
    
    Returns:
        包含多个评估指标的字典
    """
    # 处理ground_truth可能是列表的情况
    if isinstance(ground_truth, list):
        ground_truths = ground_truth
    else:
        ground_truths = [ground_truth]
    
    # 计算Exact Match
    em_score = metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths
    )
    
    # 计算F1 score
    f1_score = max([qa_f1_score(prediction, gt) for gt in ground_truths])
    
    # 计算Match (部分匹配)
    match_score = match(prediction, ground_truths)
    
    return {
        "exact_match": em_score,
        "f1": f1_score,
        "match": match_score
    }

def postprocess_answer(answer: str) -> tuple[str, str, bool]:
    """
    从 LLM 输出中提取最终答案（增强版）
    
    Returns:
        (processed_answer, parse_status, is_fallback)
        - parse_status: "parsed_final_answer" | "parsed_last_line" | "fallback_truncated" | "empty_input"
        - is_fallback: 是否使用了兜底策略
    """
    import re
    import logging
    logger = logging.getLogger(__name__)
    
    # ========== 1. 空答案保护 ==========
    if not answer:
        return "unanswerable", "empty_input", True
    
    # ========== 2. 基础清理 ==========
    answer = answer.replace("</s>", "").replace("</think>", "").strip()
    
    # ========== 3. 多次匹配 ### Final Answer:（关键修复）==========
    final_answer_pattern = r"###\s*Final\s*Answer:\s*(.+?)(?:\n|$)"
    extracted = answer
    parse_status = "fallback_truncated"
    match_count = 0
    
    # 循环匹配，直到没有更多 "### Final Answer:" 前缀
    while True:
        match = re.search(final_answer_pattern, extracted, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            match_count += 1
            parse_status = "parsed_final_answer"
        else:
            break
    
    # 如果至少匹配到一次，返回结果
    if match_count > 0:
        cleaned = extracted.strip(" .,;:!?\"'")
        if cleaned:
            return cleaned, parse_status, False
    
    # ========== 4. 次选：最后一行短答案 ==========
    lines = answer.strip().split('\n')
    exclude_keywords = ['step', 'reason', 'explain', 'note', 'context', 
                       'paragraph', 'think', 'analysis', 'conclusion']
    
    for line in reversed(lines):
        line = line.strip()
        if (line and 
            len(line) < 100 and 
            not any(k in line.lower() for k in exclude_keywords) and
            not line.startswith('#')):
            cleaned = line.strip(" .,;:!?\"'")
            if cleaned:
                return cleaned, "parsed_last_line", False
    
    # ========== 5. 兜底 + 警告日志 ==========
    logger.warning(f"⚠️ Could not parse answer (match_count={match_count}), using fallback. Output preview: {answer[:100]}...")
    
    cleaned = answer.strip(" .,;:!?\"'")[:200]
    return cleaned if cleaned else "unanswerable", "fallback_truncated", True

def run_rag_evaluation(
    data_path: str,
    model_name: str = "qwen3.5:9b",
    output_path: str = "",
    batch_size: int = 5,
    temperature: float = 0.2,
    task: str = "hotpotqa",
    method: str = "vanilla",
    build: bool = True,
    rebuild: bool = False,
    using_support_only: bool = False,
    save_interval: int = 500,
    save_prompts_only: bool = False,
    load_prompts: str = None
):
    """
    运行 RAG 评估任务 (支持断点续传和增量保存)
    注意：使用 'question' 作为唯一标识符进行去重和续传。
    """
    print(f"Loading data from {data_path}...")
    data: List[Dict[str, Any]] = load_data(data_path, task, using_support_only)
    print(f"Loaded {len(data)} samples")
    
    # 🔹 1. 加载已有结果，获取已处理的问题集合
    processed_questions = set()
    existing_results = []
    
    if output_path:
        out_file = Path(output_path) / f"{task}.json"
        if out_file.exists():
            try:
                with open(out_file, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                existing_results = old_data.get("results", [])
                
                # 提取已处理的问题文本
                for r in existing_results:
                    q = r.get("question")
                    if q:
                        processed_questions.add(q)
                
                print(f"✅ 发现已有结果文件，已加载 {len(processed_questions)} 条已完成记录。将从断点处继续。")
            except Exception as e:
                print(f"⚠️ 读取已有结果文件失败：{e}。将重新开始。")
        else:
            print("ℹ️ 未找到已有结果文件，将从头开始运行。")
            
    # 🔹 2. 加载已有 Prompts（如果指定）
    prompts_data = []
    if load_prompts:
        print(f"📂 从 {load_prompts} 加载 Prompts...")
        if load_prompts.endswith('.jsonl'):
            with jsonlines.open(load_prompts, 'r') as reader:
                prompts_data = list(reader)
        else:
            with open(load_prompts, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
        print(f"✅ 已加载 {len(prompts_data)} 条 Prompts")
        # 过滤已完成的
        prompts_data = [p for p in prompts_data if p.get('question') not in processed_questions]
        print(f"📝 剩余 {len(prompts_data)} 条待处理 Prompts")
    
    # 如果有加载的 prompts，直接使用
    if load_prompts:
        items_to_process = prompts_data
    else:
        items_to_process = data
    
    # 如果所有任务都已完成
    if len(items_to_process) - len(processed_questions) <= 0 and not load_prompts:
        # 当使用 load_prompts 时，items_to_process 已经是过滤后的了，如果为空则直接结束
        pass
    if not items_to_process:
        print("✨ 所有任务已完成！无需重新运行。")
        # 重新计算一下最终指标并返回
        all_metrics_tmp = {"exact_match": [], "f1": [], "match": []}
        for r in existing_results:
            m = r.get("metrics", {})
            for k in all_metrics_tmp:
                if k in m:
                    all_metrics_tmp[k].append(m[k])
        avg_metrics_tmp = {k: sum(v)/len(v) if v else 0 for k, v in all_metrics_tmp.items()}
        return existing_results, avg_metrics_tmp

    # 🔹 3. 初始化 LLM（如果需要）
    if not load_prompts and not save_prompts_only:
        print(f"Initializing LLM: {model_name}")
        if build or method != "hyper_simulation":
            from langchain_ollama import ChatOllama
            from hyper_simulation.llm.chat_completion import get_generate
            model = ChatOllama(model=model_name, temperature=temperature, top_p=0.95, reasoning=False, timeout=300)
    elif load_prompts:
        print(f"Initializing LLM for pre-loaded prompts: {model_name}")
        from langchain_ollama import ChatOllama
        from hyper_simulation.llm.chat_completion import get_generate
        model = ChatOllama(model=model_name, temperature=temperature, top_p=0.95, reasoning=False, timeout=300)
    
    # 🔹 4. 初始化结果容器和指标
    results = list(existing_results) 
    all_metrics = {"exact_match": [], "f1": [], "match": []}
    
    # 恢复已有结果的指标统计
    for r in existing_results:
        m = r.get("metrics", {})
        for k in all_metrics:
            if k in m:
                all_metrics[k].append(m[k])

    print(f"Starting evaluation with batch_size={batch_size}... (Skip {len(processed_questions)} done)")
    current_task.set(task)
    
    # 🔹 5. Prompt 保存路径
    prompt_save_path = None
    if save_prompts_only and output_path:
        prompt_save_path = Path(output_path) / f"{method}" / f"{task}.jsonl"
        prompt_save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"💾 Prompts 将保存到：{prompt_save_path}")
    
    config = {
        "model_name": model_name,
        "data_path": data_path,
        "batch_size": batch_size,
        "temperature": temperature,
        "method": method,
        "task": task,
        "total_samples_original": len(data)
    }

    new_results_buffer = [] 
    prompts_buffer = []  # 用于批量保存 prompts

    # 进度条初始化：只显示剩余任务数量
    pbar = tqdm(total=len(items_to_process), desc="Processing", position=0, leave=True, initial=len(processed_questions) if not load_prompts else 0)

    for batch_start in range(0, len(items_to_process), batch_size):
        batch = items_to_process[batch_start:(batch_start + batch_size)]
        filtered_batch = []
        for item in batch:
            q_text = item.get('question', '').strip() if isinstance(item, dict) else getattr(item, 'query', '')
            if q_text in processed_questions and not load_prompts:
                # 当 load_prompts 时，之前已经过滤过了
                continue
            filtered_batch.append(item)
        
        if not filtered_batch:
            pbar.update(len(batch))
            continue

        # 🔹 6. 构建 QueryInstance (如果没有加载 prompts)
        if not load_prompts:
            query_instances = []
            for item in filtered_batch:
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
                    query_instance = QueryInstance(
                        query=item['question'],
                        data=[f"{title}.\n" + "\n".join(sentences) for title, sentences in item['context']],
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
                    raw_decomposition = item.get("question_decomposition", []) or []
                    query_decomposition = None
                    if isinstance(raw_decomposition, list) and raw_decomposition:
                        if all(isinstance(d, dict) and "id" in d for d in raw_decomposition):
                            sorted_decomposition = sorted(raw_decomposition, key=lambda d: d.get("id"))
                        else:
                            sorted_decomposition = raw_decomposition
                        query_decomposition = [(d.get("question") or "").strip() for d in sorted_decomposition if isinstance(d, dict)]
                    
                    query_instance = QueryInstance(
                        query=item["question"],
                        data=[f"{title}.\n" + "\n".join(sentences) for title, sentences in item["context"]],
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
                    query_instance = QueryInstance(
                        query=item["question"],
                        data=[f"{title}.\n" + "\n".join(sentences) for title, sentences in item["context"]],
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
                    query_instance = QueryInstance(
                        query=item["question"],
                        data=[f"{title}.\n" + "\n".join(sentences) for title, sentences in item["context"]],
                        fixed_data=[],
                        answers=answers,
                        ground_truth=ground_truths,
                        context_type=context_type,
                    )
                    query_instances.append(query_instance)
                    continue # Skip the generic append below
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
                    query_instance = QueryInstance(
                        query=item["question"],
                        data=[f"{title}.\n" + "\n".join(sentences) for title, sentences in item["context"]],
                        fixed_data=[],
                        answers=answers,
                        ground_truth=ground_truths
                    )
                elif task == "ARC":
                    query_instance = QueryInstance(
                        query=item['question'],
                        data=[],
                        fixed_data=[],
                        answers=[item['answer_label']],
                        ground_truth=[]
                    )
                else:
                    raise ValueError(f"Unsupported task: {task}")
                
                query_instances.append(query_instance)
    
            if not query_instances:
                pbar.update(len(batch))
                continue
    
            # Method 处理
            method_start = time.time()
            if method == "vanilla":
                fixed_query_instances = query_instances
            elif method == "contradoc":
                from hyper_simulation.baselines.contradoc import query_fixup
                fixed_query_instances = [query_fixup(qi, model=model) for qi in query_instances]
            elif method == "sparsecl":
                from hyper_simulation.baselines.sparseCL import query_fixup
                fixed_query_instances = [query_fixup(qi, alpha=1.5) for qi in query_instances]
            elif method == "sentli":
                from hyper_simulation.baselines.sentLI import query_fixup
                fixed_query_instances = [ query_fixup(qi) for qi in query_instances]
            elif method == "hyper_simulation":
                if not build:
                    from hyper_simulation.component.build_hypergraph import build_hypergraph_batch
                    build_hypergraph_batch(query_instances, dataset_name=task, force_rebuild=rebuild)
                    print("Hypergraph built. Please re-run with --build to evaluate.")
                    pbar.update(len(batch))
                    continue
                else:
                    from hyper_simulation.component.consistent import query_fixup
                    fixed_query_instances = [query_fixup(qi, task) for qi in query_instances]
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            method_time = time.time() - method_start
            
            # 准备 Prompts
            prompts = []
            for item in fixed_query_instances:
                context_text = "\n\n".join(item.fixed_data if item.fixed_data else item.data)
                context_type = getattr(item, 'context_type', None)
                score_guide = ""
                if method == "sparsecl":
                    score_guide = ( "Note: Documents prefixed with [SparseCL: X.XX] include contradiction scores.\nHigher scores indicate higher potential contradiction with the question. \nUse this as a reference when evaluating evidence reliability.\n\n")
                if method == "sentli":
                    score_guide = ( "Note: Documents are prefixed with [SENTLI: label] indicating logical relationship.\n[SENTLI: e] = Entailment (Supported/Correct)\n[SENTLI: c] = Contradiction (Incorrect)\n[SENTLI: n] = Neutral (Not mentioned)\nUse this label to help determine the final answer.\n\n")
                context_text = score_guide + context_text
                prompt = build_prompt(item.query, context_text, task=task, context_type=context_type)
                prompts.append(prompt)
        else:
            # 🔹 7. 直接从加载的 prompts 中恢复
            fixed_query_instances = []
            prompts = []
            for item in filtered_batch:
                qi = QueryInstance(
                    query=item.get('question', ''),
                    data=[],
                    fixed_data=[],
                    answers=item.get('reference_answer', []),
                    ground_truth=[]
                )
                fixed_query_instances.append(qi)
                prompts.append(item.get('prompt', ''))
            method_time = 0
            
        # 🔹 8. 保存 Prompts（如果只保存 prompts）
        if save_prompts_only:
            for qi, prompt in zip(fixed_query_instances, prompts):
                prompt_entry = {
                    "question": qi.query,
                    "prompt": prompt,
                    "reference_answer": qi.answers,
                    "context_type": getattr(qi, 'context_type', None),
                    "method": method,
                    "task": task,
                }
                prompts_buffer.append(prompt_entry)
            
            # 批量保存
            if len(prompts_buffer) >= save_interval:
                with jsonlines.open(prompt_save_path, 'a') as writer:  # ✅ 使用 jsonlines.open
                    for entry in prompts_buffer:
                        writer.write(entry)  # ✅ 使用 Writer 对象的 write 方法
                print(f"💾 已保存 {len(prompts_buffer)} 条 Prompts 到 {prompt_save_path}")
                prompts_buffer = []
            
            pbar.update(len(batch))
            continue  # 跳过 LLM 调用

        # 🔹 9. LLM 生成
        gen_start = time.time()
        predictions = get_generate(prompts, model)
        gen_time = time.time() - gen_start
        
        n_samples = len(fixed_query_instances)
        batch_gen_time_per_item = gen_time / n_samples
        batch_method_time_per_item = method_time / n_samples
        
        # 后处理和评估
        for item, pred in zip(fixed_query_instances, predictions):
            processed_pred, parse_status, is_fallback = postprocess_answer(pred)
            print(processed_pred)
            metrics = evaluate_answer(processed_pred, item.answers)
            is_correct = metrics['exact_match'] > 0
            
            result = {
                "question": item.query,
                "prediction": processed_pred,
                "raw_prediction": pred,
                "reference_answer": item.answers,
                "is_correct": is_correct,                         
                "metrics": metrics,                               
                "status": "parsed_fallback" if is_fallback else "success",
                "parse_status": parse_status,                     
                "timing": {
                    "method_processing": batch_method_time_per_item,
                    "generation": batch_gen_time_per_item,
                    "total": batch_method_time_per_item + batch_gen_time_per_item
                },
                "context_type": getattr(item, 'context_type', None),
                "parse_fallback_used": is_fallback,
            }
            
            results.append(result)
            new_results_buffer.append(result)
            
            # 标记为已处理
            processed_questions.add(item.query)
            
            # 累积指标
            for metric_name, score in metrics.items():
                all_metrics[metric_name].append(score)
        
        pbar.update(len(batch))
        
        # 🔹 10. 检查是否需要增量保存结果
        if len(new_results_buffer) >= save_interval:
            # 合并旧结果和新缓冲区结果
            full_results_to_save = existing_results + new_results_buffer
            
            avg_metrics_curr = {
                metric_name: sum(scores) / len(scores) if scores else 0
                for metric_name, scores in all_metrics.items()
            }
            
            output_data = {
                "config": config,
                "avg_metrics": avg_metrics_curr,
                "total_processed": len(full_results_to_save),
                "results": full_results_to_save
            }
            
            if output_path:
                out_file = Path(output_path) / f"{task}.json"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                temp_file = out_file.with_suffix('.tmp')
                
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                temp_file.replace(out_file)
                
                print(f"💾 已增量保存 {len(new_results_buffer)} 条新记录 (总计：{len(full_results_to_save)})")
            
            # 更新 existing_results 以便下次增量保存包含之前的数据
            existing_results.extend(new_results_buffer)
            new_results_buffer = []

    pbar.close()
    
    # 🔹 11. 保存剩余 Prompts
    if save_prompts_only and prompts_buffer:
        with jsonlines.open(prompt_save_path, 'a') as writer:  # ✅ 使用 jsonlines.open
            for entry in prompts_buffer:
                writer.write(entry)  # ✅ 正确调用
        print(f"💾 已保存剩余 {len(prompts_buffer)} 条 Prompts")
    
    # 🔹 12. 保存剩余结果
    if new_results_buffer:
        full_results_to_save = existing_results + new_results_buffer
        avg_metrics_curr = {
            metric_name: sum(scores) / len(scores) if scores else 0
            for metric_name, scores in all_metrics.items()
        }
        output_data = {
            "config": config,
            "avg_metrics": avg_metrics_curr,
            "total_processed": len(full_results_to_save),
            "results": full_results_to_save
        }
        
        if output_path:
            out_file = Path(output_path) / f"{task}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = out_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            temp_file.replace(out_file)
            print(f"💾 已保存剩余 {len(new_results_buffer)} 条记录 (总计：{len(full_results_to_save)})")

    # 计算最终指标
    avg_metrics = {
        metric_name: sum(scores) / len(scores) if scores else 0
        for metric_name, scores in all_metrics.items()
    }
    
    if not save_prompts_only:
        print("\n" + "="*60)
        print("Evaluation Finished!")
        print(f"Total samples processed this run: {len(results) - len(existing_results) + len(new_results_buffer)}")
        print(f"Exact Match: {avg_metrics['exact_match']:.4f}")
        print(f"F1 Score: {avg_metrics['f1']:.4f}")
        print(f"Match Score: {avg_metrics['match']:.4f}")
        print("="*60)
    
    return results, avg_metrics

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation on HotpotQA without Retrieval")
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to HotpotQA data file (json or jsonl)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='qwen3.5:9b',
        help='LLM model name for Ollama'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save evaluation results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5,
        help='Batch size for LLM inference'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='LLM temperature parameter'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='vanilla',
        choices=['vanilla', 'contradoc', 'sparsecl', 'sentli', 'hyper_simulation'],
        help='Method to use: vanilla or contradoc'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='hotpotqa',
        choices = ['hotpotqa','musique', 'multihop', 'ARC', 'legalbench'],
        help='Task type (default: hotpotqa)'
    )
    
    parser.add_argument(
        '--build',
        action='store_true',
        help='Whether to build hypergraph (default: False). Set to True to build hypergraph before evaluation.'
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Whether to rebuild hypergraph (default: False). Set to True to rebuild hypergraph before evaluation.'
    )

    parser.add_argument(
        '--using_support_only',
        action='store_true',
        help='Whether to use supporting paragraphs only (default: False). Set to True to use only supporting paragraphs.'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=500,
        help='Save results every N samples (default: 500)'
    )
    
    parser.add_argument(
        '--save_prompts_only',
        action='store_true',
        help='If set, only generate and save prompts without running LLM evaluation.'
    )
    
    parser.add_argument(
        '--load_prompts',
        type=str,
        default=None,
        help='Path to a saved prompts file (jsonl or json) to load and run LLM directly.'
    )

    args = parser.parse_args()
    build_flag = args.build == False
    rebuild_flag = args.rebuild == True
    using_support_only_flag = args.using_support_only == True

    # 运行评估
    run_rag_evaluation(
        data_path=args.data_path,
        model_name=args.model_name,
        output_path=args.output_path,
        batch_size=args.batch_size,
        temperature=args.temperature,
        method=args.method,
        task=args.task,
        build=build_flag,
        rebuild=rebuild_flag,
        using_support_only=using_support_only_flag,
        save_interval=args.save_interval,
        save_prompts_only=args.save_prompts_only,
        load_prompts=args.load_prompts
    )


if __name__ == "__main__":
    main()
