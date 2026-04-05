import json
import jsonlines
from typing import List, Dict, Any
from pathlib import Path

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
                # 检查是否是我们通过 rag.py 检索出来带有 paragraphs 的数据
                paragraphs = item.get("paragraphs", [])
                
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
                # 兼容从 rag.py 中传过来的 answerKey 可能是列表或者是字符串的情况
                if isinstance(answer_label, list) and len(answer_label) > 0:
                    answer_label = answer_label[0]
                    
                answer_text = ""
                if answer_label and options_label and options_text:
                    try:
                        idx = options_label.index(answer_label)
                        answer_text = options_text[idx]
                    except (ValueError, IndexError):
                        answer_text = answer_label
                
                # 如果是带检索结果的数据，我们需要把 paragraphs 塞进 context 里
                context = []
                supporting_flags = []
                for p in paragraphs:
                    title = (p.get("title") or "").strip()
                    text = (p.get("text") or "").strip()
                    if title or text:
                        context.append((title, [text] if text else []))
                        supporting_flags.append(True)
                
                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": question_with_options,
                    "answer": answer_text,           # 答案文本（用于参考）
                    "answer_label": answer_label,    # ✅ 答案标签（用于评估）
                    "options": options_text,
                    "option_labels": options_label,
                    "context": context,              # 包含检索回来的背景知识
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
