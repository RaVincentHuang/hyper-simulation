import json
import re
import jsonlines
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path
from hyper_simulation.question_answer.vmdit.metrics import (
    exact_match_score, 
    metric_max_over_ground_truths,
    qa_f1_score,
    match
)
from hyper_simulation.query_instance import QueryInstance
from hyper_simulation.utils.log import current_task, getLogger
from hyper_simulation.llm.prompt.musique import MUSIQUE_QA_BASE


def load_musique_data(file_path: str, use_supporting_only: bool = True) -> List[Dict[str, Any]]:
    """
    加载 MuSiQue 数据集
    
    Args:
        file_path: 数据路径
        use_supporting_only: 是否只保留支持段落（关键修改！）
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
            raise ValueError("Unsupported file format. Please use .jsonl")

        with jsonlines.open(path, "r") as reader:
            for item in reader:
                paragraphs = item.get("paragraphs", []) or []

                context = []
                supporting_flags = []

                for p in paragraphs:
                    title = (p.get("title") or "").strip()
                    paragraph_text = (p.get("paragraph_text") or "").strip()
                    is_supporting = bool(p.get("is_supporting", False))
                    if use_supporting_only:
                        if is_supporting and (title or paragraph_text):
                            context.append((title, [paragraph_text] if paragraph_text else []))
                            supporting_flags.append(True)
                    else:
                        if title or paragraph_text:
                            context.append((title, [paragraph_text] if paragraph_text else []))
                            supporting_flags.append(is_supporting)

                if use_supporting_only and len(context) == 0:
                    print(f"⚠️ Warning: Sample {item.get('id', 'unknown')} has no supporting paragraphs!")
                    continue

                formatted_item = {
                    "_id": item.get("id", ""),
                    "question": (item.get("question") or "").strip(),
                    "answer": (item.get("answer") or "").strip(),
                    "answer_alias": item.get("answer_alias", []) or [],
                    "answerable": item.get("answerable", True),
                    "context": context,
                    "supporting_flags": supporting_flags,
                    "original_paragraph_count": len(paragraphs),
                    "filtered_paragraph_count": len(context),
                }
                data.append(formatted_item)

    return data


def build_prompt(question: str, context_text: str) -> str:
    """构建 MuSiQue 任务的 prompt"""
    prompt = f"""Read the following paragraphs carefully and answer the question.

PARAGRAPHS:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Answer using ONLY the information from the paragraphs above
- Your answer must be a SHORT PHRASE (1-5 words), NOT a full sentence
- If the answer is a number, date, or name, output ONLY that value
- If you cannot find the answer, output exactly: unanswerable

End your response with this exact format:
ANSWER: <your short answer here>

RESPONSE:
"""
    return prompt


def evaluate_answer(prediction: str, ground_truth: list | str) -> Dict[str, float]:
    """评估生成的答案"""
    if isinstance(ground_truth, list):
        ground_truths = ground_truth
    else:
        ground_truths = [ground_truth]
    
    em_score = metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths
    )
    
    f1_score = max([qa_f1_score(prediction, gt) for gt in ground_truths])
    match_score = match(prediction, ground_truths)
    
    return {
        "exact_match": em_score,
        "f1": f1_score,
        "match": match_score
    }


def postprocess_answer(answer: str) -> str:
    """后处理 - 提取短答案"""
    print(answer)
    if not answer:
        return "unanswerable"

    answer = answer.replace("</s>", "").strip()
    
    # 1. 提取 ANSWER: 后的内容
    patterns = [
        r"ANSWER:\s*(.+?)(?:\n|$)",
        r"Answer:\s*(.+?)(?:\n|$)",
        r"###\s*Final\s*Answer:\s*(.+?)(?:\n|$)"
    ]
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            break
    
    # 2. 如果答案太长，提取核心实体
    words = answer.split()
    if len(words) > 10:
        quote_match = re.search(r'["\']([^"\']+)["\']', answer)
        if quote_match:
            return quote_match.group(1).strip()
        
        proper_nouns = [w for w in words if w[0].isupper()]
        if proper_nouns and len(' '.join(proper_nouns)) < 50:
            return ' '.join(proper_nouns)
        
        answer = ' '.join(words[:5])
    
    # 3. 清理常见噪声
    noise_patterns = [
        r'^The\s+', r'^A\s+', r'^An\s+',
        r'\s+is\s+.*$', r'\s+was\s+.*$', r'\s+are\s+.*$',
        r'^It\s+is\s+', r'^According\s+to\s+', r'^Based\s+on\s+',
    ]
    for pattern in noise_patterns:
        answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
    
    # 4. 去除首尾标点
    answer = answer.strip(" .,;:!?\"'()[]")
    
    # 5. 处理 unanswerable 变体
    if answer.lower() in ['unanswerable', 'unknown', 'none', 'not mentioned', 'cannot be determined']:
        return "unanswerable"
    
    # 6. 截断过长答案
    if len(answer) > 100:
        answer = answer[:100].rsplit(' ', 1)[0]
    
    return answer.strip() if answer else "unanswerable"


def run_rag_evaluation(
    data_path: str,
    model_name: str = "qwen3.5:9b",
    output_path: str = "",
    batch_size: int = 5,
    temperature: float = 0.2,
    task: str = "musique",
    method: str = "vanilla",
    build: bool = True,
    rebuild: bool = False,
    use_supporting_only: bool = True  # 🔑 新增参数
):
    """运行 MuSiQue RAG 评估任务"""
    print(f"Loading data from {data_path}...")
    print(f"Use supporting paragraphs only: {use_supporting_only}")
    data: List[Dict[str, Any]] = load_musique_data(data_path, use_supporting_only=use_supporting_only)
    unanswerable_ids = []
    print(f"Loaded {len(data)} samples")
    print(f"Initializing LLM: {model_name}")
    
    from langchain_ollama import ChatOllama
    from hyper_simulation.llm.chat_completion import get_generate
    model = ChatOllama(
        model=model_name, 
        temperature=temperature, 
        top_p=0.7,
        reasoning=False,
    )
    results = []
    all_metrics = {
        "exact_match": [],
        "f1": [],
        "match": []
    }

    print(f"Starting evaluation with batch_size={batch_size}...")
    current_task.set(task)
    logger = getLogger(__name__, "INFO")
    
    debug_count = 0
    
    for batch_start in tqdm(range(0, len(data), batch_size), desc="Processing batches", position=0, leave=True):
        assert batch_start + batch_size <= len(data) 
        batch = data[batch_start:(batch_start + batch_size)]
        
        # 构建 QueryInstance
        query_instances = []
        for item in batch:
            ground_truths = []
            supporting_flags = item.get("supporting_flags", []) or []
            for idx, (title, sentences) in enumerate(item["context"]):
                # 因为只保留支持段落，这里全部为 True
                ground_truths.append((True, "\n".join(sentences)))

            answer = item.get("answer", "")
            aliases = item.get("answer_alias", []) or []
            answers = [answer] + [a for a in aliases if a != answer]

            query_instance = QueryInstance(
                query=item["question"],
                data=[
                    f"{title}.\n" + "\n".join(sentences)
                    for title, sentences in item["context"]
                ],
                fixed_data=[],
                answers=answers,
                ground_truth=ground_truths,
            )
            query_instances.append(query_instance)
        
        if method == "vanilla":
            fixed_query_instances = query_instances
        elif method == "contradoc":
            # 使用Contradoc方法修正数据
            from hyper_simulation.baselines.contradoc import query_fixup
            fixed_query_instances = [
                query_fixup(qi, model=model) for qi in query_instances
            ]
        elif method == "hyper_simulation":
            if not build:
                from hyper_simulation.component.build_hypergraph import build_hypergraph_batch
                build_hypergraph_batch(query_instances, dataset_name=task, force_rebuild=rebuild)
                continue
            else:
                from hyper_simulation.component.consistent import query_fixup
                fixed_query_instances = [query_fixup(qi, task) for qi in query_instances]
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # 准备 prompts
        prompts = []
        for item in fixed_query_instances:
            context_text = "\n\n".join(item.fixed_data if item.fixed_data else item.data)
            prompt = build_prompt(item.query, context_text)
            prompts.append(prompt)
        
        predictions = get_generate(prompts, model)
        
        # 后处理和评估
        for item, pred in zip(fixed_query_instances, predictions):
            processed_pred = postprocess_answer(pred)
            metrics = evaluate_answer(processed_pred, item.answers)
            
            result = {
                "prediction": processed_pred,
                "ground_truth": item.answers,
                "metrics": metrics,
            }
            results.append(result)
            
            for metric_name, score in metrics.items():
                all_metrics[metric_name].append(score)
    
    # 计算平均指标
    avg_metrics = {
        metric_name: sum(scores) / len(scores) if scores else 0
        for metric_name, scores in all_metrics.items()
    }
    
    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"Total samples: {len(results)}")
    print(f"Exact Match: {avg_metrics['exact_match']:.4f}")
    print(f"F1 Score: {avg_metrics['f1']:.4f}")
    print(f"Match Score: {avg_metrics['match']:.4f}")
    print("="*60)
    
    # 统计 unanswerable 比例
    unanswerable_count = sum(1 for r in results if r['prediction'].lower() == 'unanswerable')
    print(f"Unanswerable predictions: {unanswerable_count}/{len(results)} ({unanswerable_count/len(results)*100:.1f}%)")

    # 保存结果
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "config": {
                "model_name": model_name,
                "data_path": data_path,
                "batch_size": batch_size,
                "temperature": temperature,
                "use_supporting_only": use_supporting_only,
                "total_samples": len(results)
            },
            "avg_metrics": avg_metrics,
            "results": results
        }
        
        suffix = "_supporting_only" if use_supporting_only else "_all_paragraphs"
        with open(f"{output_path}/{task}_{method}{suffix}.json", 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
    
    return results, avg_metrics


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation on MuSiQue")
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='qwen3.5:27b')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--method', type=str, default='vanilla', choices=['vanilla', 'hyper_simulation', 'contradoc'])
    parser.add_argument('--task', type=str, default='musique')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--use_supporting_only', action='store_true', default=True,
                        help='Only use supporting paragraphs (default: True)')

    args = parser.parse_args()
    build_flag = args.build == False
    rebuild_flag = args.rebuild == True
    
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
        use_supporting_only=args.use_supporting_only
    )


if __name__ == "__main__":
    main()