import json
import jsonlines
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path
import logging
from hyper_simulation.utils.log import getLogger
logger = getLogger(__name__)
from hyper_simulation.question_answer.vmdit.metrics import (
    exact_match_score, 
    metric_max_over_ground_truths,
    qa_f1_score,
    match
)
from hyper_simulation.query_instance import QueryInstance

from hyper_simulation.llm.prompt.hotpot_qa import HOTPOT_QA_BASE

# home/vincent/.dataset/HotpotQA/hotpot_*.jsonl
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


def load_data(file_path: str, task: str = "hotpotqa") -> List[Dict[str, Any]]:
    """
    通用数据加载接口，支持不同任务的数据集
    """
    if task == "hotpotqa":
        return load_hotpotqa_data(file_path)
    else:
        raise ValueError(f"Unsupported task: {task}")



def build_prompt(question: str, context_text: str) -> str:
    """
    构建用于LLM的prompt
    
    Args:
        question: 问题文本
        context_texts: 格式化后的context文本
    
    Returns:
        完整的prompt
    """
    prompt = HOTPOT_QA_BASE.format(
        context_text=context_text,
        question=question
    )

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


def postprocess_answer(answer: str) -> str:
    """
    后处理LLM生成的答案
    
    Args:
        answer: 原始答案
    
    Returns:
        处理后的答案
    """
    # 移除特殊标记
    answer = answer.replace("</s>", "").strip()
    
    # 如果答案以空格开头，移除
    if answer and answer[0] == " ":
        answer = answer[1:]
    
    # 只保留第一段（如果有多段）
    # if "\n\n" in answer:
    #     answer = answer.split("\n\n")[0]
    
    return answer.strip()


def run_rag_evaluation(
    data_path: str,
    model_name: str = "qwen2.5:14b",
    output_path: str = "",
    batch_size: int = 5,
    temperature: float = 0.7,
    task: str = "hotpotqa",
    method: str = "vanilla",
    build: bool = True
):
    """
    运行RAG评估任务
    
    Args:
        data_path: HotpotQA数据文件路径
        model_name: LLM模型名称
        output_path: 结果输出路径
        batch_size: 批处理大小
        temperature: LLM温度参数
        method: 评估方法
        task: 任务类型
        build: 判断是否已经转了超图
    """
    print(f"Loading data from {data_path}...")
    data: List[Dict[str, Any]] = load_data(data_path, task)
    
    print(f"Loaded {len(data)} samples")
    print(f"Initializing LLM: {model_name}")
    
    if build:
        from langchain_ollama import ChatOllama
        from hyper_simulation.llm.chat_completion import get_generate
        # 初始化LLM
        model = ChatOllama(model=model_name, temperature=temperature, top_p=0.95)
    
    results = []
    all_metrics = {
        "exact_match": [],
        "f1": [],
        "match": []
    }
    
    print(f"Starting evaluation with batch_size={batch_size}...")
    
    # 按批次处理
    for batch_start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        assert batch_start + batch_size <= len(data) 
        batch = data[batch_start:(batch_start + batch_size)]
        
        # build data as `QueryInstance` by task
        if task == "hotpotqa":
            # query_instance.query = data["question"]
            # query_instance.data from data["context"]
            # fixed_data is empty
            # query_instance.answer form data["answer"]
            # query_instance.ground_truths form data["supporting_facts"]
            query_instances = []
            for item in batch:
                
                # build ground_truth from supporting_facts
                # output as (bool, str)
                # since `supporting_facts`: {`title`: [...], `sent_id`: [...]}
                # and context is List of (title, [sentences])
                # therefore if a title in supporting_facts, we set has_contradiction to True
                # and evidence is the sentences under that title joined as str, for each cor context by sent_id
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
                        f"{title}\n" + "\n".join(sentences)
                        for title, sentences in item['context']
                    ],
                    fixed_data=[],
                    answers=item['answer'],
                    ground_truth=ground_truths
                )
                query_instances.append(query_instance)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        if method == "vanilla":
            # 不进行任何修改，直接使用原始数据
            fixed_query_instances = query_instances
        elif method == "contradoc":
            # 使用Contradoc方法修正数据
            from hyper_simulation.baselines.contradoc import query_fixup
            fixed_query_instances = [
                query_fixup(qi, model=model) for qi in query_instances
            ]
        elif method == "hyper_simulation":
            # 前提：将query_instances的每个item都转换为.pkl文件。
            if not build:
                from hyper_simulation.component.build_hypergraph import build_hypergraph_batch
                build_hypergraph_batch(query_instances, dataset_name=task)
                print("build hypergraph")
                continue
            else:
                from hyper_simulation.component.consistent import query_fixup
                fixed_query_instances = [query_fixup(qi, task) for qi in query_instances[:1]]
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # 准备prompts
        prompts = []
        for item in fixed_query_instances:
            # 格式化context
            context_text = "\n\n".join(item.fixed_data if item.fixed_data else item.data)
            # 构建prompt
            prompt = build_prompt(item.query, context_text)
            prompts.append(prompt)
        
        # 批量调用LLM
        predictions = get_generate(prompts, model)
        
        # 后处理和评估
        for item, pred in zip(fixed_query_instances, predictions):
            # 后处理答案
            processed_pred = postprocess_answer(pred)
            
            # 评估答案
            metrics = evaluate_answer(processed_pred, item.answers)
            
            # 记录结果
            result = {
                "prediction": processed_pred,
                "ground_truth": item.answers,
                "metrics": metrics,
            }
            results.append(result)
            
            # 累积指标
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
    
    # 按类型和难度统计
    if any('type' in r and r['type'] != 'unknown' for r in results):
        print("\nBreakdown by Type:")
        types = set(r['type'] for r in results if r['type'] != 'unknown')
        for qtype in sorted(types):
            type_results = [r for r in results if r['type'] == qtype]
            type_em = sum(r['metrics']['exact_match'] for r in type_results) / len(type_results)
            type_f1 = sum(r['metrics']['f1'] for r in type_results) / len(type_results)
            print(f"  {qtype}: EM={type_em:.4f}, F1={type_f1:.4f} (n={len(type_results)})")
    
    if any('level' in r and r['level'] != 'unknown' for r in results):
        print("\nBreakdown by Level:")
        levels = set(r['level'] for r in results if r['level'] != 'unknown')
        for level in sorted(levels):
            level_results = [r for r in results if r['level'] == level]
            level_em = sum(r['metrics']['exact_match'] for r in level_results) / len(level_results)
            level_f1 = sum(r['metrics']['f1'] for r in level_results) / len(level_results)
            print(f"  {level}: EM={level_em:.4f}, F1={level_f1:.4f} (n={len(level_results)})")
    
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
                "total_samples": len(results)
            },
            "avg_metrics": avg_metrics,
            "results": results
        }
        
        with open(f"{output_path}/{task}_{method}.json", 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
    
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
        default='qwen2.5:14b',
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
        default=0.7,
        help='LLM temperature parameter'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='vanilla',
        choices=['vanilla', 'contradoc', 'hyper_simulation'],
        help='Method to use: vanilla or contradoc'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='hotpotqa',
        help='Task type (default: hotpotqa)'
    )

    parser.add_argument(
        '--build',
        type=str,
        default='True',
        help='Whether hypergraphs already exist (default: "True"). Set to "False" to rebuild.'
    )

    args = parser.parse_args()
    build_flag = (args.build.strip().lower() == 'true')
    # 运行评估
    run_rag_evaluation(
        data_path=args.data_path,
        model_name=args.model_name,
        output_path=args.output_path,
        batch_size=args.batch_size,
        temperature=args.temperature,
        method=args.method,
        task=args.task,
        build=build_flag
    )


if __name__ == "__main__":
    main()
