import json
import jsonlines
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path

from langchain_ollama import ChatOllama
from hyper_simulation.llm.chat_completion import get_generate
from hyper_simulation.question_answer.vmdit.metrics import (
    exact_match_score, 
    metric_max_over_ground_truths,
    qa_f1_score,
    match
)

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
                        ]
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
                        ]
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

def format_context(context: List[List]) -> str:
    """
    将HotpotQA的context格式化为可读的文本
    
    Args:
        context: [[title1, [sent1, sent2, ...]], [title2, [sent1, sent2, ...]], ...]
    
    Returns:
        格式化后的context文本
    """
    formatted_parts = []
    
    for idx, (title, sentences) in enumerate(context, 1):
        # 添加标题
        formatted_parts.append(f"Document {idx}: {title}")
        # 添加句子
        for sent_idx, sentence in enumerate(sentences, 1):
            formatted_parts.append(f"  {sent_idx}. {sentence}")
        formatted_parts.append("")  # 空行分隔不同文档
    
    return "\n".join(formatted_parts)


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
):
    """
    运行RAG评估任务
    
    Args:
        data_path: HotpotQA数据文件路径
        model_name: LLM模型名称
        output_path: 结果输出路径
        batch_size: 批处理大小
        temperature: LLM温度参数
    """
    print(f"Loading data from {data_path}...")
    data = load_hotpotqa_data(data_path)
    
    print(f"Loaded {len(data)} samples")
    print(f"Initializing LLM: {model_name}")
    
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
        batch = data[batch_start:batch_start + batch_size]
        
        # 准备prompts
        prompts = []
        for item in batch:
            # 格式化context
            context_texts = format_context(item['context'])
            # 构建prompt
            prompt = build_prompt(item['question'], context_texts)
            prompts.append(prompt)
        
        # 批量调用LLM
        predictions = get_generate(prompts, model)
        
        # 后处理和评估
        for item, pred in zip(batch, predictions):
            # 后处理答案
            processed_pred = postprocess_answer(pred)
            
            # 评估答案
            metrics = evaluate_answer(processed_pred, item['answer'])
            
            # 记录结果
            result = {
                "id": item.get('_id', item.get('id', 'unknown')),
                "question": item['question'],
                "answer": item['answer'],
                "prediction": processed_pred,
                "raw_prediction": pred,
                "type": item.get('type', 'unknown'),
                "level": item.get('level', 'unknown'),
                "metrics": metrics
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
        
        with open(f"{output_path}/hotpot_qa.json", 'w', encoding='utf-8') as f:
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
        '--no_instruction',
        action='store_true',
        help='Do not use instruction format in prompts'
    )
    
    args = parser.parse_args()
    
    # 运行评估
    run_rag_evaluation(
        data_path=args.data_path,
        model_name=args.model_name,
        output_path=args.output_path,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
