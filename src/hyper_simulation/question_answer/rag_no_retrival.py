import sentencepiece
import json
import re
import jsonlines
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path
import time

from hyper_simulation.query_instance import QueryInstance, build_query_instance_for_task
from hyper_simulation.utils.log import current_task
from hyper_simulation.question_answer.utils.load_data import load_data
from hyper_simulation.question_answer.utils.build_prompt import build_prompt
from hyper_simulation.question_answer.utils.post_answer import postprocess_answer, evaluate_answer

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
    save_interval: int = 100,
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
    if load_prompts:
        pbar_initial = 0
        pbar_total = len(items_to_process)
    else:
        pbar_initial = len(processed_questions)
        pbar_total = len(data)
    # 进度条初始化：只显示剩余任务数量
    pbar = tqdm(total=pbar_total, desc=f"Processing {task}_{method}", position=0, leave=True, initial=pbar_initial)

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
            continue

        # 🔹 6. 构建 QueryInstance (如果没有加载 prompts)
        if not load_prompts:
            query_instances = [build_query_instance_for_task(item, task) for item in filtered_batch]
            if not query_instances:
                pbar.update(len(filtered_batch))
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
            elif method == "cdit":
                from hyper_simulation.baselines.CDIT import query_fixup
                fixed_query_instances = [query_fixup(qi, model=model) for qi in query_instances]
            elif method == "bsim":
                from hyper_simulation.baselines.BSIM import run_bsim_for_query
                fixed_query_instances = [run_bsim_for_query(qi, task=task) for qi in query_instances]
            elif method == "hyper_simulation":
                if not build:
                    from hyper_simulation.component.build_hypergraph import build_hypergraph_batch_gpu
                    build_hypergraph_batch_gpu(query_instances, dataset_name=task, force_rebuild=rebuild, batch_size=128)
                    print("Hypergraph built. Please re-run with --build to evaluate.")
                    pbar.update(len(filtered_batch))
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
            
            pbar.update(len(filtered_batch))
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
            # import sys
            # tqdm.write(f"[{pbar.n}] {processed_pred}", file=sys.stdout)
            metrics = evaluate_answer(processed_pred, item.answers)
            
            pbar.update(1)  # 补回这行，确保处理过的每个条目都能推动进度条前进
            
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
        # 如果是 load_prompts，说明这原本是一个从 .jsonl 加载的任务（如 sentli）
        # 此时 prompt_save_path 是 data/mid_result/{method}/{task}.jsonl
        # 我们确保用 jsonlines 追加写入
        if prompt_save_path:
            with jsonlines.open(prompt_save_path, 'a') as writer:
                for entry in prompts_buffer:
                    writer.write(entry)
            print(f"💾 已保存剩余 {len(prompts_buffer)} 条 Prompts 到 {prompt_save_path}")
        else:
            print("⚠️ 提示词缓冲非空，但未设置 prompt_save_path！")
    
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
        default=100,
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
        choices=['vanilla', 'contradoc', 'sparsecl', 'sentli', 'cdit', 'bsim', 'hyper_simulation'],
        help='Method to use: vanilla, contradoc, sparsecl, sentli, cdit, bsim, hyper_simulation'
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
        default=100,
        help='Save results every N samples (default: 100)'
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

# 1. 先在excel上进行记录表
# > 小模型