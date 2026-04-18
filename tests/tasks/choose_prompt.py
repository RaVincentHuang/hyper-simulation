import os
import json
import jsonlines
import re
from pathlib import Path

def normalize_text(text):
    """
    统一格式化文本，忽略多余空格，以便于准确比对。
    """
    if not text:
        return ""
    # 将所有的空白字符（包括换行、制表符等）替换为一个空格，并去掉首尾空格
    return re.sub(r'\s+', ' ', text).strip()

def extract_prompts_for_task(eval_file_path, task_name, mid_result_filename):
    """
    提取指定任务在各个 baseline 方法下对应的 prompt，
    分别保存到独立的文件中。
    
    Args:
        eval_file_path: 原始评测文件路径 (如: data/eval_data/contract_qa.jsonl)
        task_name: 任务名称，用于创建输出目录 (如: contract_qa)
        mid_result_filename: 在 mid_result 中对应的文件名 (如: legalbench.jsonl 或 multihop.jsonl)
    """
    # 1. 提取问题的 question 及其对应的完整原始数据
    questions_to_extract = []
    eval_data_list = [] # 记录所有原始数据，因为可能有重复的 question
    normalized_questions_map = {} # 规范化后的问题映射回原问题
    
    with jsonlines.open(eval_file_path) as reader:
        for item in reader:
            q = item.get('question') or item.get('query')
            if not q:
                continue
            questions_to_extract.append(q)
            eval_data_list.append(item)
            normalized_questions_map[normalize_text(q)] = q

    print(f"✅ [{task_name}] 需要提取的 {len(questions_to_extract)} 个问题已加载。")

    # 2. 指定这四个方法对应的 mid_result 目录
    methods = ["sparsecl", "bsim", "sentli", "her"]
    base_dir = "/home/vincent/hyper-simulation/data/mid_result"
    output_dir = f"/home/vincent/hyper-simulation/data/lagel_multihop/{task_name}"
    
    os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

    # 3. 遍历提取并保存
    # 记录哪些规范化后的问题在任何一个方法中缺失
    not_extracted_norm_q_global = {normalize_text(q) for q in questions_to_extract}
    
    for method in methods:
        file_path = os.path.join(base_dir, method, mid_result_filename)
        output_prompts = os.path.join(output_dir, "prompts", f"{method}_prompts.jsonl")
        
        if not os.path.exists(file_path):
            print(f"  ⚠️ {file_path} 不存在，跳过。")
            continue
            
        extracted_data = []
        with jsonlines.open(file_path) as reader:
            for item in reader:
                q = item.get("question") or item.get("query")
                norm_q = normalize_text(q)
                
                if norm_q in normalized_questions_map:
                    item["question"] = normalized_questions_map[norm_q]
                    extracted_data.append(item)
                    
        # 计算哪些问题没有在这个方法的 mid_result 中出现
        extracted_norm_q = {normalize_text(item.get("question") or item.get("query")) for item in extracted_data}
        
        # 交集：只要有一个方法没提取到，我们就把这个问题的 norm_q 视为全局未提取
        all_norm_q = {normalize_text(q) for q in questions_to_extract}
        not_extracted_norm_q_global = not_extracted_norm_q_global.intersection(all_norm_q - extracted_norm_q)
        
        # 为了保证输出的 prompt 数量和 eval_data 一致（即保留重复），
        # 我们根据 eval_data_list 来生成最终的 prompt 文件，对于找到的 prompt，按顺序复用。
        # 构建一个根据 norm_q 查找提取到的 prompt 的字典
        prompt_by_norm_q = {normalize_text(item.get("question") or item.get("query")): item for item in extracted_data}
        
        final_prompts = []
        for eval_item in eval_data_list:
            q = eval_item.get("question") or eval_item.get("query")
            norm_q = normalize_text(q)
            if norm_q in prompt_by_norm_q:
                # 复制一份以防多次引用同一个对象被意外修改
                final_prompts.append(prompt_by_norm_q[norm_q].copy())
                
        with jsonlines.open(output_prompts, 'w') as writer:
            writer.write_all(final_prompts)
            
        print(f"  🎯 成功为方法 {method} 提取 {len(final_prompts)} 个 prompt")

    # 4. 保存未匹配到的原始数据到唯一的 data.jsonl 中
    output_data = os.path.join(output_dir, "data.jsonl")
    
    not_extracted_data = []
    for item in eval_data_list:
        q = item.get("question") or item.get("query")
        if normalize_text(q) in not_extracted_norm_q_global:
            not_extracted_data.append(item)
            
    with jsonlines.open(output_data, 'w') as writer:
        writer.write_all(not_extracted_data)
    print(f"  🎯 成功将 {len(not_extracted_data)} 个未匹配的原始数据保存到 {output_data}")
    print("-" * 50)

def main():
    # 定义需要提取的任务及其对应的原始数据路径和在 mid_result 中的文件名
    tasks = [
        {
            "name": "hotpot_distractor_easy",
            "eval_path": "/home/vincent/hyper-simulation/data/hotpotqa/hotpot_distractor_easy.jsonl",
            "mid_file": "hotpotqa.jsonl"
        },
        {
            "name": "hotpot_distractor_medium",
            "eval_path": "/home/vincent/hyper-simulation/data/hotpotqa/hotpot_distractor_medium.jsonl",
            "mid_file": "hotpotqa.jsonl"
        },
        {
            "name": "hotpot_distractor_hard",
            "eval_path": "/home/vincent/hyper-simulation/data/hotpotqa/hotpot_distractor_hard.jsonl",
            "mid_file": "hotpotqa.jsonl"
        }
    ]
    
    for task in tasks:
        extract_prompts_for_task(
            eval_file_path=task["eval_path"],
            task_name=task["name"],
            mid_result_filename=task["mid_file"]
        )

if __name__ == "__main__":
    main()
