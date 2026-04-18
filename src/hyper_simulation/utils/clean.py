import re
import os
import json
from pathlib import Path

def clean_text_for_spacy(text: str) -> str:
    if not text:
        return ""

    # 1. 去除注释标记，形如 [n 123]
    text = re.sub(r'\[n\s+\d+\]', '', text)
    
    # 2. 将所有类型的空白字符（包括转义字符、全角空格、不换行空格）替换为标准半角空格
    # \s 在正则中匹配 [ \t\n\r\f\v] 以及 Unicode 定义的所有空格
    text = re.sub(r'\s+', ' ', text)
    
    # 3. 去除首尾空格
    text = text.strip()
    
    # 4. 处理替换非标准连字符（如全角连字符、长破折号等）为标准半角连字符 '-' 例如 –
    text = re.sub(r'[–—―]', '-', text)
    
    return text

def deduplicate_jsonl_files():
    """
    遍历指定的目录，读取所有的 .jsonl 文件，基于 'question' 字段进行去重，
    并将去重后的数据覆盖写回原文件。
    """
    target_dirs = [
        "/home/vincent/hyper-simulation/data/baseline",
        "/home/vincent/hyper-simulation/data/mid_result",
        "/home/vincent/.dataset/LegalBench/sample500",
        "/home/vincent/.dataset/ARC/sample_ARC",
        "/home/vincent/.dataset/MultiHop/sample2500",
        "/home/vincent/.dataset/musique/sample3000",
        "/home/vincent/.dataset/HotpotQA/sample1000"
    ]
    
    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            print(f"⚠️ 目录不存在跳过: {target_dir}")
            continue
            
        # 递归查找所有 .jsonl 文件
        for filepath in Path(target_dir).rglob("*.jsonl"):
            print(f"🔄 正在处理: {filepath}")
            seen_questions = set()
            unique_lines = []
            original_count = 0
            
            try:
                # 读取并去重
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        original_count += 1
                        try:
                            data = json.loads(line)
                            question = data.get('question', '')
                            # 如果没有 question 字段或者 question 还没出现过
                            if not question or question not in seen_questions:
                                if question:
                                    seen_questions.add(question)
                                unique_lines.append(line)
                        except json.JSONDecodeError:
                            # 容错：如果无法解析为 JSON，则保留原行
                            unique_lines.append(line)
                
                # 写回原文件
                if len(unique_lines) < original_count:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        for line in unique_lines:
                            f.write(line + '\n')
                    print(f"  ✅ 完成去重: 原有 {original_count} 行 -> 去重后 {len(unique_lines)} 行 (删除了 {original_count - len(unique_lines)} 条重复数据)")
                else:
                    print(f"  ✨ 无需去重: 文件共 {original_count} 行，没有发现重复的 question。")
                    
            except Exception as e:
                print(f"  ❌ 处理文件 {filepath} 时出错: {e}")

def deduplicate_json_files():
    """
    遍历指定的目录，读取所有的 .json 文件（整个文件是一个 JSON 对象，包含 'results' 列表），
    基于 'question' 字段对 'results' 列表进行去重，
    更新 'total_processed' 并将去重后的数据覆盖写回原文件。
    """
    target_dirs = [
        "/home/vincent/hyper-simulation/data/baseline",
        "/home/vincent/hyper-simulation/data/mid_result",
        "/home/vincent/.dataset/LegalBench/sample500",
        "/home/vincent/.dataset/ARC/sample_ARC",
        "/home/vincent/.dataset/MultiHop/sample2500",
        "/home/vincent/.dataset/musique/sample3000",
        "/home/vincent/.dataset/HotpotQA/sample1000"
    ]
    
    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            continue
            
        for filepath in Path(target_dir).rglob("*.json"):
            print(f"🔄 正在处理 JSON: {filepath}")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "results" not in data or not isinstance(data["results"], list):
                    print(f"  ⏭️ 跳过: 格式不符合预期 (缺少 results 列表)")
                    continue
                
                results = data["results"]
                original_count = len(results)
                
                seen_questions = set()
                unique_results = []
                
                for item in results:
                    question = item.get('question', '')
                    if not question or question not in seen_questions:
                        if question:
                            seen_questions.add(question)
                        unique_results.append(item)
                
                if len(unique_results) < original_count or data.get("total_processed") != len(unique_results):
                    # 更新结果列表和统计数量
                    data["results"] = unique_results
                    data["total_processed"] = len(unique_results)
                    
                    # 同时更新 config 里面的 total_samples_original (如果实际能够确认，但这里通常它代表原本数据集的大小)
                    # 不过为了严谨，如果用户要求都改成现有真实数据，我们就把 config.total_samples_original 也同步（尽管它可能本意是数据集总长度）
                    # 考虑到后续继续跑可能需要知道目标，我们可以让它等于当前的 total_processed，或者根据需要调整。
                    # 这里按照用户的要求，将两者都更新为当前干净数据的真实长度。
                    if "config" in data:
                        data["config"]["total_samples_original"] = len(unique_results)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        
                    print(f"  ✅ 完成去重并更新统计: 原有 {original_count} 条 -> 去重后 {len(unique_results)} 条 (已更新 total_processed 和 total_samples_original)")
                else:
                    # 即使没有重复项，也检查并修复 total_processed 和 total_samples_original 字段
                    needs_update = False
                    if data.get("total_processed") != original_count:
                        data["total_processed"] = original_count
                        needs_update = True
                    if "config" in data and data["config"].get("total_samples_original") != original_count:
                        data["config"]["total_samples_original"] = original_count
                        needs_update = True
                        
                    if needs_update:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"  🔧 修复了文件 {filepath} 中的统计字段，统一为 {original_count}")
                    else:
                        print(f"  ✨ 无需去重: 文件共 {original_count} 条，没有发现重复的 question。")
                    
            except json.JSONDecodeError:
                print(f"  ❌ 跳过: 无法解析 JSON 格式文件 {filepath}")
            except Exception as e:
                print(f"  ❌ 处理文件 {filepath} 时出错: {e}")

def fix_arc_baseline_metrics():
    """
    针对 baseline/*/ARC.json 文件，重新计算 metrics 并修复缺失的 reference_answer。
    同时，去重重复的 question 并修复 metrics。
    """
    import jsonlines
    from pathlib import Path
    from hyper_simulation.question_answer.utils.post_answer import evaluate_answer
    
    # 1. 尝试从原有的评测集或修复后的数据里加载标准答案映射
    answer_map = {}
    reference_file = "/home/vincent/hyper-simulation/data/retr_result/arc/arc_with_context.jsonl"
    if os.path.exists(reference_file):
        with jsonlines.open(reference_file) as reader:
            for item in reader:
                q = item.get('question', '').strip()
                ans = item.get('answerKey', '')
                if isinstance(ans, list) and ans:
                    ans = ans[0]
                if q and ans:
                    answer_map[q] = ans

    # 或者如果带 options 的不匹配，可以从原始数据里匹配
    orig_file = "/home/vincent/hyper-simulation/data/eval_data/arc_challenge_processed.jsonl"
    if os.path.exists(orig_file):
        with jsonlines.open(orig_file) as reader:
            for item in reader:
                q = item.get('question', '').strip()
                ans = item.get('answerKey', '')
                if isinstance(ans, list) and ans:
                    ans = ans[0]
                if q and ans:
                    answer_map[q] = ans
                    
    # 如果 eval_data 没有，再从完整的 ARC dataset 里找
    dataset_dir = Path("/home/vincent/.dataset/ARC/sample_ARC")
    if dataset_dir.exists():
        for file_path in dataset_dir.glob("*.jsonl"):
            with jsonlines.open(file_path) as reader:
                for item in reader:
                    q = item.get('question', '').strip()
                    ans = item.get('answerKey', '')
                    if isinstance(ans, list) and ans:
                        ans = ans[0]
                    if q and ans:
                        answer_map[q] = ans

    baseline_dir = Path("/home/vincent/hyper-simulation/data/baseline")
    for arc_file in baseline_dir.rglob("ARC.json"):
        print(f"🔧 正在修复并清理 {arc_file}...")
        try:
            with open(arc_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "results" not in data:
                continue
                
            seen_questions = set()
            fixed_results = []
            
            for item in data["results"]:
                # 去除 \n\nOptions:\n 等冗余后缀以便更好地匹配
                q_raw = item.get("question", "")
                q_clean = q_raw.split('\n\nOptions:')[0].strip()
                
                # 去重
                if not q_clean or q_clean in seen_questions:
                    continue
                seen_questions.add(q_clean)
                
                # 修复 reference_answer
                current_ans = item.get("reference_answer", [])
                if not current_ans or current_ans == ["[]"] or current_ans == []:
                    # 尝试从映射里找
                    ans = answer_map.get(q_clean)
                    if not ans:
                        # 尝试更宽松的匹配
                        for mq, ma in answer_map.items():
                            if q_clean.startswith(mq) or mq.startswith(q_clean):
                                ans = ma
                                break
                    if ans:
                        item["reference_answer"] = [ans]
                    else:
                        print(f"⚠️ 找不到问题答案: {q_clean[:50]}...")
                
                # 重新计算 metrics
                pred = item.get("prediction", "")
                ref = item.get("reference_answer", [])
                
                # 如果从文件中提取的 ref 为 ["[]"]，实际上应该被当作空处理
                if ref == ["[]"]:
                    ref = []
                
                if ref:
                    metrics = evaluate_answer(pred, ref)
                    item["metrics"] = metrics
                    item["is_correct"] = metrics["exact_match"] > 0
                else:
                    item["metrics"] = {"exact_match": False, "f1": 0, "match": 0}
                    item["is_correct"] = False
                
                fixed_results.append(item)
            
            # 更新整体统计
            data["results"] = fixed_results
            data["total_processed"] = len(fixed_results)
            if "config" in data:
                data["config"]["total_samples_original"] = len(fixed_results)
                
            # 重新计算 avg_metrics
            all_metrics = {"exact_match": [], "f1": [], "match": []}
            for item in fixed_results:
                m = item.get("metrics", {})
                for k in all_metrics:
                    if k in m:
                        all_metrics[k].append(m[k])
                        
            data["avg_metrics"] = {
                k: sum(v)/len(v) if v else 0 
                for k, v in all_metrics.items()
            }
            
            with open(arc_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"  ✅ 修复完成，共计 {len(fixed_results)} 条。Accuracy: {data['avg_metrics'].get('exact_match', 0):.4f}")
                
        except Exception as e:
            print(f"❌ 修复文件 {arc_file} 失败: {e}")

def deduplicate_musique_eval_data():
    """
    针对 /home/vincent/hyper-simulation/data/eval_data/musique_answerable.jsonl
    基于 'id' 字段进行去重。
    """
    filepath = "/home/vincent/hyper-simulation/data/eval_data/musique_answerable.jsonl"
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} 不存在，跳过。")
        return
        
    print(f"🔄 正在处理 musique 评测数据: {filepath}")
    seen_ids = set()
    unique_lines = []
    original_count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                original_count += 1
                try:
                    data = json.loads(line)
                    item_id = data.get('question')
                    
                    if item_id is None:
                        unique_lines.append(line)
                    elif item_id not in seen_ids:
                        seen_ids.add(item_id)
                        unique_lines.append(line)
                except json.JSONDecodeError:
                    unique_lines.append(line)
                    
        if len(unique_lines) < original_count:
            with open(filepath, 'w', encoding='utf-8') as f:
                for line in unique_lines:
                    f.write(line + '\n')
            print(f"  ✅ 完成去重: 原有 {original_count} 行 -> 去重后 {len(unique_lines)} 行 (删除了 {original_count - len(unique_lines)} 条重复数据)")
        else:
            print(f"  ✨ 无需去重: 文件共 {original_count} 行，没有发现重复的 id。")
            
    except Exception as e:
        print(f"  ❌ 处理文件时出错: {e}")

if __name__ == "__main__":
    print("开始执行 musique_answerable.jsonl 的专属去重任务...")
    deduplicate_musique_eval_data()
    
    # print("\n开始执行 ARC Metrics 修复任务...")
    # fix_arc_baseline_metrics()
    
    # print("\n开始执行 JSONL 去重任务...")
    # deduplicate_jsonl_files()
    
    # print("\n开始执行 JSON 去重任务...")
    # deduplicate_json_files()
    
    print("\n去重任务执行完毕。")

