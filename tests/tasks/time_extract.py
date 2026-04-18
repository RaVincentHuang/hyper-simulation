import os
import json
import jsonlines

def extract_timing():
    methods = ["her", "bsim"]
    datasets = ["hotpotqa", "musique"]
    base_dir = "/home/vincent/hyper-simulation/data/baseline"
    output_dir = "/home/vincent/hyper-simulation/data/time"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for method in methods:
        for dataset in datasets:
            input_file = os.path.join(base_dir, method, f"{dataset}.json")
            output_file = os.path.join(output_dir, f"{method}_{dataset}.jsonl")
            
            if not os.path.exists(input_file):
                print(f"⚠️ {input_file} 不存在，跳过。")
                continue
                
            print(f"🔄 正在处理: {input_file}")
            extracted_data = []
            
            with open(input_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    results = data.get("results", [])
                    
                    for item in results:
                        question = item.get("question")
                        timing = item.get("timing")
                        
                        if question and timing:
                            # 提取出只包含 question 和 timing 的数据
                            extracted_data.append({
                                "question": question,
                                "timing": timing
                            })
                except json.JSONDecodeError as e:
                    print(f"❌ 解析 {input_file} 失败: {e}")
                    continue
                    
            with jsonlines.open(output_file, 'w') as writer:
                writer.write_all(extracted_data)
                
            print(f"  ✅ 成功提取 {len(extracted_data)} 条 timing 数据，保存至 {output_file}")

if __name__ == "__main__":
    extract_timing()