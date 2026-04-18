import jsonlines
from pathlib import Path

def debug():
    files = {
        "contract_qa": "/home/vincent/hyper-simulation/data/lagel_multihop/contract_qa.jsonl",
        "privacy_policy_qa": "/home/vincent/hyper-simulation/data/lagel_multihop/privacy_policy_qa.jsonl"
    }
    
    for name, path in files.items():
        if not Path(path).exists():
            print(f"File not found: {path}")
            continue
            
        questions = []
        with jsonlines.open(path) as reader:
            for item in reader:
                q = item.get('question') or item.get('query')
                questions.append(q)
                
        q_set = set(questions)
        print(f"[{name}] 总行数: {len(questions)}, 去重后: {len(q_set)}")

if __name__ == "__main__":
    debug()
