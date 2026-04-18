import json
import jsonlines

def check_missing():
    # 1. 读取 eval_data 中的 1416 个 question
    eval_questions = []
    with jsonlines.open("/home/vincent/hyper-simulation/data/eval_data/musique_answerable copy.jsonl") as reader:
        for item in reader:
            eval_questions.append(item.get('question'))
            
    eval_q_set = set(eval_questions)
    print(f"✅ eval_data 中共有 {len(eval_questions)} 个问题，去重后有 {len(eval_q_set)} 个唯一问题。")
    
    # 2. 读取 mid_result/sparsecl 中的 question
    sparsecl_questions = []
    with jsonlines.open("/home/vincent/hyper-simulation/data/mid_result/sparsecl/musique.jsonl") as reader:
        for item in reader:
            sparsecl_questions.append(item.get('question'))
            
    sparsecl_q_set = set(sparsecl_questions)
    print(f"✅ sparsecl 中共有 {len(sparsecl_questions)} 个问题，去重后有 {len(sparsecl_q_set)} 个唯一问题。")
    
    # 3. 找出缺失的
    missing = eval_q_set - sparsecl_q_set
    print(f"\n⚠️ 共有 {len(missing)} 个问题在 sparsecl 中找不到！")
    
    # 打印前 5 个缺失的看看
    for idx, q in enumerate(list(missing)[:5]):
        print(f"缺失 {idx+1}: {q}")

if __name__ == "__main__":
    check_missing()
