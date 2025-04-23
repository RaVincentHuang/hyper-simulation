import hyper_simulation.question_answer.vmdit.retrieval as retrieval
import hyper_simulation.question_answer.vmdit.relation as relation
import hyper_simulation.question_answer.vmdit.rewrite as rewrite
import hyper_simulation.question_answer.vmdit.trim as trim


if __name__ == "__main__":
    # Load the data
    
    # task_name = "popqa_longtail_w_gs"
    task_names = ["popqa_longtail_w_gs", "triviaqa_test_w_gs", "health_claims_processed", "arc_challenge_processed"]
    task_id = 0
    
    for task_id in range(1, 4):
        data_path = f"data/eval_data/{task_names[task_id]}.jsonl"
        rewrite_path = f"data/new/{task_names[task_id]}.jsonl"
        rel_path = f"data/relation/relation_context_{task_names[task_id]}.json"
        retrieval_path = f"data/retr_result/{task_names[task_id]}.jsonl"
        trimed_path = f"data/trimmed_evidences/trimmed_evidences_{task_names[task_id]}.json"
        
        print(f"calculating relations for {data_path} to {rel_path}")
        relation.calc_relations(data_path, rel_path)
        print(f"calculating rewrites for {data_path} to {rewrite_path}")
        rewrite.calc_rewrite(data_path, rewrite_path)
        print(f"calculating retrieval for {data_path} to {trimed_path}")
        retrieval.calc_retrieval(data_path, rewrite_path)
        print(f"calculating trimming for {data_path} to {trimed_path}")
        trim.calc_trim(rel_path, retrieval_path, trimed_path)
        
    exit()
    
    data_path = f"data/eval_data/{task_names[task_id]}.jsonl"
    rewrite_path = f"data/new/{task_names[task_id]}.jsonl"
    rel_path = f"data/relation/relation_context_{task_names[task_id]}.json"
    retrieval_path = f"data/retr_result/{task_names[task_id]}.jsonl"
    trimed_path = f"data/trimmed_evidences/trimmed_evidences_{task_names[task_id]}.json"
    
    
    
    print(f"calculating relations for {data_path} to {rel_path}")
    relation.calc_relations(data_path, rel_path)
    print(f"calculating rewrites for {data_path} to {rewrite_path}")
    rewrite.calc_rewrite(data_path, rewrite_path)
    print(f"calculating retrieval for {data_path} to {trimed_path}")
    retrieval.calc_retrieval(data_path, rewrite_path)
    print(f"calculating trimming for {data_path} to {trimed_path}")
    trim.calc_trim(rel_path, retrieval_path, trimed_path)
    