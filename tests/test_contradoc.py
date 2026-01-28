# contradoc.py
from hyper_simulation.baselines.contradoc import judge_contradiction_batch
from hyper_simulation.question_answer.rag_no_retrival import load_data
from langchain_ollama import ChatOllama

# ==========================================
# 示例运行逻辑
# ==========================================
if __name__ == "__main__":
    # 示例数据
    model = ChatOllama(model="qwen2.5:14b", temperature=0.8, top_p=0.95)
    file_dir = "/home/vincent/.dataset/HotpotQA/sample500/"
    data = load_data(file_dir, task="hotpotqa")
    sample_size = 5
    context_size = 4
    # total 5 * 4 = 20 samples
    # doc_a from all the questions, and doc_b form the context corresponding to each question
    doc_a_list = [item["question"] for item in data[:sample_size]]
    doc_b_list = []
    for item in data[:sample_size]:
        ctxs = item["context"][:context_size]
        for ctx in ctxs:
            context_sentence = " ".join(ctx[1])
            doc_b_list.append(f"{ctx[0]}\n{context_sentence}")
    doc_a_list = doc_a_list * context_size  # repeat for each context
    print(f"Total {len(doc_a_list)} pairs to judge contradiction.")
    judge_contradiction_batch(doc_a_list, doc_b_list, model=model)  # replace None with your LLM model
