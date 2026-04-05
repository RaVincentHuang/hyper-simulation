import torch
import argparse
import os
import json
import jsonlines
from types import SimpleNamespace
from typing import List, Dict, Union
import glob
import time
from pathlib import Path
# ==============================================================================
# 1. 复用现有的模块 (Reusing Existing Interfaces)
# ==============================================================================

# 复用检索模块的工具函数
#
from hyper_simulation.question_answer.vmdit.retrieval import (
    embed_queries, 
    add_passages, 
    add_hasanswer,
    index_encoded_data
)
# 复用 Contriever 底层接口
#
import contrievers
import contrievers.index
import contrievers.data

# 复用 LLM 调用接口
#
from hyper_simulation.llm.chat_completion import get_generate
from langchain_ollama import ChatOllama

# 复用数据处理和 Prompt 模板
#
from hyper_simulation.question_answer.vmdit.utils import (
    PROMPT_DICT, 
    TASK_INST, 
    postprocess_answers_closed,
    preprocess_input_data
)

# ==============================================================================
# 2. RAG 框架实现 (RAG Framework Implementation)
# ==============================================================================

class RAGPipeline:
    def __init__(self, 
                 retriever_model_path: str = "models/contriever-msmarco",
                 passages_path: str = "data/psgs_w100.tsv",
                 index_path: str = "index_hnsw/",
                 embedding_dir: str = "data/wikipedia_embeddings",
                 llm_model_name: str = "qwen3.5:9b",
                 device: str = "cuda"):
        """
        初始化 RAG 流水线，加载所有必要的模型和索引。
        """
        self.device = device
        
        # --- 初始化检索器 (Retrieval Setup) ---
        print(f"Loading Retriever from {retriever_model_path}...")
        # 直接复用 contrievers.load_retriever
        self.retriever_model, self.retriever_tokenizer, _ = contrievers.load_retriever(retriever_model_path)
        self.retriever_model.eval()
        self.retriever_model.to(device)
        if device == "cuda":
            self.retriever_model.half()

        # 加载索引 (Index)
        # 复用 contrievers.index.Indexer
        print(f"Loading Index from {index_path}...")
        self.index = contrievers.index.Indexer(vector_sz=768, n_subquantizers=0, n_bits=8, mode='hnsw')
        # 替换原有的 self.index.deserialize_from(index_path) 及其周边代码
        index_dir = index_path.rstrip('/')
        index_file = os.path.join(index_dir, "index.faiss")
        meta_file = os.path.join(index_dir, "index_meta.faiss")
        if os.path.exists(index_file):
            import faiss
            import pickle
            print(f"⚡ Loading 65GB index via Memory-Mapped I/O (MMAP) to bypass RAM limit...")
            faiss_idx = faiss.read_index(index_file, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
            target_attr = "index" if hasattr(self.index, "index") else "faiss_index"
            setattr(self.index, target_attr, faiss_idx)
            
            if os.path.exists(meta_file):
                print(f"Loading meta data from {meta_file}")
                with open(meta_file, "rb") as reader:
                    self.index.index_id_to_db_id = pickle.load(reader)
            else:
                print(f"⚠️ Warning: Meta data not found at {meta_file}")
                
            print(f"✅ Index mapped successfully. Physical RAM usage stable.")            
        else:
            print(f"Index not found at {index_path}. Building from embeddings in {embedding_dir}...")
            
            # 获取所有 embedding 文件 (.pkl)
            input_paths = glob.glob(os.path.join(embedding_dir, "passages_*")) 
            input_paths = sorted(input_paths)
            
            if not input_paths:
                 raise FileNotFoundError(f"No embedding files found in {embedding_dir}. Please run generate_passage_embedding.py first.")

            # 构建索引
            start_time = time.time()
            index_encoded_data(self.index, input_paths, indexing_batch_size=1000000) #
            print(f"Indexing finished in {time.time()-start_time:.1f} s.")
            
            # 保存索引以便下次使用
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            self.index.serialize(index_path)
            print(f"Index saved to {index_path}")

        # 加载文档库 (Passages)
        # 复用 contrievers.data.load_passages
        print(f"Loading Passages from {passages_path}...")
        self.passages = contrievers.data.load_passages(passages_path)
        self.passage_id_map = {x["id"]: x for x in self.passages}

        # --- 初始化生成器 (Generation Setup) ---
        print(f"Loading LLM {llm_model_name}...")
        # 复用 ChatOllama
        self.llm = ChatOllama(model=llm_model_name, temperature=0.8, top_p=0.95)

    def retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        执行检索步骤。
        完全复用 vmdit/retrieval.py 中的逻辑。
        """
        # 构造 args 对象以适配 embed_queries 函数的签名
        #
        args = SimpleNamespace(
            lowercase=False, 
            normalize_text=True, 
            per_gpu_batch_size=32, 
            question_maxlength=512
        )

        print("Embedding queries...")
        # 复用 embed_queries
        query_embeddings = embed_queries(args, queries, self.retriever_model, self.retriever_tokenizer)

        print("Searching index...")
        # 复用 index.search_knn
        top_ids_and_scores = self.index.search_knn(query_embeddings, top_k)

        # 构造临时数据结构以利用 add_passages 函数
        dummy_data = [{"question": q} for q in queries]
        
        # 复用 add_passages 将检索结果注入数据
        add_passages(dummy_data, self.passage_id_map, top_ids_and_scores)
        
        # 返回每个 query 对应的 ctxs 列表
        return [item["ctxs"] for item in dummy_data]

    def generate(self, items: List[Dict], task: str = "qa", top_n: int = 5, save_prompts_only: bool = False, prompt_save_path: str = None) -> List[str]:
        """
        执行生成步骤。
        复用 base_line_lm.py 和 vmdit/utils.py 的逻辑。
        """
        
        # 1. 准备 Prompts
        prompts = []
        for item in items:
            # 这里的 item 应该已经包含 'ctxs' (由 retrieve 步骤产生)
            
            # A. 拼接检索到的段落 (Context Construction)
            # 逻辑来源:
            retrieval_result = item.get("ctxs", [])[:top_n]
            evidences = [
                "[{}] {}\n{}".format(i+1, ctx["title"], ctx["text"]) 
                for i, ctx in enumerate(retrieval_result)
            ]
            paragraph = "\n".join(evidences)

            # B. 处理指令和选项 (Instruction Formatting)
            # 逻辑来源:
            # 我们手动构建 preprocess_input_data 的效果
            instruction_text = TASK_INST.get(task, item.get("question", ""))
            
            # 处理 ARC/多选题的选项格式化
            choices_str = ""
            if task in ["arc_c", "arc_easy", "obqa"] and "choices" in item:
                # 简化的选项格式化逻辑，参考 utils.py
                choices = item["choices"]
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                formatted = []
                map_key = {"1": "A", "2": "B", "3": "C", "4": "D"}
                for l, t in zip(labels, texts):
                    k = map_key.get(l, l)
                    formatted.append(f"{k}: {t}")
                if formatted:
                    choices_str = "\n" + "\n".join(formatted)
            
            full_instruction = f"{instruction_text}\n\n### Input:\n{item['question']}{choices_str}"
            
            # C. 应用模板
            # 复用 PROMPT_DICT
            prompt = PROMPT_DICT["prompt_no_input_retrieval"].format(
                paragraph=paragraph,
                instruction=full_instruction
            )
            prompts.append(prompt)

        # 🔹 如果只需要保存 prompts (在这里，我们保存的是原始数据加上 ctxs)
        if save_prompts_only and prompt_save_path:
            prompts_buffer = []
            for item in items:
                # 重新构造类似原始 jsonl 的结构，但加上了 retrieved 的 ctxs
                # 为了适配后续 rag_no_retrival.py 的 load_data 能够读取，我们把 ctxs 里的 text 提取出来作为支持事实
                
                # 提取 retrieved 文本作为支持文档 (这里模拟 hotpotqa/musique 的数据结构)
                retrieval_result = item.get("ctxs", [])[:top_n]
                paragraphs = []
                for ctx in retrieval_result:
                    title = ctx.get("title", "")
                    text = ctx.get("text", "")
                    # 模拟段落结构，如果是支持文档就加上
                    paragraphs.append({
                        "title": title,
                        "text": text,
                        "is_supporting": True  # 假设所有检索到的都视为可用的支持文档
                    })
                
                # 构建输出字典
                prompt_entry = {
                    "question": item.get("question", ""),
                    "answerKey": item.get("answers", []), # ARC 专用的答案字段
                    "choices": item.get("choices", {}),   # ARC 专用的选项字段
                    "paragraphs": paragraphs              # 新增的检索出来的上下文
                }
                prompts_buffer.append(prompt_entry)
            
            # 批量保存
            Path(prompt_save_path).parent.mkdir(parents=True, exist_ok=True)
            with jsonlines.open(prompt_save_path, 'a') as writer:
                for entry in prompts_buffer:
                    writer.write(entry)
            print(f"💾 已保存 {len(prompts_buffer)} 条带 Retrieval Context 的数据到 {prompt_save_path}")
            return [""] * len(items)  # 占位返回

        # 2. 批量生成
        print(f"Generating responses for {len(prompts)} prompts...")
        # 复用 get_generate
        raw_responses = get_generate(prompts, self.llm)
        print(f"Raw responses is {raw_responses}")

        # 3. 后处理
        final_results = []
        for resp in raw_responses:
            # 基础清洗
            cleaned = resp.split("\n\n")[0].replace("</s>", "").strip()
            
            # 针对特定任务的提取
            # 复用 postprocess_answers_closed
            choices_arg = "A B C D" if task in ["arc_c", "arc_easy"] else None
            final_out = postprocess_answers_closed(cleaned, task, choices=choices_arg)
            final_results.append(final_out)

        return final_results

    def run_batch(self, input_data: List[Dict], task: str = "qa", top_n: int = 5, save_prompts_only: bool = False, prompt_save_path: str = None):
        """
        端到端运行：输入数据 -> 检索 -> 生成
        """
        # 1. 提取 Query
        queries = [item["question"] for item in input_data]
        
        # 2. 检索
        print("--- Start Retrieval ---")
        ctxs_list = self.retrieve(queries, top_k=top_n)
        
        # 3. 将检索结果合并回 input_data
        for item, ctxs in zip(input_data, ctxs_list):
            item["ctxs"] = ctxs
            
        # 4. 生成 (或只保存 Prompt)
        print("--- Start Generation ---")
        answers = self.generate(input_data, task=task, top_n=top_n, save_prompts_only=save_prompts_only, prompt_save_path=prompt_save_path)
                
        # 5. 结果合并
        for item, ans in zip(input_data, answers):
            item["output"] = ans
            
        return input_data

# ==============================================================================
# 3. 使用示例 (Usage Example)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help="Path to ARC data")
    parser.add_argument('--save_prompts_only', action='store_true', help="Only retrieve and save prompts")
    parser.add_argument('--prompt_save_path', type=str, default="/home/vincent/hyper-simulation/data/mid_result/arc/arc_retrieved.jsonl")
    args = parser.parse_args()

    # 初始化 pipeline (请确保路径指向你实际的模型文件)
    rag = RAGPipeline(
        retriever_model_path="models/contriever-msmarco", # 需替换为实际路径
        passages_path="data/psgs_w100.tsv",             # 需替换为实际路径
        index_path="../index_hnsw/"                     # 需替换为实际路径
    )

    if args.data_path:
        # 如果传入了真实数据路径，则加载数据
        from hyper_simulation.question_answer.utils.load_data import load_data
        
        # ARC 数据需要以特定方式加载，这里借用已有的 load_data
        raw_data = load_data(args.data_path, "ARC")
        print(f"Loaded {len(raw_data)} samples from {args.data_path}")
        
        # 将原始数据转换为 rag.py 需要的格式
        test_data = []
        for item in raw_data:
            test_data.append({
                "question": item.get("question", ""),
                "choices": item.get("choices", {}),
                "answers": item.get("answerKey", [])  # 仅作参考保留
            })
            
        # 批量处理
        # 如果数据量很大，建议外层再套一个 batch 循环，这里直接跑
        batch_size = 100
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            print(f"Processing batch {i} to {i+len(batch)}...")
            rag.run_batch(
                batch, 
                task="arc_c", 
                top_n=5, 
                save_prompts_only=args.save_prompts_only,
                prompt_save_path=args.prompt_save_path
            )
        
        print("Done!")
    else:
        # 默认的模拟数据测试
        test_data = [
            {
                "id": 1,
                "question": "what is the capital of China?",
            },
            {
                "id": 2,
                "question": "Which material conducts heat best?",
                "choices": {"text": ["Wood", "Copper", "Plastic", "Glass"], "label": ["A", "B", "C", "D"]}
            }
        ]

        # 运行 PopQA 风格任务
        results = rag.run_batch(test_data[:1], task="qa")
        print(f"QA Result: {results[0]['output']}")

        # 运行 ARC 风格任务
        results_arc = rag.run_batch(test_data[1:], task="arc_c")
        print(f"ARC Result: {results_arc[0]['output']}")