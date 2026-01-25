# contradoc.py
import ast

from langchain_ollama import ChatOllama

from hyper_simulation.llm.chat_completion import get_generate

def construct_conflict_prompt(doc_a, doc_b):
    """
    构造用于检测两个文档间事实冲突的 Prompt。
    基于 CONTRADOC 论文 Appendix D 的 'Judge then Find' 任务逻辑进行改造。
    """
    prompt = f"""
You are an expert in detecting logical and factual inconsistencies between documents.
Your task is to determine whether **Document A** and **Document B** contain any factual contradictions.

A contradiction occurs when Document A states a fact that is mutually exclusive with a fact stated in Document B (e.g., conflicting numbers, dates, locations, actions, or specific details about the same entity).

**Instructions:**
1. Read both documents carefully.
2. Determine if a contradiction exists (Yes/No).
3. If **Yes**: Provide the specific evidence as a Python list containing exactly two strings: 
   - The sentence from Document A.
   - The contradictory sentence from Document B.
4. If **No**: Provide an empty list.

--- Document A ---
{doc_a}

--- Document B ---
{doc_b}
------------------

**Response Format:**
Please strictly follow the format below (do not provide explanations, only the format):

Judgment: yes OR no
Evidence: ["sentence_from_doc_A", "sentence_from_doc_B"] OR []
"""
    return prompt

def parse_model_response(response_content):
    """
    解析模型返回的文本，提取 Judgment 和 Evidence List。
    使用 ast.literal_eval 安全地解析 Python 列表格式的字符串。
    """
    if not response_content:
        return "error", []

    lines = response_content.strip().split('\n')
    judgment = "no"
    evidence = []

    for line in lines:
        clean_line = line.strip()
        
        # 提取 Judgment
        if clean_line.lower().startswith("judgment:"):
            if "yes" in clean_line.lower():
                judgment = "yes"
        
        # 提取 Evidence List
        if clean_line.lower().startswith("evidence:"):
            try:
                # 去掉前缀 "Evidence:"，提取列表字符串部分
                list_str = clean_line.split(":", 1)[1].strip()
                # 尝试解析
                parsed_list = ast.literal_eval(list_str)
                if isinstance(parsed_list, list):
                    evidence = parsed_list
            except Exception as e:
                print(f"[Warn] Failed to parse evidence list directly: {e}")
                # 简单的降级处理：如果不符合 list 格式，返回原始字符串以便调试
                evidence = list_str

    return judgment, evidence

def detect_conflict(doc_a, doc_b, model: ChatOllama):
    """
    主函数：执行文档冲突检测
    """
    # 1. 构造 Prompt
    print("Constructing prompt...")
    prompt = construct_conflict_prompt(doc_a, doc_b)
    
    # 2. 调用 LLM
    print(f"Calling LLM ({model})...")
    response_content = get_generate([prompt], model=model) # openai, vllm, transformers, ollama
    
    if not response_content:
        return {"error": "LLM request failed"}

    # 3. 解析结果
    judgment, evidence = parse_model_response(response_content[0])
    
    return {
        "has_conflict": judgment == "yes",
        "evidence": evidence,
        "raw_response": response_content
    }

# ==========================================
# 示例运行逻辑
# ==========================================
if __name__ == "__main__":
    # 示例数据
    document_1 = """
    Project Alpha was officially launched on January 15, 2023. 
    The total budget allocated for the first phase was $2 million.
    The project manager is Sarah Connor.
    """
    
    document_2 = """
    According to the Q1 report, Project Alpha started on February 1, 2023.
    The budget for phase one is recorded as $2 million.
    Sarah Connor is leading the team.
    """
    
    model = ChatOllama(model='qwen2.5:14b', temperature=0.8, top_p=0.95)
    
    print("--- Starting Detection ---")
    result = detect_conflict(document_1, document_2, model=model)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nDetection Result:")
        print(f"Has Conflict: {result['has_conflict']}")
        print(f"Evidence: {result['evidence']}")
        
        # 打印原始回复供调试
        # print(f"\nRaw LLM Response:\n{result['raw_response']}")