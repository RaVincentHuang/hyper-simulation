from typing import Dict
from hyper_simulation.question_answer.vmdit.metrics import (
    exact_match_score, 
    metric_max_over_ground_truths,
    qa_f1_score,
    match
)

def evaluate_answer(prediction: str, ground_truth: list | str) -> Dict[str, float]:
    """
    评估生成的答案
    
    Args:
        prediction: 模型生成的答案
        ground_truth: 正确答案
    
    Returns:
        包含多个评估指标的字典
    """
    # 处理ground_truth可能是列表的情况
    if isinstance(ground_truth, list):
        ground_truths = ground_truth
    else:
        ground_truths = [ground_truth]
    
    # 计算Exact Match
    em_score = metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths
    )
    
    # 计算F1 score
    f1_score = max([qa_f1_score(prediction, gt) for gt in ground_truths])
    
    # 计算Match (部分匹配)
    match_score = match(prediction, ground_truths)
    
    return {
        "exact_match": em_score,
        "f1": f1_score,
        "match": match_score
    }

def postprocess_answer(answer: str) -> tuple[str, str, bool]:
    """
    从 LLM 输出中提取最终答案（增强版）
    
    Returns:
        (processed_answer, parse_status, is_fallback)
        - parse_status: "parsed_final_answer" | "parsed_last_line" | "fallback_truncated" | "empty_input"
        - is_fallback: 是否使用了兜底策略
    """
    import re
    import logging
    logger = logging.getLogger(__name__)
    
    # ========== 1. 空答案保护 ==========
    if not answer:
        return "unanswerable", "empty_input", True
    
    # ========== 2. 基础清理 ==========
    answer = answer.replace("</s>", "").replace("</think>", "").strip()
    
    # ========== 3. 多次匹配 ### Final Answer:（关键修复）==========
    final_answer_pattern = r"###\s*Final\s*Answer:\s*(.+?)(?:\n|$)"
    extracted = answer
    parse_status = "fallback_truncated"
    match_count = 0
    
    # 循环匹配，直到没有更多 "### Final Answer:" 前缀
    while True:
        match = re.search(final_answer_pattern, extracted, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            match_count += 1
            parse_status = "parsed_final_answer"
        else:
            break
    
    # 如果至少匹配到一次，返回结果
    if match_count > 0:
        cleaned = extracted.strip(" .,;:!?\"'")
        if cleaned:
            return cleaned, parse_status, False
    
    # ========== 4. 次选：最后一行短答案 ==========
    lines = answer.strip().split('\n')
    exclude_keywords = ['step', 'reason', 'explain', 'note', 'context', 
                       'paragraph', 'think', 'analysis', 'conclusion']
    
    for line in reversed(lines):
        line = line.strip()
        if (line and 
            len(line) < 100 and 
            not any(k in line.lower() for k in exclude_keywords) and
            not line.startswith('#')):
            cleaned = line.strip(" .,;:!?\"'")
            if cleaned:
                return cleaned, "parsed_last_line", False
    
    # ========== 5. 兜底 + 警告日志 ==========
    logger.warning(f"⚠️ Could not parse answer (match_count={match_count}), using fallback. Output preview: {answer[:100]}...")
    
    cleaned = answer.strip(" .,;:!?\"'")[:200]
    return cleaned if cleaned else "unanswerable", "fallback_truncated", True
