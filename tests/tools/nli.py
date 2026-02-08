"""
NLI (Natural Language Inference) 工具 - 用于处理文本对的推理关系
"""


# 添加源代码路径

from hyper_simulation.component.nli import (
    get_nli_labels_batch,
    get_nli_label,
    get_nli_entailment_score_batch,
    get_nli_contradiction_score_batch
)


def process_text_pairs(pairs: list[tuple[str, str]]) -> dict:
    """
    处理多对文本，返回NLI的详细结果
    
    Args:
        pairs: 文本对列表，每个元素是 (premise, hypothesis) 的元组
    
    Returns:
        包含以下信息的字典：
        - labels: NLI标签列表 ('contradiction', 'entailment', 'neutral')
        - entailment_scores: 蕴含分数列表
        - contradiction_scores: 矛盾分数列表
        - details: 详细结果列表，每个元素包含 premise, hypothesis, label, scores
    """
    labels = get_nli_labels_batch(pairs)
    entailment_scores = get_nli_entailment_score_batch(pairs)
    contradiction_scores = get_nli_contradiction_score_batch(pairs)
    
    details = []
    for i, (premise, hypothesis) in enumerate(pairs):
        details.append({
            'premise': premise,
            'hypothesis': hypothesis,
            'label': labels[i],
            'entailment_score': entailment_scores[i],
            'contradiction_score': contradiction_scores[i]
        })
    
    return {
        'labels': labels,
        'entailment_scores': entailment_scores,
        'contradiction_scores': contradiction_scores,
        'details': details
    }


def print_nli_results(result: dict, verbose: bool = True) -> None:
    """
    打印NLI结果
    
    Args:
        result: process_text_pairs 返回的结果字典
        verbose: 是否打印详细信息
    """
    print(f"\n总共处理 {len(result['details'])} 对文本\n")
    
    if verbose:
        for i, detail in enumerate(result['details'], 1):
            print(f"文本对 {i}:")
            print(f"  前提: {detail['premise']}")
            print(f"  假设: {detail['hypothesis']}")
            print(f"  标签: {detail['label']}")
            print(f"  蕴含分数: {detail['entailment_score']:.4f}")
            print(f"  矛盾分数: {detail['contradiction_score']:.4f}")
            print()
    else:
        for i, label in enumerate(result['labels'], 1):
            print(f"{i}. {label}")


if __name__ == "__main__":
    # 示例：测试数据
    test_pairs = [
        ("200-lop", "300-lop"),
    ]
    
    print("=" * 60)
    print("NLI (Natural Language Inference) 演示")
    print("=" * 60)
    
    # 处理文本对
    result = process_text_pairs(test_pairs)
    
    # 打印详细结果
    print_nli_results(result, verbose=True)
    
    # 打印摘要
    print("=" * 60)
    print("结果摘要:")
    print(f"- 蕴含 (entailment): {result['labels'].count('entailment')} 对")
    print(f"- 矛盾 (contradiction): {result['labels'].count('contradiction')} 对")
    print(f"- 中性 (neutral): {result['labels'].count('neutral')} 对")
    print("=" * 60)
