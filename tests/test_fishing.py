import spacy
from spacy.tokens import Span

def main():
    # 1. 加载模型
    print(">>> 正在加载 en_core_web_trf 模型...")
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        print("错误: 请先安装模型 python -m spacy download en_core_web_trf")
        return

    # 2. 添加 Entity Linker
    # 注意：第一次运行需要先下载知识库: python -m spacy_entity_linker "download_knowledge_base"
    try:
        nlp.add_pipe("entityLinker", last=True)
    except Exception as e:
        print(f"错误: 加载 entityLinker 失败。{e}")
        return

    text = "Scholar Nilsson delivered a keynote at Stockholmsmässan on August. He also participated in roundtable discussions. That day, the venue hosted an AI ethics seminar, which featured his keynote and discussions."
    print(f"\n>>> 处理文本: \"{text}\"")
    
    doc = nlp(text)

    print("\n" + "="*100)
    print(f"{'Entity (TRF)':<20} {'Label':<10} {'Wikidata ID':<15} {'Description'}")
    print("-" * 100)
    
    for ent in doc._.linkedEntities:
        print(f"Linked Entity: '{ent.get_span()}', Label: {ent.get_label()}, Description: {ent.get_description()}")

    # for ent in doc.ents:
    #     best_qid = "N/A"
    #     best_desc = "-"
        
    #     # --- 核心修改部分 ---
    #     matches = []
    #     for linked_ent in doc._.linkedEntities:
    #         # 修复 1: 去掉 doc 参数，调用 get_span()
    #         # 这会返回一个 SpanInfo 对象 (包含 start/end token 索引)
    #         linked_span_info = linked_ent.get_span()
            
    #         # 修复 2: 使用基于 Token 的重叠检测函数
    #         if spans_overlap_token(ent, linked_span_info):
    #             matches.append(linked_ent)
    #     # -------------------

    #     if matches:
    #         best_match = matches[0]
    #         best_qid = f"Q{best_match.get_id()}"
    #         best_desc = best_match.get_description()

    #     print(f"{ent.text:<20} {ent.label_:<10} {best_qid:<15} {best_desc}")
        
    # print("="*100)

def spans_overlap_token(span1, span2_info) -> bool:
    """
    判断两个 span 在 Token 层面是否重叠。
    span1: spaCy 的 Span 对象 (有 .start, .end)
    span2_info: EntityLinker 的 SpanInfo 对象 (有 .start, .end)
    """
    # 只要一个 span 的开始在另一个的结束之前，且结束在另一个的开始之后，就是重叠
    return (span1.start < span2_info.end) and (span1.end > span2_info.start)

if __name__ == "__main__":
    main()