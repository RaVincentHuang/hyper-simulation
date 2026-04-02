"""
超图构建器：将QueryInstance转换为持久化超图文件
目录结构: {base_dir}/{dataset_name}/{instance_id}/{query.pkl, data_0.pkl, ...}
"""
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Union
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
from fastcoref import spacy_component
from hyper_simulation.query_instance import QueryInstance
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph

from hyper_simulation.hypergraph.dependency import Node, LocalDoc, Dependency
from hyper_simulation.hypergraph.combine import combine, calc_correfs_str, combine_links
from hyper_simulation.utils.clean import clean_text_for_spacy
from hyper_simulation.hypergraph.abstraction import TokenEntityAdder
from hyper_simulation.utils.log import getLogger

from hyper_simulation.hypergraph.corref import CorrefCluster, mark_corref
from spacy.symbols import ORTH

_NLP: Optional[spacy.Language] = None

def normalize_special_chars(text: str) -> str:
    """
    处理字符串中的特殊转义字符，将它们全部转化为空格
    
    Args:
        text: 输入字符串
    
    Returns:
        处理后的字符串，所有特殊转义字符都被替换为空格
    
    Examples:
        >>> normalize_special_chars("Hello\\nWorld\\tTest")
        "Hello World Test"
        >>> normalize_special_chars("Line1\\rLine2")
        "Line1 Line2"
    """
    # 替换常见的转义字符为空格
    special_chars = {
        '\\n': ' ',   # 换行符
        '\\r': ' ',   # 回车符
        '\\t': ' ',   # 制表符
        '\\v': ' ',   # 垂直制表符
        '\\f': ' ',   # 换页符
        '\\b': ' ',   # 退格符
        '\\a': ' ',   # 响铃符
        '\n': ' ',    # 实际换行符
        '\r': ' ',    # 实际回车符
        '\t': ' ',    # 实际制表符
        '\v': ' ',    # 实际垂直制表符
        '\f': ' ',    # 实际换页符
        '\b': ' ',    # 实际退格符
        '\a': ' ',    # 实际响铃符
    }
    
    result = text
    for char, replacement in special_chars.items():
        result = result.replace(char, replacement)
    
    # 合并多个连续空格为一个空格
    result = re.sub(r' +', ' ', result)
    
    return result.strip()


def get_nlp() -> spacy.Language:
    global _NLP
    if _NLP is None:
        _NLP = spacy.load('en_core_web_trf')
        if 'fastcoref' not in _NLP.pipe_names:
            local_model_path = "/home/vincent/.cache/huggingface/hub/models--biu-nlp--lingmess-coref/snapshots/fa5d8a827a09388d03adbe9e800c7d8c509c3935"
            _NLP.add_pipe('fastcoref', config={ 'model_architecture': 'LingMessCoref', 'model_path': local_model_path, 'device': 'cpu'})
    
    # Tokenizer: special cases
    # R1: I. => I .  (防止把 "I." 误识别为一个token，导致 "I" 的lemma无法正确识别为 "I")
    _NLP.tokenizer.add_special_case("I.", [{ORTH: "I"}, {ORTH: "."}])
    ROMAN_NUMERALS = ["II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII"]
    for numeral in ROMAN_NUMERALS:
        _NLP.tokenizer.add_special_case(f"{numeral}.", [{ORTH: numeral}, {ORTH: "."}])
    return _NLP

def text_to_doc(text: str) -> Doc:
    """
    将文本转换为 spaCy Doc 对象，带 fastcoref 错误处理
    """
    logger = getLogger(__name__)
    
    nlp = get_nlp()
    
    # ✅ 尝试使用 fastcoref
    if "fastcoref" in nlp.pipe_names:
        try:
            cfg = {"fastcoref": {'resolve_text': True}}
            doc = nlp(text, component_cfg=cfg)
            
            # ✅ 验证 coref_clusters 格式是否正确
            if hasattr(doc._, "coref_clusters") and doc._.coref_clusters:
                for cluster in doc._.coref_clusters:
                    if cluster is None:
                        raise ValueError("coref_clusters contains None")
                    for span in cluster:
                        if span is None or not isinstance(span, (list, tuple)) or len(span) != 2:
                            raise ValueError(f"Invalid span in coref_clusters: {span}")
            
            return doc
            
        except Exception as e:
            # ✅ fastcoref 失败，降级为不使用 coref
            logger.warning(
                f"[text_to_doc] fastcoref failed for text (len={len(text)}): {type(e).__name__}: {e}, "
                f"falling back to no coref"
            )
    
    # ✅ 兜底：不使用 fastcoref
    doc = nlp(text)
    return doc

def doc_to_hypergraph(doc: Doc, text: str, is_query: bool = False) -> LocalHypergraph:
    correfs = calc_correfs_str(doc) if hasattr(doc._, "coref_clusters") else set()
    
    abstractor = TokenEntityAdder("qwen_ontology_mapping.json")
    links_to_merge = combine_links(doc)
    with doc.retokenize() as retokenizer:
        for link in links_to_merge:
            retokenizer.merge(link)
    corref_clusters = CorrefCluster.from_doc(doc)
    spans_to_merge = combine(doc, correfs, is_query=is_query, corefs_clusters=corref_clusters)
    abstractor.set_entity_from_spans(spans_to_merge, doc)
    with doc.retokenize() as retokenizer:
        for span in spans_to_merge:
            if span.start < span.end:
                retokenizer.merge(span)
    corref_clusters = CorrefCluster.update_by_doc(corref_clusters, doc)
    nodes, roots = Node.from_doc(doc, abstractor)
    nodes = mark_corref(nodes, corref_clusters)
    local_doc = LocalDoc(doc)
    dep = Dependency(nodes, roots, local_doc, is_query=is_query)
    vertices, rels, id_map = (
        dep.solve_conjunctions().mark_pronoun_antecedents().mark_prefixes().mark_vertex().compress_dependencies().calc_relationships()
    )
    hypergraph = LocalHypergraph.from_rels(vertices, rels, id_map, local_doc)
    hypergraph.original_text = text
    return hypergraph

def text_to_hypergraph(text: str, is_query: bool = False) -> LocalHypergraph:
    text = clean_text_for_spacy(text)
    # print(f"\n[Original Text]:\n{text}\n")
    doc = text_to_doc(text)
    return doc_to_hypergraph(doc, text, is_query=is_query)

def generate_instance_id(query: str) -> str:
    normalized = ''.join(query.split()).lower()
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]


def build_hypergraph_for_query_instance(
    query_instance: QueryInstance,
    dataset_name: str = "hotpotqa",
    base_dir: Union[str, Path] = "data/hypergraph",
    force_rebuild: bool = False
) -> str:
    instance_id = generate_instance_id(query_instance.query)
    instance_dir = Path(base_dir) / dataset_name / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = instance_dir / "metadata.json"
    if metadata_path.exists() and not force_rebuild:
        return str(instance_dir.resolve())
    
    # 构建query超图
    query_path = instance_dir / "query.pkl"
    if not query_path.exists() or force_rebuild:
        query_hg = text_to_hypergraph(query_instance.query, is_query=True)
        query_hg.save(str(query_path))
    
    # 构建data超图
    for idx, doc_text in enumerate(query_instance.data):
        data_path = instance_dir / f"data_{idx}.pkl"
        if data_path.exists() and not force_rebuild:
            continue
        data_hg = text_to_hypergraph(doc_text)
        data_hg.save(str(data_path))
    
    # 保存元数据
    metadata = {
        "instance_id": instance_id,
        "num_data": len(query_instance.data),
        "data_lengths": [len(d) for d in query_instance.data],
    }
    with open(metadata_path, 'w') as f:
        import json
        json.dump(metadata, f)
    
    return str(instance_dir.resolve())

def test_build_hypergraph_for_query_instance(query_instance: QueryInstance) -> tuple[LocalHypergraph, List[LocalHypergraph]]:
    query_hg = text_to_hypergraph(query_instance.query, is_query=True)
    data_list = [text_to_hypergraph(doc_text, is_query=False) for doc_text in query_instance.data]
    with open("missing.txt", "a") as f:
        for h in [query_hg] + data_list:
            for v in h.vertices:
                if not v.is_noun():
                    continue
                if v.has_entity():
                    continue
                f.write(f"{v.text().strip().lower()}\n")
    return query_hg, data_list

def build_hypergraph_batch(
    query_instances: List[QueryInstance],
    dataset_name: str = "hotpotqa",
    base_dir: Union[str, Path] = "data/hypergraph",
    force_rebuild: bool = False
) -> List[str]:
    instance_dirs = []
    for qi in tqdm(query_instances, desc="Building hypergraphs", position=1, leave=False):
        instance_dir = build_hypergraph_for_query_instance(
            qi, dataset_name, base_dir, force_rebuild
        )
        instance_dirs.append(instance_dir)
    return instance_dirs