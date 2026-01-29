"""
超图构建器：将QueryInstance转换为持久化超图文件
目录结构: {base_dir}/{dataset_name}/{instance_id}/{query.pkl, data_0.pkl, ...}
"""
import hashlib
from pathlib import Path
from typing import List, Optional, Union
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
from fastcoref import spacy_component
from hyper_simulation.query_instance import QueryInstance
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph
from hyper_simulation.hypergraph.dependency import Node, LocalDoc, Dependency
from hyper_simulation.hypergraph.combine import combine, calc_correfs_str

_NLP: Optional[spacy.Language] = None


def get_nlp() -> spacy.Language:
    global _NLP
    if _NLP is None:
        _NLP = spacy.load('en_core_web_trf')
        if 'fastcoref' not in _NLP.pipe_names:
            local_model_path = "/home/vincent/.cache/huggingface/hub/models--biu-nlp--lingmess-coref/snapshots/fa5d8a827a09388d03adbe9e800c7d8c509c3935"
            # _NLP.add_pipe('fastcoref', config={ 'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})
            _NLP.add_pipe('fastcoref', config={ 'model_architecture': 'LingMessCoref', 'model_path': local_model_path, 'device': 'cpu'})
    return _NLP

def text_to_doc(text: str) -> Doc:
    nlp = get_nlp()
    cfg = {"fastcoref": {'resolve_text': True}} if "fastcoref" in nlp.pipe_names else {}
    return nlp(text, component_cfg=cfg)

def doc_to_hypergraph(doc: Doc, text: str) -> LocalHypergraph:
    correfs = calc_correfs_str(doc) if hasattr(doc._, "coref_clusters") else set()
    spans_to_merge = combine(doc, correfs)
    with doc.retokenize() as retokenizer:
        for span in spans_to_merge:
            retokenizer.merge(span)
    nodes, roots = Node.from_doc(doc)
    local_doc = LocalDoc(doc)
    dep = Dependency(nodes, roots, local_doc)
    vertices, rels, id_map = (
        dep.solve_conjunctions().mark_pronoun_antecedents().mark_prefixes().mark_vertex().compress_dependencies().calc_relationships()
    )
    hypergraph = LocalHypergraph.from_rels(vertices, rels, id_map, local_doc)
    hypergraph.original_text = text
    return hypergraph

def text_to_hypergraph(text: str) -> LocalHypergraph:
    doc = text_to_doc(text)
    return doc_to_hypergraph(doc, text)

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
        query_hg = text_to_hypergraph(query_instance.query)
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


def build_hypergraph_batch(
    query_instances: List[QueryInstance],
    dataset_name: str = "hotpotqa",
    base_dir: Union[str, Path] = "data/hypergraph",
    force_rebuild: bool = False
) -> List[str]:
    instance_dirs = []
    for qi in tqdm(query_instances, desc="Building hypergraphs"):
        instance_dir = build_hypergraph_for_query_instance(
            qi, dataset_name, base_dir, force_rebuild
        )
        instance_dirs.append(instance_dir)
    return instance_dirs