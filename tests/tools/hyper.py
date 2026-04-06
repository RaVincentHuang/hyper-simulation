import hashlib
from pathlib import Path
from typing import List, Tuple, Set, Dict

from hyper_simulation.query_instance import QueryInstance
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.linguistic import Entity, Pos, Dep
from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.embedding import get_embedding_batch, cosine_similarity
from hyper_simulation.utils.log import getLogger
from tqdm import tqdm
from hyper_simulation.utils.log import current_query_id
from hyper_simulation.hypergraph.union import MultiHopFusion
from hyper_simulation.component.nli import init_nli_model
from hyper_simulation.component.embedding import init_embedding_model
from hyper_simulation.question_answer.decompose import decompose_question
import json
import time


def load_musique_case(json_path: str) -> tuple[str, list[str], list[int]]:
    """
    Load one MuSiQue item and map fields:
    - query <- question
    - dataset <- paragraphs[*].paragraph_text
    - supports <- question_decomposition[*].paragraph_support_idx
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"MuSiQue input file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    item = raw[0] if isinstance(raw, list) else raw
    if not isinstance(item, dict):
        raise ValueError("Expected a JSON object (or a list whose first element is object).")

    query = item.get("question", "")

    paragraphs = item.get("paragraphs", [])
    dataset = [p.get("paragraph_text", "") for p in paragraphs if isinstance(p, dict)]

    supports_set: set[int] = set()
    for step in item.get("question_decomposition", []):
        if not isinstance(step, dict):
            continue
        paragraph_idx = step.get("paragraph_support_idx")
        if paragraph_idx is None:
            continue
        try:
            supports_set.add(int(paragraph_idx))
        except (TypeError, ValueError):
            continue

    supports = sorted(supports_set)
    return query, dataset, supports

def query_fixup(path: str = 'logs/debugs'):
    """
    基于 hyper simulation 的一致性修复
    """
    # Load query and data hypergraphs
    # f"{path}/query_hypergraph.pkl"
    # f"{path}/data_hypergraph0.pkl", f"{path}/data_hypergraph1.pkl", ...
    query_path = f"{path}/query_hypergraph.pkl"
    query_hg = LocalHypergraph.load(query_path)
    data_hgs = []
    for i in range(20):  # 假设最多有 20 个相关
        data_path = f"{path}/data_hypergraph{i}.pkl"
        if Path(data_path).exists():
            data_hgs.append(LocalHypergraph.load(data_path))
        else:
            data_hgs.append(None)
    
    fusion = MultiHopFusion()
        
    valid_indices = [i for i, hg in enumerate(data_hgs) if hg is not None]
    valid_hgs = [data_hgs[i] for i in valid_indices]
    
    query_text, valid_texts, _ = load_musique_case(f"/home/vincent/.dataset/musique/x.json")
    
    decomposes = decompose_question(query_text, query_hg)
    print("\n=== Decomposed Sub-questions ===")
    for i, (sub_q, vertex_ids) in enumerate(decomposes):
        print(f"[{i + 1}] {sub_q} (related vertex IDs: {sorted(vertex_ids)})")
    
        # merge + reverse trace
        
    context = fusion.process(query_hg, valid_hgs, valid_texts)
    
    # print("\n=== Fusion Result ===")
    # print(context)

if __name__ == "__main__":
    init_nli_model()
    init_embedding_model()
    time1 = time.time()
    query_fixup()
    time2 = time.time()
    print(f"Total time taken: {time2 - time1:.2f} seconds")
