import hashlib
from pathlib import Path
from typing import List, Tuple, Set, Dict

from hyper_simulation.query_instance import QueryInstance
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex, Hyperedge
from hyper_simulation.hypergraph.linguistic import Entity, Pos, Dep
from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.embedding import get_embedding_batch, cosine_similarity
from hyper_simulation.utils.log import getLogger
from tqdm import tqdm
from hyper_simulation.utils.log import current_query_id
from hyper_simulation.hypergraph.union import MultiHopFusion
from hyper_simulation.component.nli import init_nli_model
from hyper_simulation.component.embedding import init_embedding_model
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

if __name__ == "__main__":
    path: str = 'logs/debugs'
    query_path = f"{path}/query_hypergraph.pkl"
    query_hg = LocalHypergraph.load(query_path)
    data_hgs: list[LocalHypergraph | None] = []
    
    query, valid_texts, _ = load_musique_case(f"/home/vincent/.dataset/musique/x.json")
    
    for i in range(20):  # 假设最多有 20 个相关
        data_path = f"{path}/data_hypergraph{i}.pkl"
        if Path(data_path).exists():
            data_hgs.append(LocalHypergraph.load(data_path))
        else:
            data_hgs.append(None)
    
    # Dump query hypergraph
    print("Query Hypergraph:")
    print(query)
    print(f"Vertices ({len(query_hg.vertices)}):")
    for i, v in enumerate(query_hg.vertices):
        if v.is_query():
            print(f"  - [{i}] '{v.text()}' TYPE: {v.query_type()}")
            continue
        print(f"  - [{i}] '{v.text()}' TYPE: {v.type()}")
    # for i, v in enumerate(query_hg.vertices):
    #     if v.is_query() or v.is_verb() or v.is_virtual() or v.is_adjective() or v.is_adverb():
    #         continue
    #     if not v.type():
    #         print(f"[{i}] '{v.text()}'")
    #         for node in v.nodes:
    #             print(f"    - '{node.text}' (pos={node.pos.name})")
    
    for idx, data_hg in enumerate(data_hgs):
        if data_hg is None:
            print(f"\nData Hypergraph {idx}: MISSING")
            continue
        print(f"\nData Hypergraph {idx}:")
        print(valid_texts[idx])
        print(f"Vertices ({len(data_hg.vertices)}):")
        for i, v in enumerate(data_hg.vertices):
            print(f"  - [{i}] '{v.text()}' TYPE: {v.type()}")
        # for i, v in enumerate(data_hg.vertices):
        #     if v.is_query() or v.is_verb() or v.is_virtual() or v.is_adjective() or v.is_adverb():
        #         continue
        #     if not v.type():
        #         print(f"[{i}] '{v.text()}'")
        #         for node in v.nodes:
        #             print(f"    - '{node.text}' (pos={node.pos.name})")
    