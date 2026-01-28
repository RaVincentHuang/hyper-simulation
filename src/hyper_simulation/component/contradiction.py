import hashlib
from pathlib import Path
from typing import List, Tuple, Set, Dict

from hyper_simulation.query_instance import QueryInstance
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.dependency import Entity, Pos, Dep
from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.embedding import get_embedding_batch, cosine_similarity


def generate_instance_id(query: str) -> str:
    normalized = ''.join(query.split()).lower()
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]


def load_hypergraphs_for_instance(
    query_instance: QueryInstance,
    dataset_name: str = "hotpotqa",
    base_dir: str = "data/hypergraph"
) -> Tuple[LocalHypergraph, List[LocalHypergraph]]:
    instance_id = generate_instance_id(query_instance.query)
    instance_dir = Path(base_dir) / dataset_name / instance_id
    
    if not instance_dir.exists():
        raise FileNotFoundError(
            f"Hypergraphs not found. Run build_hypergraph_batch first.\nDirectory: {instance_dir}"
        )
    
    query_hg = LocalHypergraph.load(str(instance_dir / "query.pkl"))
    
    data_hgs = []
    for idx in range(len(query_instance.data)):
        data_path = instance_dir / f"data_{idx}.pkl"
        if data_path.exists():
            try:
                data_hgs.append(LocalHypergraph.load(str(data_path)))
            except:
                data_hgs.append(None)
        else:
            data_hgs.append(None)
    
    return query_hg, data_hgs


def is_critical_vertex(vertex: Vertex) -> bool:
    """定义需强制匹配的关键顶点"""
    if any(e != Entity.NOT_ENTITY for e in vertex.ents):
        return True
    dep = vertex.dep()
    if dep in {Dep.nsubj, Dep.nsubjpass, Dep.dobj, Dep.iobj, Dep.pobj}:
        return True
    if vertex.pos_equal(Pos.NOUN) or vertex.pos_equal(Pos.PROPN):
        return True
    return False


def get_distance(text1: str, text2: str) -> float:
    emb1 = get_embedding_batch([text1])[0]
    emb2 = get_embedding_batch([text2])[0]
    return 1.0 - cosine_similarity(emb1, emb2)


def detect_contradiction_via_simulation(
    query_hg: LocalHypergraph,
    data_hg: LocalHypergraph,
    query_text: str,
    data_text: str,
    distance_threshold: float = 0.25
) -> Tuple[bool, str]:
    """
    严格遵循矛盾检测：
    当 δ > θ 时，验证 hyper simulation 是否满足 ∀u∈V_q, ∃v∈V_d: (u,v)∈Π
    """
    # Step 1: 计算向量距离
    distance = get_distance(query_text, data_text)
    
    # 距离足够近 → 自动一致
    if distance <= distance_threshold:
        return False, f"[CONSISTENT] Distance={distance:.3f} ≤ threshold={distance_threshold}"
    
    # Step 2: 执行hyper simulation
    simulation, q_vertices_map, d_vertices_map = compute_hyper_simulation(query_hg, data_hg)
    
    # Step 3: 验证全覆盖条件
    evidence_lines = []
    has_contradiction = False
    
    critical_q_vertices = [v for v in query_hg.vertices if is_critical_vertex(v)]
    
    for q_vertex in critical_q_vertices:
        # 检查该顶点是否在simulation中有匹配
        matched = False
        for sim_q_id, sim_d_ids in simulation.items():
            # 通过ID映射找到对应的原始顶点
            if q_vertices_map[sim_q_id].id == q_vertex.id and len(sim_d_ids) > 0:
                matched = True
                break
        
        if not matched:
            has_contradiction = True
            evidence_lines.append(f"MISSING: '{q_vertex.text()}'")
    
    # 生成证据（严格基于hyper simulation结果）
    if has_contradiction:
        evidence = (
            f"[CONTRADICTION] Distance={distance:.3f} > threshold={distance_threshold}\n"
            + "\n".join(f"  • {line}" for line in evidence_lines)
        )
    else:
        evidence = (
            f"[CONSISTENT] Distance={distance:.3f} > threshold but structural coverage satisfied\n"
            f"  ✓ All {len(critical_q_vertices)} critical vertices matched via hyper simulation"
        )
    
    return has_contradiction, evidence


def query_fixup(query: QueryInstance, dataset_name: str = "hotpotqa") -> QueryInstance:
    """
    基于hyper simulation的一致性修复
    """
    query_hg, data_hgs = load_hypergraphs_for_instance(query, dataset_name)
    
    fixed_data = []
    for doc_text, data_hg in zip(query.data, data_hgs):
        if data_hg is None:
            fixed_data.append(doc_text)
            continue
        
        has_contradiction, evidence = detect_contradiction_via_simulation(
            query_hg, data_hg, query.query, doc_text
        )
        
        if has_contradiction:
            fixed_doc = (
                f"{doc_text}\n\n"
                f"[INCONSISTENT DETECTED - USE WITH CAUTION]\n"
                f"Evidence:\n{evidence}"
            )
        else:
            fixed_doc = doc_text
        
        fixed_data.append(fixed_doc)
    
    query.fixed_data = fixed_data
    return query