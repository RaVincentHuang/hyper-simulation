import hashlib
from pathlib import Path
from typing import List, Tuple, Set, Dict

from hyper_simulation.query_instance import QueryInstance
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.dependency import Entity, Pos, Dep
from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.embedding import get_embedding_batch, cosine_similarity
from hyper_simulation.utils.log import getLogger
from tqdm import tqdm
from hyper_simulation.utils.log import current_query_id
# from tqdm.contrib.logging import logging_redirect_tqdm

def generate_instance_id(query: str) -> str:
    normalized = ''.join(query.split()).lower()
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]


def load_hypergraphs_for_instance(
    query_instance: QueryInstance,
    dataset_name: str = "hotpotqa",
    base_dir: str = "data/hypergraph"
) -> Tuple[LocalHypergraph, List[LocalHypergraph]]:
    instance_id = generate_instance_id(query_instance.query)
    current_query_id.set(instance_id)
    instance_dir = Path(base_dir) / dataset_name / instance_id
    
    if not instance_dir.exists():
        raise FileNotFoundError(
            f"Hypergraphs not found. Run build_hypergraph_batch first.\nDirectory: {instance_dir}"
        )
    
    query_hg = LocalHypergraph.load(str(instance_dir / "query.pkl"))
    
    data_hgs = []
    for idx in (range(len(query_instance.data))):
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


def consistent_detection(
    query: QueryInstance,
    query_hg: LocalHypergraph,
    data_hg: LocalHypergraph,
    query_text: str,
    data_text: str,
    distance_threshold: float = 0.25
) -> Tuple[bool, str]:
    """
    严格遵循一致性检测：
    当 δ > θ 时，验证 hyper simulation 是否满足 ∀u∈V_q, ∃v∈V_d: (u,v)∈Π
    """
    consistent_logger = getLogger("consistency")
    consistent_logger.debug("Enter the consistent detection")

    # Step 1: 计算向量距离
    distance = get_distance(query_text, data_text)
    consistent_logger.info(f"Compute cosine distance: {distance:.4f}, threshold={distance_threshold}")

    # 距离足够近 → 自动一致（注意：此处原逻辑 return False 表示“无矛盾”，但语义易混淆）
    if distance <= distance_threshold:
        evidence = f"[CONSISTENT] Distance={distance:.4f} ≤ threshold={distance_threshold}"
        consistent_logger.info(evidence)
        return False, evidence  # False = no contradiction

    # Step 2: 执行 hyper simulation
    consistent_logger.debug("Running hyper_simulation...")
    simulation, q_vertices_map, d_vertices_map = compute_hyper_simulation(query_hg, data_hg)

    # Step 3: 获取 critical vertices
    critical_q_vertices = [v for v in query_hg.vertices if is_critical_vertex(v)]
    consistent_logger.info(f"Critical Q vertices to cover: {len(critical_q_vertices)}")
    for v in critical_q_vertices:
        consistent_logger.info(f"  • Q{v.id}: '{v.text()}'")

    if not critical_q_vertices:
        evidence = f"[CONSISTENT] No critical vertices to cover (distance={distance:.4f} > threshold)"
        consistent_logger.info(evidence)
        return False, evidence

    # Step 4: 验证全覆盖
    evidence_lines = []
    has_contradiction = False

    # 构建 reverse map: sim_q_id → q_vertex
    sim_id_to_q_vertex = {sim_id: q_vertices_map[sim_id] for sim_id in q_vertices_map}

    for q_vertex in critical_q_vertices:
        matched = False
        # 查找该 q_vertex 对应的 sim_id
        target_sim_id = None
        for sim_id, v in q_vertices_map.items():
            if v.id == q_vertex.id:
                target_sim_id = sim_id
                break

        if target_sim_id is not None and target_sim_id in simulation:
            sim_d_ids = simulation[target_sim_id]
            if len(sim_d_ids) > 0:
                matched = True

        if not matched:
            has_contradiction = True
            evidence_lines.append(f"Q vertex unmatched in D: '{q_vertex.text()}' (ID={q_vertex.id})")

    # 全量日志输出（无截断）
    if evidence_lines:
        consistent_logger.info("Unmatched critical vertices in Q:")
        for line in evidence_lines:
            consistent_logger.info(f"  • {line}")
    else:
        consistent_logger.info("All critical Q vertices are matched in D.")

    # 生成最终证据
    if has_contradiction:
        evidence = (
            f"[CONTRADICTION] Distance={distance:.4f} > threshold={distance_threshold}\n"
            + "\n".join(f"  • {line}" for line in evidence_lines)
        )
    else:
        evidence = (
            f"[CONSISTENT] Distance={distance:.4f} > threshold but structural coverage satisfied\n"
            f"  ✓ All {len(critical_q_vertices)} critical Q vertices matched in D via hyper simulation"
        )

    consistent_logger.info(evidence)
    return has_contradiction, evidence


def query_fixup(query: QueryInstance, dataset_name: str = "hotpotqa") -> QueryInstance:
    """
    基于hyper simulation的一致性修复
    """
    query_hg, data_hgs = load_hypergraphs_for_instance(query, dataset_name)
    hg_logger = getLogger("hypergraph", level="DEBUG")
    hg_logger.debug(f"=== Query Text===")
    hg_logger.debug(f"{query.query}")
    hg_logger.debug(f"=== Query Hypergraph===")
    query_hg.log_summary(hg_logger)
    for i, d in enumerate(data_hgs):
        if d:
            hg_logger.debug(f"=== Query Text #{i}===")
            hg_logger.debug(f"{query.data[i]}")
            hg_logger.debug(f"=== Data Hypergraph #{i} ===")
            d.log_summary(hg_logger)
    
    fixed_data = []
    for doc_text, data_hg in tqdm(zip(query.data, data_hgs), desc='\tHyper Simulation for Query.', leave=True):
        if data_hg is None:
            fixed_data.append(doc_text)
            continue
        
        has_contradiction, evidence = consistent_detection(
            query, query_hg, data_hg, query.query, doc_text
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