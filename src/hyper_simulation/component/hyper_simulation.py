import time
from typing import Dict, List, Set, Tuple
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from simulation import Hypergraph as SimHypergraph, Hyperedge as SimHyperedge, Node, Delta, DMatch
from hyper_simulation.component.nli import get_nli_labels_batch
from hyper_simulation.component.semantic_cluster import get_semantic_cluster_pairs, get_d_match
from hyper_simulation.hypergraph.dependency import Pos
import warnings
from tqdm import tqdm
from hyper_simulation.utils.log import getLogger
import logging
def convert_local_to_sim(
    local_hg: LocalHypergraph,
) -> Tuple[SimHypergraph, Dict[int, str], Dict[int, Vertex], Dict[int, List[SimHyperedge]], Dict[Vertex, int]]:
    """转换LocalHypergraph → SimHypergraph，返回Sim ID空间映射"""
    sim_hg = SimHypergraph()
    vertex_id_map: Dict[int, int] = {}
    node_text: Dict[int, str] = {}
    sim_id_to_vertex: Dict[int, Vertex] = {}
    node_to_edges: Dict[int, List[SimHyperedge]] = {}
    vertex_to_sim_id: Dict[Vertex, int] = {}
    
    for idx, vertex in enumerate(sorted(local_hg.vertices, key=lambda v: v.id)):
        sim_hg.add_node(vertex.text())
        vertex_id_map[vertex.id] = idx
        node_text[idx] = vertex.text()
        sim_id_to_vertex[idx] = vertex
        vertex_to_sim_id[vertex] = idx
    
    edge_id = 0
    for local_edge in local_hg.hyperedges:
        node_ids = {vertex_id_map[v.id] for v in local_edge.vertices if v.id in vertex_id_map}
        if not node_ids:
            continue
        sim_edge = SimHyperedge(node_ids, local_edge.desc, edge_id)
        sim_hg.add_hyperedge(sim_edge)
        for nid in node_ids:
            node_to_edges.setdefault(nid, []).append(sim_edge)
        edge_id += 1
    
    return sim_hg, node_text, sim_id_to_vertex, node_to_edges, vertex_to_sim_id

def compute_allowed_pairs(
    query_vertices: Dict[int, Vertex],
    data_vertices: Dict[int, Vertex]
) -> Set[Tuple[int, int]]:
    """
    宽松语义允许性判定：仅排除 contradiction
    理论依据：type_same 应定义为“语义不冲突”，而非“语义等价”
    """
    logger = getLogger("denial_comment")
    
    if not query_vertices or not data_vertices:
        if logger:
            logger.info("Empty query or data vertices. Returning empty allowed set.")
        return set()

    # 构建所有 (q_id, d_id) 对及其文本，避免 key 冲突
    pairs_list: List[Tuple[int, int, Vertex, Vertex]] = []
    text_pairs: List[Tuple[str, str]] = []

    for q_id, q_vertex in query_vertices.items():
        for d_id, d_vertex in data_vertices.items():
            pairs_list.append((q_id, d_id, q_vertex, d_vertex))
            text_pairs.append((q_vertex.text(), d_vertex.text()))

    labels = get_nli_labels_batch(text_pairs)

    allowed: Set[Tuple[int, int]] = set()
    allowed_logs: List[Tuple[int, int, str, str]] = []
    contradicted_logs: List[Tuple[int, int, str, str]] = []

    for (q_id, d_id, q_v, d_v), label in zip(pairs_list, labels):
        qt, dt = q_v.text(), d_v.text()
        if label != "contradiction":
            allowed.add((q_id, d_id))
            allowed_logs.append((q_id, d_id, qt, dt))
        else:
            contradicted_logs.append((q_id, d_id, qt, dt))

    # === 全量日志输出（无任何截断）===
    if logger is not None:
        total = len(pairs_list)
        logger.info(f"Total Q-D pairs processed: {total}")
        logger.info(f"Allowed pairs count: {len(allowed_logs)}")
        logger.info(f"Contradicted pairs count: {len(contradicted_logs)}")

        # Log EVERY allowed pair
        if allowed_logs:
            logger.info("=== BEGIN ALLOWED PAIRS ===")
            for idx, (q_id, d_id, qt, dt) in enumerate(allowed_logs, start=1):
                # 不截断文本，保留完整内容（便于精确比对）
                logger.info(f"[ALLOWED {idx}] Q{q_id} ⇨ D{d_id} | '{qt}' vs '{dt}'")
            logger.info("=== END ALLOWED PAIRS ===")
        else:
            logger.info("No allowed pairs.")

        # Log EVERY contradicted pair with explicit reason
        if contradicted_logs:
            logger.info("=== BEGIN CONTRADICTED PAIRS ===")
            for idx, (q_id, d_id, qt, dt) in enumerate(contradicted_logs, start=1):
                logger.info(f"[CONTRADICTED {idx}] Q{q_id} ⇏ D{d_id} | '{qt}' vs '{dt}' (reason: NLI=contradiction)")
            logger.info("=== END CONTRADICTED PAIRS ===")
        else:
            logger.info("No contradicted pairs.")

    return allowed

def build_delta_and_dmatch(
    query: SimHypergraph,
    data: SimHypergraph,
    query_texts: Dict[int, str],
    data_texts: Dict[int, str],
    query_node_edges: Dict[int, List[SimHyperedge]],
    data_node_edges: Dict[int, List[SimHyperedge]],
    allowed_pairs: Set[Tuple[int, int]],
    query_local_hg: LocalHypergraph,
    data_local_hg: LocalHypergraph,
    vertex_to_sim_id_q: Dict[Vertex, int],
    vertex_to_sim_id_d: Dict[Vertex, int],
    dmatch_threshold: float = 0.3
) -> Tuple[Delta, DMatch]:
    """
    构建Delta和D-Match，确保100%覆盖allowed_pairs
    关键设计：
      1. 多节点簇：结构化匹配（异常时降级为空匹配）
      2. 单节点簇：无条件兜底（绕过LocalVertex映射，直接使用Sim ID）
      3. D-Match完备性：每个Delta条目必有D-Match条目（空集也有效）
    """
    
    delta_start = time.time()
    sc_logger = getLogger("semantic_cluster") 
    sc_logger.debug(f"\t\tcalc the delta")
    
    delta = Delta()
    d_delta_matches: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    
    # Step 1: 多节点语义簇（结构化匹配，带异常隔离）
    cluster_count = 0
    # === 阶段1：记录原始结果（来自 get_semantic_cluster_pairs）===
    raw_pairs = list(get_semantic_cluster_pairs(query_local_hg, data_local_hg, sc_logger))
    sc_logger.info(f"语义簇生成完成: 共 {len(raw_pairs)} 个原始簇对")
    # === 阶段2：处理并记录过滤后结果 ===
    cluster_count = 0
    for sc_q, sc_d, sim_score in raw_pairs:
        # --- 提取结构信息 ---
        q_vertices = sc_q.get_vertices()
        d_vertices = sc_d.get_vertices()
        q_edges = sc_q.hyperedges
        d_edges = sc_d.hyperedges

        q_triples = sc_q.to_triple() or []
        d_triples = sc_d.to_triple() or []

        # 取第一个三元组作为代表（若存在）
        q_triple_repr = str(q_triples[0]) if q_triples else "(no triple)"
        d_triple_repr = str(d_triples[0]) if d_triples else "(no triple)"

        q_text = sc_q.text()
        d_text = sc_d.text()

        # --- 日志：原始簇详情（无论是否采纳）---
        sc_logger.info(
            f"→ 原始簇对 | score={sim_score:.3f}\n"
            f"  Q: text='{q_text}'\n"
            f"     triple={q_triple_repr}\n"
            f"     nodes={len(q_vertices)}, edges={len(q_edges)}\n"
            f"  D: text='{d_text}'\n"
            f"     triple={d_triple_repr}\n"
            f"     nodes={len(d_vertices)}, edges={len(d_edges)}"
        )

        # --- 过滤逻辑（保持不变）---
        if sim_score < 0.5:
            sc_logger.info(f"  → 跳过: 低相似度 ({sim_score:.3f})")
            continue

        q_vs = [v for v in q_vertices if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
        d_vs = [v for v in d_vertices if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
        if not q_vs or not d_vs:
            sc_logger.info(f"  → 跳过: 无名词节点 (Q:{len(q_vs)}/{len(q_vertices)}, D:{len(d_vs)}/{len(d_vertices)})")
            continue

        q_rep = min(q_vs, key=lambda v: v.id)
        d_rep = min(d_vs, key=lambda v: v.id)
        q_nid = vertex_to_sim_id_q.get(q_rep)
        d_nid = vertex_to_sim_id_d.get(d_rep)
        if q_nid is None or d_nid is None:
            sc_logger.info(f"  → 跳过: 映射缺失 (Q{q_rep.id}→{q_nid}, D{d_rep.id}→{d_nid})")
            continue

        q_es = list({e for v in q_vs for e in query_node_edges.get(vertex_to_sim_id_q.get(v), []) if e})
        d_es = list({e for v in d_vs for e in data_node_edges.get(vertex_to_sim_id_d.get(v), []) if e})

        sc_id = delta.add_sematic_cluster_pair(
            Node(q_nid, q_text),
            Node(d_nid, d_text),
            q_es,
            d_es
        )

        try:
            matches = {
                (vertex_to_sim_id_q[vq], vertex_to_sim_id_d[vd])
                for vq, vd, _ in get_d_match(sc_q, sc_d, dmatch_threshold)
                if vq in vertex_to_sim_id_q and vd in vertex_to_sim_id_d
            }
        except (AssertionError, AttributeError, IndexError) as e:
            sc_logger.warning(f"  → 语义簇匹配异常: {type(e).__name__}, 降级为空匹配")
            matches = set()

        d_delta_matches[(sc_id, sc_id)] = matches
        cluster_count += 1

        # --- 日志：采纳的簇（含完整结构）---
        sc_logger.info(
            f"→ 采纳 #{cluster_count} | score={sim_score:.3f}\n"
            f"  Q_rep=Q{q_rep.id}('{q_rep.text()}')\n"
            f"     full_text='{q_text}'\n"
            f"     triple={q_triple_repr}\n"
            f"     nodes={len(q_vertices)} (noun={len(q_vs)}), edges={len(q_edges)}\n"
            f"  D_rep=D{d_rep.id}('{d_rep.text()}')\n"
            f"     full_text='{d_text}'\n"
            f"     triple={d_triple_repr}\n"
            f"     nodes={len(d_vertices)} (noun={len(d_vs)}), edges={len(d_edges)}\n"
            f"  D-Match count: {len(matches)}"
        )

    sc_logger.info(f"语义簇构建完成: 原始 {len(raw_pairs)} → 有效 {cluster_count} 个簇对")   
    
    # Step 2: 为allowed_pairs中每个节点对创建单节点簇
    for q_id, d_id in allowed_pairs:
        sc_id = delta.add_sematic_cluster_pair(
            Node(q_id, query_texts.get(q_id, "")),
            Node(d_id, data_texts.get(d_id, "")),
            query_node_edges.get(q_id, []),
            data_node_edges.get(d_id, [])
        )
        d_delta_matches[(sc_id, sc_id)] = {(q_id, d_id)}  # 单节点簇必有自身匹配
    
    return delta, DMatch.from_dict(d_delta_matches)


def compute_hyper_simulation(
    query_hg: LocalHypergraph,
    data_hg: LocalHypergraph
) -> Tuple[Dict[int, Set[int]], Dict[int, Vertex], Dict[int, Vertex]]:
    """
    执行超图模拟
    理论保证：type_same(u,v)=True ⇒ ∃语义簇覆盖(u,v)（通过无条件兜底实现）
    """
    sim_logger = getLogger("hyper_simulation")
    sim_logger.debug(f"\tStart Hyper Simulation")
    
    # 转换到SimHypergraph空间（获得连续Node ID）
    q_sim, q_texts, q_vertices, q_edges, q_vid_map = convert_local_to_sim(query_hg)
    d_sim, d_texts, d_vertices, d_edges, d_vid_map = convert_local_to_sim(data_hg)
    
    denial_start = time.time()
    sim_logger.debug(f"\tstart denial comment calc")
    # 计算宽松的语义允许性
    dc_logger = getLogger("denial_comment")
    allowed = compute_allowed_pairs(q_vertices, d_vertices)
    
    # 定义type_same_fn（基于Sim ID空间）
    def type_same_fn(x_id: int, y_id: int) -> bool:
        return (x_id, y_id) in allowed
    
    q_sim.set_type_same_fn(type_same_fn)
    d_sim.set_type_same_fn(type_same_fn)
    
    denial_end = time.time()
    dc_logger.info(f"\tdenial comment cost {denial_end - denial_start}s")
    
    sim_logger.debug(f"\tdenial comment cost {denial_end - denial_start}s")
    sim_logger.debug(f"\tstart build delta and d-match")
    
    # 构建Delta/D-Match（100%覆盖保障 + 异常隔离）
    delta, d_match = build_delta_and_dmatch(
        q_sim, d_sim, q_texts, d_texts, q_edges, d_edges, allowed,
        query_local_hg=query_hg,
        data_local_hg=data_hg,
        vertex_to_sim_id_q=q_vid_map,
        vertex_to_sim_id_d=d_vid_map,
        dmatch_threshold=0.3
    )
    
    # 执行超图模拟
    start_time = time.time()
    sim_logger.info("\t执行超图模拟...")
    simulation = SimHypergraph.get_hyper_simulation(q_sim, d_sim, delta, d_match)
    # === 新增：结构化输出 simulation 结果（INFO 级别）===
    sim_logger.info("\t=== Hyper Simulation Mapping ===")
    for q_id, d_ids in sorted(simulation.items()):
        # Query 侧文本
        q_text = q_vertices[q_id].text() if q_id in q_vertices else f"[Q{q_id}]"
        
        # Data 侧：ID + 文本
        if d_ids:
            d_items = []
            for d_id in sorted(d_ids):
                if d_id in d_vertices:
                    d_text = d_vertices[d_id].text()
                    d_items.append(f"D{d_id}: '{d_text}'")
                else:
                    d_items.append(f"D{d_id}")
            targets = ", ".join(d_items)
        else:
            targets = "-"

        sim_logger.info(f"\t  Q{q_id}: '{q_text}' → {targets}")
    sim_logger.info("\t================================")
    end_time = time.time()
    sim_logger.info(f"\t模拟完成: {len(simulation)}个映射")
    sim_logger.info(f"\thyper simulation main cost {end_time - start_time}s")

    return simulation, q_vertices, d_vertices