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
    宽松语义允许性判定：仅排除contradiction
    理论依据：type_same应定义"语义不冲突"，而非"语义等价"
    """
    logger = getLogger("denial_comment")
    text_pair_to_ids: Dict[Tuple[str, str], Tuple[int, int, Vertex, Vertex]] = {}
    for q_id, q_vertex in query_vertices.items():
        for d_id, d_vertex in data_vertices.items():
            key = (q_vertex.text(), d_vertex.text())
            text_pair_to_ids[key] = (q_id, d_id, q_vertex, d_vertex)
    
    text_pairs = list(text_pair_to_ids.keys())
    if not text_pairs:
        return set()
    
    labels = get_nli_labels_batch(text_pairs)
    allowed: Set[Tuple[int, int]] = set()
    
    # === 新增：准备日志数据（不改变任何逻辑）===
    if logger is not None:
        contradicted_logs = []
    # =========================================
    
    for (text1, text2), label in zip(text_pairs, labels):
        q_id, d_id, _, _ = text_pair_to_ids[(text1, text2)]
        if label != "contradiction":  # 仅排除矛盾
            allowed.add((q_id, d_id))
        # === 新增：收集 contradiction 日志（不改变判断逻辑）===
        elif logger is not None:
            q_v, d_v = text_pair_to_ids[(text1, text2)][2], text_pair_to_ids[(text1, text2)][3]
            contradicted_logs.append((q_id, d_id, q_v.text(), d_v.text()))
        # =========================================
    
    # === 新增：输出日志（不影响返回值）===
    if logger is not None and 'contradicted_logs' in locals():
        if contradicted_logs:
            logger.info(f"Contradiction pairs ({len(contradicted_logs)}):")
            for i, (q_id, d_id, qt, dt) in enumerate(contradicted_logs[:5]):
                qt_s = qt[:50] + "..." if len(qt) > 50 else qt
                dt_s = dt[:50] + "..." if len(dt) > 50 else dt
                logger.info(f"  [{i+1}] Q{q_id} ⇏ D{d_id} | '{qt_s}' vs '{dt_s}'")
        else:
            logger.info("No contradiction pairs found.")
    # =========================================
    
    return allowed  # ← 返回值与原函数完全一致


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

    # # 可选：记录所有原始簇摘要（避免日志爆炸）
    # if raw_pairs:
    #     sc_logger.info("原始簇对摘要（前5个）:")
    #     for i, (sc_q, sc_d, score) in enumerate(raw_pairs[:5]):
    #         q_text = sc_q.text()[:40] + "..." if len(sc_q.text()) > 40 else sc_q.text()
    #         d_text = sc_d.text()[:40] + "..." if len(sc_d.text()) > 40 else sc_d.text()
    #         sc_logger.info(f"  [{i+1}] score={score:.3f} | Q: '{q_text}' | D: '{d_text}'")
    #     if len(raw_pairs) > 5:
    #         sc_logger.info(f"  ... (+{len(raw_pairs) - 5} more)")

    # === 阶段2：处理并记录过滤后结果 ===
    cluster_count = 0
    for sc_q, sc_d, sim_score in raw_pairs:
        # 记录当前处理的原始簇
        q_text_short = sc_q.text()
        d_text_short = sc_d.text()
        
        if sim_score < 0.5:
            sc_logger.info(f"  → 跳过: 低相似度 ({sim_score:.3f}) | Q: '{q_text_short}' | D: '{d_text_short}'")
            continue
        
        q_vs = [v for v in sc_q.get_vertices() if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
        d_vs = [v for v in sc_d.get_vertices() if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
        if not q_vs or not d_vs:
            sc_logger.info(f"  → 跳过: 无名词节点 (Q:{len(q_vs)}/{len(sc_q.get_vertices())}, D:{len(d_vs)}/{len(sc_d.get_vertices())}) | Q: '{q_text_short}'")
            continue
            
        q_rep = min(q_vs, key=lambda v: v.id)
        d_rep = min(d_vs, key=lambda v: v.id)
        q_nid = vertex_to_sim_id_q.get(q_rep)
        d_nid = vertex_to_sim_id_d.get(d_rep)
        if q_nid is None or d_nid is None:
            sc_logger.info(f"  → 跳过: 映射缺失 (Q{q_rep.id}→{q_nid}, D{d_rep.id}→{d_nid}) | Q: '{q_text_short}'")
            continue
                
        q_es = list({e for v in q_vs for e in query_node_edges.get(vertex_to_sim_id_q.get(v), []) if e})
        d_es = list({e for v in d_vs for e in data_node_edges.get(vertex_to_sim_id_d.get(v), []) if e})
        
        sc_id = delta.add_sematic_cluster_pair(
            Node(q_nid, sc_q.text()),
            Node(d_nid, sc_d.text()),
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

    # 记录采纳的簇（含文本）
    sc_logger.info(
        f"  → 采纳 #{cluster_count}: "
        f"score={sim_score:.3f}, "
        f"Q_rep=Q{q_rep.id}('{q_rep.text()}'), "
        f"D_rep=D{d_rep.id}('{d_rep.text()}'), "
        f"Q_nodes={len(q_vs)}, D_nodes={len(d_vs)}, "
        f"matches={len(matches)}"
    )

    delta_end = time.time()
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