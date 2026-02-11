import time
from typing import Dict, List, Set, Tuple
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from simulation import Hypergraph as SimHypergraph, Hyperedge as SimHyperedge, Node, Delta, DMatch
from hyper_simulation.component.nli import get_nli_labels_batch, get_nli_label
from hyper_simulation.component.semantic_cluster import get_semantic_cluster_pairs, get_d_match
from hyper_simulation.hypergraph.dependency import Pos, Entity, QueryType
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

def denial_comment(u: Vertex, v: Vertex) -> Tuple[bool, str]:
    """
    返回 (是否非冲突, 原因说明)
    """
    ut = u.text().strip()
    vt = v.text().strip()
    
    # 1. 空文本检查
    if not ut or not vt:
        return False, "Empty text"
    
    # 2. 基于 QueryType 的疑问词处理
    if getattr(u, 'is_query', False):
        query_types = {n.query_type for n in u.nodes if hasattr(n, 'query_type') and n.query_type}
        if query_types:
            matched_ent = []
            for qtype in query_types:
                if qtype == QueryType.PERSON and v.ent_range(Entity.PERSON):
                    matched_ent.append("PERSON")
                elif qtype == QueryType.TIME and (v.ent_range(Entity.DATE) or v.ent_range(Entity.TIME)):
                    matched_ent.append("DATE/TIME")
                elif qtype == QueryType.LOCATION and (v.ent_range(Entity.GPE) or v.ent_range(Entity.LOC)):
                    matched_ent.append("GPE/LOC")
                elif qtype == QueryType.NUMBER and (v.ent_range(Entity.CARDINAL) or v.ent_range(Entity.QUANTITY) or v.pos_range(Pos.NUM)):
                    matched_ent.append("CARDINAL/QUANTITY/NUM")
                elif qtype == QueryType.BELONGS and (v.ent_range(Entity.PERSON) or v.ent_range(Entity.ORG) or v.ent_range(Entity.GPE)):
                    matched_ent.append("PERSON/ORG/GPE")
                elif qtype in {QueryType.WHAT, QueryType.WHICH} and (any(e != Entity.NOT_ENTITY for e in v.ents) or v.pos_range(Pos.NOUN) or v.pos_range(Pos.PROPN)):
                    matched_ent.append("NON_EMPTY_ENTITY_OR_NOUN")
                elif qtype == QueryType.ATTRIBUTE and (v.pos_range(Pos.ADJ) or v.pos_range(Pos.ADV)):
                    matched_ent.append("ADJ/ADV")
                elif qtype == QueryType.REASON and not v.pos_equal(Pos.PUNCT):
                    matched_ent.append("NON_PUNCT")
            
            if matched_ent:
                qtype_names = [str(qt).split('.')[-1] for qt in query_types]
                return True, f"QueryType={qtype_names} → matched Data entity types: {matched_ent}"
            else:
                data_ents = [str(e).split('.')[-1] for e in v.ents if e != Entity.NOT_ENTITY]
                data_poses = [str(p).split('.')[-1] for p in v.poses]
                qtype_names = [str(qt).split('.')[-1] for qt in query_types]
                return False, f"QueryType={qtype_names} → Data has ents={data_ents}, poses={data_poses}"


    # 3. 同类型实体豁免
    u_has_ent = any(e != Entity.NOT_ENTITY for e in u.ents)
    v_has_ent = any(e != Entity.NOT_ENTITY for e in v.ents)
    if u_has_ent and v_has_ent:
        entity_groups = [
            ({Entity.PERSON, Entity.NORP}, "PERSON_GROUP"),
            ({Entity.GPE, Entity.LOC, Entity.FAC, Entity.ORG, Entity.NORP}, "LOCATION_ORG_GROUP"),
            ({Entity.DATE, Entity.TIME}, "TIME_GROUP"),
            ({Entity.PRODUCT, Entity.WORK_OF_ART}, "PRODUCT_GROUP"),
            ({Entity.MONEY, Entity.PERCENT, Entity.QUANTITY, Entity.CARDINAL}, "NUMBER_GROUP"),
            ({Entity.EVENT, Entity.LAW, Entity.LANGUAGE}, "EVENT_GROUP"),
        ]
        for group_entities, group_name in entity_groups:
            u_in_group = any(u.ent_range(e) for e in group_entities)
            v_in_group = any(v.ent_range(e) for e in group_entities)
            if u_in_group and v_in_group:
                return True, f"Entity-compatible: both in {group_name}"

        u_ents = [str(e).split('.')[-1] for e in u.ents if e != Entity.NOT_ENTITY]
        v_ents = [str(e).split('.')[-1] for e in v.ents if e != Entity.NOT_ENTITY]
        return False, f"Entity-mismatch: Query ents={u_ents} vs Data ents={v_ents}"

    # 4. NLI 矛盾检测
    label = get_nli_label(ut, vt)
    if label != "contradiction":
        return True, f"NLI={label} (non-contradiction)"
    else:
        return False, f"NLI={label} (contradiction)"


def compute_allowed_pairs(
    query_vertices: Dict[int, Vertex],
    data_vertices: Dict[int, Vertex]
) -> Set[Tuple[int, int]]:
    """
    批量计算 allowed pairs，并输出详细日志（含未匹配的 Q/D 文本）
    """
    logger = getLogger("denial_comment")
    
    if not query_vertices or not data_vertices:
        if logger:
            logger.info("Empty query or data vertices. Returning empty allowed set.")
        return set()

    allowed: Set[Tuple[int, int]] = set()
    allowed_logs: List[str] = []
    contradicted_logs: List[str] = []

    for q_id, q_vertex in query_vertices.items():
        for d_id, d_vertex in data_vertices.items():
            is_allowed, reason = denial_comment(q_vertex, d_vertex)
            qt = q_vertex.text()
            dt = d_vertex.text()
            
            log_entry = f"Q{q_id}: '{qt}' vs D{d_id}: '{dt}' (reason: {reason})"
            if is_allowed:
                allowed.add((q_id, d_id))
                allowed_logs.append(log_entry)
            else:
                contradicted_logs.append(log_entry)

    # === 全量日志输出（无截断）===
    if logger is not None:
        total = len(query_vertices) * len(data_vertices)
        logger.info(f"Total Q-D pairs processed: {total}")
        logger.info(f"Allowed pairs count: {len(allowed_logs)}")
        logger.info(f"Contradicted pairs count: {len(contradicted_logs)}")

        if allowed_logs:
            logger.info("=== BEGIN ALLOWED PAIRS ===")
            for idx, log in enumerate(allowed_logs, start=1):
                logger.info(f"[ALLOWED {idx}] {log}")
            logger.info("=== END ALLOWED PAIRS ===")
        else:
            logger.info("No allowed pairs.")

        if contradicted_logs:
            logger.info("=== BEGIN CONTRADICTED PAIRS ===")
            for idx, log in enumerate(contradicted_logs, start=1):
                logger.info(f"[CONTRADICTED {idx}] {log}")
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
    raw_pairs = get_semantic_cluster_pairs(query_local_hg, data_local_hg, allowed_pairs, vertex_to_sim_id_q, vertex_to_sim_id_d, max_hops=2, cluster_sim_threshold=0.4, logger=sc_logger)
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
                for vq, vd, _ in get_d_match(sc_q, sc_d, dmatch_threshold, force_include=(q_vertex, d_vertex))
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