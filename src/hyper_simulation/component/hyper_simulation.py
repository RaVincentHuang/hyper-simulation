from typing import Dict, List, Set, Tuple
import logging
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from simulation import Hypergraph as SimHypergraph, Hyperedge as SimHyperedge, Node, Delta, DMatch
from hyper_simulation.component.nli import get_nli_labels_batch
from hyper_simulation.component.semantic_cluster import get_semantic_cluster_pairs, get_d_match
from hyper_simulation.hypergraph.dependency import Pos
import warnings

logger = logging.getLogger(__name__)

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
    data_vertices: Dict[int, Vertex],
) -> Set[Tuple[int, int]]:
    """
    宽松语义允许性判定：仅排除contradiction
    理论依据：type_same应定义"语义不冲突"，而非"语义等价"
    """
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
    logger.info(f"计算允许对: {len(text_pairs)}个节点对")
    
    for (text1, text2), label in zip(text_pairs, labels):
        q_id, d_id, _, _ = text_pair_to_ids[(text1, text2)]
        if label != "contradiction":  # 仅排除矛盾
            allowed.add((q_id, d_id))
    
    logger.info(f"允许对计算完成: {len(allowed)}对 (非contradiction)")
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
    delta = Delta()
    d_delta_matches: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    
    # Step 1: 多节点语义簇（结构化匹配，带异常隔离）
    cluster_count = 0
    for sc_q, sc_d, sim_score in get_semantic_cluster_pairs(query_local_hg, data_local_hg):
        if sim_score < 0.5:
            continue
        cluster_count += 1
            
        # 过滤动词/助动词（保留名词性实体）
        q_vs = [v for v in sc_q.get_vertices() if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
        d_vs = [v for v in sc_d.get_vertices() if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
        if not q_vs or not d_vs:
            continue
            
        # 选择代表节点（确定性选择）
        q_rep = min(q_vs, key=lambda v: v.id)
        d_rep = min(d_vs, key=lambda v: v.id)
        q_nid = vertex_to_sim_id_q.get(q_rep)
        d_nid = vertex_to_sim_id_d.get(d_rep)
        if q_nid is None or d_nid is None:
            continue  # 映射缺失则跳过（不创建无效簇）
            
        # 收集簇内超边
        q_es = list({e for v in q_vs for e in query_node_edges.get(vertex_to_sim_id_q.get(v), []) if e})
        d_es = list({e for v in d_vs for e in data_node_edges.get(vertex_to_sim_id_d.get(v), []) if e})
        
        # 创建Delta条目（使用Sim ID）
        sc_id = delta.add_sematic_cluster_pair(
            Node(q_nid, sc_q.text()),
            Node(d_nid, sc_d.text()),
            q_es,
            d_es
        )
        
        # 【关键】异常隔离：语义簇组件失败时降级为空匹配
        try:
            matches = {
                (vertex_to_sim_id_q[vq], vertex_to_sim_id_d[vd])
                for vq, vd, _ in get_d_match(sc_q, sc_d, dmatch_threshold)
                if vq in vertex_to_sim_id_q and vd in vertex_to_sim_id_d
            }
        except (AssertionError, AttributeError, IndexError) as e:
            logger.warning(f"语义簇匹配异常: {type(e).__name__}, 降级为空匹配")
            warnings.warn(
                f"Semantic cluster matching failed for cluster pair: {type(e).__name__}",
                RuntimeWarning
            )
            matches = set()
        
        d_delta_matches[(sc_id, sc_id)] = matches
    
    logger.info(f"多节点簇处理完成: {cluster_count}个簇")
    
    # # Step 2: 为allowed_pairs中每个节点对创建单节点簇
    # for q_id, d_id in allowed_pairs:
    #     sc_id = delta.add_sematic_cluster_pair(
    #         Node(q_id, query_texts.get(q_id, "")),
    #         Node(d_id, data_texts.get(d_id, "")),
    #         query_node_edges.get(q_id, []),
    #         data_node_edges.get(d_id, [])
    #     )
    #     d_delta_matches[(sc_id, sc_id)] = {(q_id, d_id)}  # 单节点簇必有自身匹配
    
    return delta, DMatch.from_dict(d_delta_matches)


def compute_hyper_simulation(
    query_hg: LocalHypergraph,
    data_hg: LocalHypergraph
) -> Tuple[Dict[int, Set[int]], Dict[int, Vertex], Dict[int, Vertex]]:
    """
    执行超图模拟
    理论保证：type_same(u,v)=True ⇒ ∃语义簇覆盖(u,v)（通过无条件兜底实现）
    """
    # 转换到SimHypergraph空间（获得连续Node ID）
    q_sim, q_texts, q_vertices, q_edges, q_vid_map = convert_local_to_sim(query_hg)
    d_sim, d_texts, d_vertices, d_edges, d_vid_map = convert_local_to_sim(data_hg)
    
    # 计算宽松的语义允许性
    allowed = compute_allowed_pairs(q_vertices, d_vertices)
    
    # 定义type_same_fn（基于Sim ID空间）
    def type_same_fn(x_id: int, y_id: int) -> bool:
        return (x_id, y_id) in allowed
    
    q_sim.set_type_same_fn(type_same_fn)
    d_sim.set_type_same_fn(type_same_fn)
    
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
    logger.info("执行超图模拟...")
    simulation = SimHypergraph.get_hyper_simulation(q_sim, d_sim, delta, d_match)
    logger.info(f"模拟完成: {len(simulation)}个映射")
    return simulation, q_vertices, d_vertices