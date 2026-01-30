from typing import Dict, List, Set, Tuple
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from simulation import Hypergraph as SimHypergraph, Hyperedge as SimHyperedge, Node, Delta, DMatch
from hyper_simulation.component.nli import get_nli_labels_batch


def convert_local_to_sim(
    local_hg: LocalHypergraph,
) -> Tuple[SimHypergraph, Dict[int, str], Dict[int, Vertex], Dict[int, List[SimHyperedge]]]:
    sim_hg = SimHypergraph()
    vertex_id_map: Dict[int, int] = {}
    node_text: Dict[int, str] = {}
    sim_id_to_vertex: Dict[int, Vertex] = {}
    node_to_edges: Dict[int, List[SimHyperedge]] = {}
    
    for idx, vertex in enumerate(sorted(local_hg.vertices, key=lambda v: v.id)):
        sim_hg.add_node(vertex.text())
        vertex_id_map[vertex.id] = idx
        node_text[idx] = vertex.text()
        sim_id_to_vertex[idx] = vertex
    
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
    
    return sim_hg, node_text, sim_id_to_vertex, node_to_edges


def compute_allowed_pairs(
    query_vertices: Dict[int, Vertex],
    data_vertices: Dict[int, Vertex],
) -> Set[Tuple[int, int]]:
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
    
    for (text1, text2), label in zip(text_pairs, labels):
        q_id, d_id, q_vertex, d_vertex = text_pair_to_ids[(text1, text2)]
        if label == "contradiction":
            continue
        if label == "entailment":
            allowed.add((q_id, d_id))
        elif label == "neutral" and q_vertex.is_domain(d_vertex):
            allowed.add((q_id, d_id))
    
    return allowed


def build_delta_and_dmatch(
    query: SimHypergraph,
    data: SimHypergraph,
    query_texts: Dict[int, str],
    data_texts: Dict[int, str],
    query_node_edges: Dict[int, List[SimHyperedge]],
    data_node_edges: Dict[int, List[SimHyperedge]],
    allowed_pairs: Set[Tuple[int, int]],
) -> Tuple[Delta, DMatch]:
    delta = Delta()
    d_delta_matches: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    
    # TODO: Add semantic clusters and D-Match based on components
    # Delta: `get_semantic_cluster_pairs` -> list[tuple[u, v, cluster_u, cluster_v]]
    # D-Match: `get_d_match` dict: (sc_id_u, sc_id_v) -> set of (u_id, v_id)
    
    for q_id, d_id in sorted(allowed_pairs):
        cluster_u = query_node_edges.get(q_id, [])
        cluster_v = data_node_edges.get(d_id, [])
        u_node = Node(q_id, query_texts.get(q_id, ""))
        v_node = Node(d_id, data_texts.get(d_id, ""))
        sc_id = delta.add_sematic_cluster_pair(u_node, v_node, cluster_u, cluster_v)
        d_delta_matches[(sc_id, sc_id)] = {(q_id, d_id)}
    
    return delta, DMatch.from_dict(d_delta_matches)


def compute_hyper_simulation(
    query_hg: LocalHypergraph,
    data_hg: LocalHypergraph
) -> Tuple[Dict[int, Set[int]], Dict[int, Vertex], Dict[int, Vertex]]:
    q_sim, q_texts, q_vertices, q_edges = convert_local_to_sim(query_hg)
    d_sim, d_texts, d_vertices, d_edges = convert_local_to_sim(data_hg)
    
    allowed = compute_allowed_pairs(q_vertices, d_vertices)
    
    def type_same_fn(x_id: int, y_id: int) -> bool:
        return (x_id, y_id) in allowed
    
    q_sim.set_type_same_fn(type_same_fn)
    d_sim.set_type_same_fn(type_same_fn)
    
    delta, d_match = build_delta_and_dmatch(
        q_sim, d_sim, q_texts, d_texts, q_edges, d_edges, allowed
    )
    
    simulation = SimHypergraph.get_hyper_simulation(q_sim, d_sim, delta, d_match)
    return simulation, q_vertices, d_vertices