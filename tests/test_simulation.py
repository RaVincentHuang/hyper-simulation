import sys
from pathlib import Path

from hypergraph import Hypergraph as LocalHypergraph, Vertex as LocalVertex, DebugLogger
from nli import get_nli_labels_batch
from simulation import Hypergraph as SimHypergraph, Hyperedge as SimHyperedge, Node, Delta, DMatch

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log_path = DebugLogger.init("NER-wordnet-wikidata.txt")

QUERY_FILE = "data/hypergraph/query_hypergraph.pkl"
DATA_FILE = "data/hypergraph/data_hypergraph.pkl"


def normalize(text: str) -> str:
    return text.strip().lower()


def convert_local_to_sim(
    local_hg: LocalHypergraph,
) -> tuple[SimHypergraph, dict[int, str], dict[int, LocalVertex], dict[int, list[SimHyperedge]]]:
    """Translate a LocalHypergraph into simulation-alg structures while keeping node metadata."""
    sim_hg = SimHypergraph()
    vertex_id_map: dict[int, int] = {}
    node_text: dict[int, str] = {}
    sim_id_to_vertex: dict[int, LocalVertex] = {}
    node_to_edges: dict[int, list[SimHyperedge]] = {}
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
    query_vertices: dict[int, LocalVertex],
    data_vertices: dict[int, LocalVertex],
) -> set[tuple[int, int]]:
    """Use NLI + domain heuristics to decide which node IDs can share type."""
    text_pair_to_ids: dict[tuple[str, str], tuple[int, int, LocalVertex, LocalVertex]] = {}
    for q_id, q_vertex in query_vertices.items():
        for d_id, d_vertex in data_vertices.items():
            key = (q_vertex.text(), d_vertex.text())
            text_pair_to_ids[key] = (q_id, d_id, q_vertex, d_vertex)
    text_pairs = list(text_pair_to_ids.keys())
    if not text_pairs:
        return set()
    labels = get_nli_labels_batch(text_pairs)
    allowed: set[tuple[int, int]] = set()
    
    for (text1, text2), label in zip(text_pairs, labels):
        q_id, d_id, q_vertex, d_vertex = text_pair_to_ids[(text1, text2)]
        if label == "contradiction":
            continue
        if label == "entailment":
            allowed.add((q_id, d_id))
        elif label == "neutral" and q_vertex.is_domain(d_vertex, debug=True):
            allowed.add((q_id, d_id))
    return allowed

def build_delta_and_dmatch(
    query: SimHypergraph,
    data: SimHypergraph,
    query_texts: dict[int, str],
    data_texts: dict[int, str],
    query_node_edges: dict[int, list[SimHyperedge]],
    data_node_edges: dict[int, list[SimHyperedge]],
    allowed_pairs: set[tuple[int, int]],
) -> tuple[Delta, DMatch]:
    """Register each allowed node pair as a semantic cluster pair and build DMatch."""
    delta = Delta() # 看这个计算结果目前是多少pairs
    d_delta_matches: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for q_id, d_id in sorted(allowed_pairs):
        cluster_u = query_node_edges.get(q_id, [])
        cluster_v = data_node_edges.get(d_id, [])
        u_node = Node(q_id, query_texts.get(q_id, ""))
        v_node = Node(d_id, data_texts.get(d_id, ""))
        sc_id = delta.add_sematic_cluster_pair(u_node, v_node, cluster_u, cluster_v)
        d_delta_matches[(sc_id, sc_id)] = {(q_id, d_id)}
    return delta, DMatch.from_dict(d_delta_matches)


def format_result(
    title: str,
    result: dict[int, set[int]],
    query_texts: dict[int, str],
    data_texts: dict[int, str],
) -> None:
    print(title)
    for q_id, vs in sorted(result.items()):
        q = query_texts.get(q_id, "")
        targets = ", ".join(data_texts[v] for v in sorted(vs)) if vs else "-"
        print(f"  [{q_id}] {q} -> {targets}")
    print("--------------------------------")
    print(f"result: {result}")


def main() -> None:
    query_local = LocalHypergraph.load(QUERY_FILE)
    data_local = LocalHypergraph.load(DATA_FILE)
    query, query_texts, query_vertices, query_node_edges = convert_local_to_sim(query_local)
    data, data_texts, data_vertices, data_node_edges = convert_local_to_sim(data_local)
    allowed_pairs = compute_allowed_pairs(query_vertices, data_vertices)

    def type_same_fn(x_id: int, y_id: int) -> bool:
        return (x_id, y_id) in allowed_pairs

    query.set_type_same_fn(type_same_fn)
    data.set_type_same_fn(type_same_fn)

    delta, d_match = build_delta_and_dmatch(
        query,
        data,
        query_texts,
        data_texts,
        query_node_edges,
        data_node_edges,
        allowed_pairs,
    )
    sim = SimHypergraph.get_hyper_simulation(query, data, delta, d_match)
    format_result("get_hyper_simulation:", sim, query_texts, data_texts)
    DebugLogger.close()

if __name__ == "__main__":
    main()

# 目标：
# 1. 文本有冲突能检查出来
# 2. 现有冲突没有沿着超边进行传递

# 冲突的结果和不冲突的结果 Boundary的设定