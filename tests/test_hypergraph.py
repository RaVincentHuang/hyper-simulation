import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from hyper_simulation.hypergraph.hypergraph import Hypergraph
from hyper_simulation.component.semantic_cluster import get_semantic_cluster_pairs, SemanticCluster, get_d_match


query_file = "data/hypergraph/query_hypergraph.pkl"
data_file = "data/hypergraph/data_hypergraph.pkl"

query_hypergraph = Hypergraph.load(query_file)
data_hypergraph = Hypergraph.load(data_file)

pairs = get_semantic_cluster_pairs(query_hypergraph, data_hypergraph)
# 可能需要尝试：如果存在子集能包含所有的情况，则替换为子集
for qc, dc, score in pairs:
    print(f"Query Cluster Text: {qc.text()}, qc len: {sum(len(he.vertices) for he in qc.hyperedges)}")
    print(f"Vertices in Query Cluster:[{', '.join(v.text() for v in qc.get_vertices())}]")
    print(f"Data Cluster Text: {dc.text()}, dc len: {sum(len(he.vertices) for he in dc.hyperedges)}")
    print(f"Vertices in Data Cluster:[{', '.join(v.text() for v in dc.get_vertices())}]")
    print(f"Similarity Score: {score:.4f}")
    matches = get_d_match(qc, dc)
    for u, v, m_score in matches:
        print(f"  Match Vertex Pair: '{u.text()}' <-> '{v.text()}', Match Score: {m_score:.4f}")
    print("-----")

print(f"Total Pairs: {len(pairs)}")