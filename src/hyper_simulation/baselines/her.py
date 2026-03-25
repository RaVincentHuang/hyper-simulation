import sys
import os
from typing import Tuple

# Ensure HER is in path
if "/home/vincent/HER" not in sys.path:
    sys.path.insert(0, "/home/vincent/HER")

from src.models import PathModelMrho, RankingModelMr, VertexModelMv
from src.paramatch import ParaMatchConfig, ParaMatcher
from src.vparamatch import VParaMatcher
from src.graph import Node, Edge as HEREdge, GraphView

from hyper_simulation.query_instance import QueryInstance
from hyper_simulation.hypergraph.graph import Graph as LocalGraph
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph
from hyper_simulation.component.consistent import load_hypergraphs_for_instance, get_distance, is_critical_vertex
from hyper_simulation.utils.log import getLogger
from tqdm import tqdm
import copy


class HERScorer:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self.mv = VertexModelMv(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.mrho = PathModelMrho(model_name="BAAI/bge-reranker-base")
        self.mr = RankingModelMr(llm_name="Qwen/Qwen3.5-0.8B-Base")
        self.matcher = ParaMatcher(
            mv=self.mv,
            mrho=self.mrho,
            mr=self.mr,
            config=ParaMatchConfig(sigma=0.75, delta=0.15, k=5)
        )
        self.vmatcher = VParaMatcher(self.matcher, top_n=1, max_candidates=300)
        self._initialized = True


def convert_local_to_her(local_hg: LocalHypergraph, name: str) -> GraphView:
    local_graph = LocalGraph.from_hypergraph(local_hg)
    
    nodes = {}
    for v in local_graph.vertices:
        nodes[str(v.id)] = Node(id=str(v.id), label=v.text())
        
    edges = []
    for edge in local_graph.edges:
        edges.append(HEREdge(
            source=str(edge.src.id),
            target=str(edge.dst.id),
            label=edge.label
        ))
        
    return GraphView(name=name, nodes=nodes, edges=edges)


def her_consistent_detection(
    scorer: HERScorer,
    query: QueryInstance,
    query_hg: LocalHypergraph,
    data_hg: LocalHypergraph,
    query_text: str,
    data_text: str,
    distance_threshold: float = 0.25
) -> Tuple[bool, str]:
    distance = get_distance(query_text, data_text)
    if distance <= distance_threshold:
        return False, f"[CONSISTENT] Distance={distance:.4f} ≤ threshold={distance_threshold}"

    q_her_graph = convert_local_to_her(query_hg, "Q")
    d_her_graph = convert_local_to_her(data_hg, "D")
    
    critical_q_vertices = [v for v in query_hg.vertices if is_critical_vertex(v)]
    if not critical_q_vertices:
        return False, f"[CONSISTENT] No critical vertices to cover (distance={distance:.4f} > threshold)"

    evidence_lines = []
    has_contradiction = False

    for q_vertex in critical_q_vertices:
        q_node_id = str(q_vertex.id)
        try:
            # vmatcher.run 寻找 Q graph 到 D graph 的对应节点
            pairs = scorer.vmatcher.run(q_her_graph, d_her_graph, q_node_id)
            matched = len(pairs) > 0
        except Exception as e:
            matched = False
            
        if not matched:
            has_contradiction = True
            evidence_lines.append(f"Q vertex unmatched in D: '{q_vertex.text()}' (ID={q_vertex.id})")

    if has_contradiction:
        evidence = (
            f"[CONTRADICTION] Distance={distance:.4f} > threshold={distance_threshold}\n"
            + "\n".join(f"  • {line}" for line in evidence_lines)
        )
    else:
        evidence = (
            f"[CONSISTENT] Distance={distance:.4f} > threshold but structural coverage satisfied\n"
            f"  ✓ All {len(critical_q_vertices)} critical Q vertices matched in D via HER"
        )
    return has_contradiction, evidence


def query_fixup(query_instance: QueryInstance, dataset_name: str = "hotpotqa", base_dir: str = "/home/vincent/hyper-simulation/data/hypergraph") -> QueryInstance:
    query_hg, data_hgs = load_hypergraphs_for_instance(query_instance, dataset_name, base_dir=base_dir)
    
    scorer = HERScorer()
    fixed_data = []
    
    for doc_text, data_hg in tqdm(zip(query_instance.data, data_hgs), desc='\tHER Simulation for Query.', leave=True, total=len(query_instance.data)):
        if data_hg is None:
            fixed_data.append(doc_text)
            continue
        
        has_contradiction, evidence = her_consistent_detection(
            scorer, query_instance, query_hg, data_hg, query_instance.query, doc_text
        )
        
        if has_contradiction:
            fixed_doc = (
                f"{doc_text}\n\n"
                f"[HER: INCONSISTENT DETECTED - USE WITH CAUTION]\n"
                f"Evidence:\n{evidence}"
            )
        else:
            fixed_doc = doc_text
        
        fixed_data.append(fixed_doc)

    fixed_instance = copy.deepcopy(query_instance)
    fixed_instance.fixed_data = fixed_data
    return fixed_instance
