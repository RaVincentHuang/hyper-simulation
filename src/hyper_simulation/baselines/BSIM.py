
from typing import Any, Callable, Dict, Set

import networkx as nx

from hyper_simulation.component.denial import denial_comment
from hyper_simulation.hypergraph.graph import Graph
from simulation import get_bounded_simulation


def graph_to_networkx(graph: Graph, default_bound: int = 5) -> nx.DiGraph:
	"""Convert local Graph into a networkx directed graph."""
	nx_graph = nx.DiGraph()

	for vertex in graph.vertices:
		nx_graph.add_node(
			vertex.id,
			vertex_id=vertex.id,
			vertex=vertex,
			text=vertex.text(),
		)

	for edge in graph.edges:
		nx_graph.add_edge(
			edge.src.id,
			edge.dst.id,
			label=edge.label,
			bound=default_bound,
		)

	return nx_graph


def build_compare_table(graph1: Graph, graph2: Graph) -> Dict[tuple[int, int], bool]:
	"""
	Pre-compute node compatibility table using denial_comment.
	Key is (query_vertex_id, data_vertex_id).
	"""
	compare_table: Dict[tuple[int, int], bool] = {}

	for q_vertex in graph1.vertices:
		for d_vertex in graph2.vertices:
			is_allowed, _ = denial_comment(q_vertex, d_vertex)
			compare_table[(q_vertex.id, d_vertex.id)] = is_allowed

	return compare_table


def _normalize_node_id(node: Any) -> int:
	"""Extract integer node id from int or simulation Node object."""
	if isinstance(node, int):
		return node

	node_id = getattr(node, "id", None)
	if isinstance(node_id, int):
		return node_id

	if callable(node_id):
		extracted = node_id()
		if isinstance(extracted, int):
			return extracted

	raise ValueError(f"Unsupported simulation node type: {type(node)}")


def get_bsim_baseline(
	graph1: Graph,
	graph2: Graph,
	is_label_cached: bool = False,
) -> Dict[int, Set[int]]:
	"""
	Run bounded simulation baseline (BSIM) on local Graph objects.

	Steps:
	1. Convert Graph -> networkx.DiGraph
	2. Precompute compare table using denial_comment
	3. Call simulation.get_bounded_simulation with compare and bound callables
	"""
	nx_graph1 = graph_to_networkx(graph1, default_bound=5)
	nx_graph2 = graph_to_networkx(graph2, default_bound=5)
	compare_table = build_compare_table(graph1, graph2)

	def compare(attr1: Dict[str, Any], attr2: Dict[str, Any]) -> bool:
		q_id = attr1.get("vertex_id")
		d_id = attr2.get("vertex_id")
		if not isinstance(q_id, int) or not isinstance(d_id, int):
			return False
		return compare_table.get((q_id, d_id), False)

	def bound(*_args: Any, **_kwargs: Any) -> int:
		return 5

	raw_simulation = get_bounded_simulation(
		nx_graph1,
		nx_graph2,
		compare,
		bound,
		is_label_cached=is_label_cached,
	)

	normalized: Dict[int, Set[int]] = {}
	for src_node, target_nodes in raw_simulation.items():
		src_id = _normalize_node_id(src_node)
		normalized[src_id] = {_normalize_node_id(dst_node) for dst_node in target_nodes}

	return normalized


