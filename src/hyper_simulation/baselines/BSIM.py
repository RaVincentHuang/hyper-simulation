from __future__ import annotations

from typing import Any, Dict, Set, Tuple, List
import networkx as nx

from hyper_simulation.hypergraph.graph import Graph
from hyper_simulation.hypergraph.hypergraph import Hypergraph
from hyper_simulation.component.denial import denial_comment
from simulation import get_bounded_simulation
from hyper_simulation.component.consistent import load_hypergraphs_for_instance
from hyper_simulation.query_instance import QueryInstance

def graph_to_networkx(graph: Graph, default_bound: int = 5) -> nx.DiGraph:
	"""
	将本地 Graph 转换为 networkx.DiGraph
	
	使用 Graph 类的 vertices 和 edges 属性
	"""
	nx_graph = nx.DiGraph()
	
	# 添加节点
	for vertex in graph.vertices:
		nx_graph.add_node(
			vertex.id,
			vertex_id=vertex.id,
			vertex=vertex,
			text=vertex.text(),
		)
	
	# 添加边
	for edge in graph.edges:
		nx_graph.add_edge(
			edge.src.id,
			edge.dst.id,
			label=edge.label,
			bound=default_bound,
		)
	
	return nx_graph

def build_compare_table(graph1: Graph, graph2: Graph) -> Dict[Tuple[int, int], bool]:
	compare_table: Dict[Tuple[int, int], bool] = {}
	for q_vertex in graph1.vertices:
		for d_vertex in graph2.vertices:
			q_text = q_vertex.text().lower()
			d_text = d_vertex.text().lower()
			if not any(word in d_text for word in q_text.split() if len(word) > 3):
				compare_table[(q_vertex.id, d_vertex.id)] = False
				continue
			is_allowed, _ = denial_comment(q_vertex, d_vertex)
			compare_table[(q_vertex.id, d_vertex.id)] = is_allowed
	return compare_table

def get_bsim_baseline(
	hypergraph1: Hypergraph,
	hypergraph2: Hypergraph,
	default_bound: int = 5,
	is_label_cached: bool = False,
) -> Dict[int, Set[int]]:
	"""
	运行 Bounded Simulation Baseline (BSIM)
	
	Returns:
		Dict[query_node_id, Set[data_node_ids]] - 每个查询节点匹配的数据节点集合
	"""
	# 1. Hypergraph → Graph (使用现有的 from_hypergraph 方法)
	graph1 = Graph.from_hypergraph(hypergraph1)
	graph2 = Graph.from_hypergraph(hypergraph2)
	
	# 2. Graph → networkx.DiGraph
	nx_graph1 = graph_to_networkx(graph1, default_bound)
	nx_graph2 = graph_to_networkx(graph2, default_bound)
	
	# 3. 预计算兼容性表
	compare_table = build_compare_table(graph1, graph2)
	
	# 4. 定义 compare 和 bound 回调
	def compare(attr1: Dict[str, Any], attr2: Dict[str, Any]) -> bool:
		q_id = attr1.get("vertex_id")
		d_id = attr2.get("vertex_id")
		if not isinstance(q_id, int) or not isinstance(d_id, int):
			return False
		return compare_table.get((q_id, d_id), False)
	
	def bound(*_args: Any, **_kwargs: Any) -> int:
		return default_bound
	
	# 5. 运行有界模拟
	raw_simulation = get_bounded_simulation(nx_graph1, nx_graph2, compare, bound, is_label_cached=is_label_cached)
	
	# 6. 规范化输出 (确保 node_id 是 int)
	normalized: Dict[int, Set[int]] = {}
	for src_node, target_nodes in raw_simulation.items():
		src_id = src_node if isinstance(src_node, int) else getattr(src_node, 'id', src_node)
		normalized[src_id] = {
			dst_node if isinstance(dst_node, int) else getattr(dst_node, 'id', dst_node)
			for dst_node in target_nodes
		}
	
	return normalized
# hyper_simulation/baselines/bsim.py

def run_bsim_for_query(
	qi: QueryInstance,
	task: str,
	hypergraph_dir: str = "/home/vincent/hyper-simulation/data/hypergraph",
	default_bound: int = 5,
) -> QueryInstance:
	"""
	对单个 QueryInstance 运行 BSIM 匹配
	"""
	query_hg, context_hgs = load_hypergraphs_for_instance(qi, dataset_name=task, base_dir=hypergraph_dir)
	
	context_hgs = [hg for hg in context_hgs if hg is not None]
	
	if query_hg and context_hgs:
		matched_context_indices = []
		for idx, context_hg in enumerate(context_hgs):
			matches = get_bsim_baseline(query_hg, context_hg, default_bound=default_bound)
			if matches: 
				matched_context_indices.append(idx)
		if matched_context_indices:
			qi.fixed_data = [qi.data[i] for i in matched_context_indices if i < len(qi.data)]
		else:
			qi.fixed_data = qi.data
	else:
		qi.fixed_data = qi.data
	
	return qi