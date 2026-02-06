import itertools
from os import path
import re
from hyper_simulation.hypergraph.hypergraph import Hyperedge, Hypergraph, Node, Vertex, LocalDoc, Path
from hyper_simulation.hypergraph.dependency import LocalDoc, Node, Pos, Dep
import numpy as np
import logging

from hyper_simulation.component.embedding import get_embedding_batch, cosine_similarity, get_similarity_batch, get_similarity

from hyper_simulation.component.nli import get_nli_label, get_nli_labels_batch

from tqdm import tqdm
from hyper_simulation.utils.log import getLogger

def abstraction_lca(query: list[str], data: list[str]) -> tuple[str, int]:
    """
    计算 LCA。如果两个路径完全没有重合（根节点不同），返回 None, -1。
    """
    if not query or not data:
        return '', -1
        
    # 检查根节点是否相同
    # 名词根是 entity，动词根可能是 act，如果不相同，说明属于不同词性域
    if query[0] != data[0]:
        return '', -1
        
    lca = query[0]
    depth = 0
    
    min_len = min(len(query), len(data))
    for i in range(min_len):
        if query[i] == data[i]:
            lca = query[i]
            depth = i
        else:
            break
            
    return lca, depth

def _vertex_sort_key(vertex: Vertex) -> tuple[int, str]:
    return (vertex.id, vertex.text())


def _hyperedge_signature(hyperedge: Hyperedge) -> tuple[int, int, int, str]:
    root_id = hyperedge.root.id if hyperedge.root else -1
    return (root_id, hyperedge.start, hyperedge.end, hyperedge.desc)


def _path_sort_key(path: Path) -> tuple:
    sig = [_hyperedge_signature(he) for he in path.hyperedges]
    return (len(path.hyperedges), sig)


def _cluster_sort_key(cluster: 'SemanticCluster') -> tuple:
    return cluster.signature()


class TarjanLCA:
    def __init__(self, edges: list[tuple[Node, Node]], queries: list[tuple[Node, Node]]) -> None:
        # build adjacency list (directed) and node set
        self.adj: dict[Node, list[Node]] = {}
        self.nodes: set[Node] = set()
        
        # 统计入度，用于寻找有向图/树的根节点
        in_degree: dict[Node, int] = {}

        for a, b in edges:
            self.nodes.add(a)
            self.nodes.add(b)
            if a not in self.adj:
                self.adj[a] = []
            self.adj[a].append(b)
            
            # 初始化入度
            if a not in in_degree: in_degree[a] = 0
            if b not in in_degree: in_degree[b] = 0
            in_degree[b] += 1

        # store queries and build per-node query map
        self.queries = list(queries)
        self.query_map: dict[Node, list[tuple[Node, int]]] = {}
        
        for i, (u, v) in enumerate(self.queries):
            self.nodes.add(u)
            self.nodes.add(v)
            if u not in in_degree: in_degree[u] = 0
            if v not in in_degree: in_degree[v] = 0

            # 建立双向映射
            if u not in self.query_map: self.query_map[u] = []
            if v not in self.query_map: self.query_map[v] = []
            
            self.query_map[u].append((v, i))
            if u != v:
                self.query_map[v].append((u, i))

        # union-find parent and ancestor used by Tarjan's algorithm
        self.uf_parent: dict[Node, Node] = {}
        self.ancestor: dict[Node, Node] = {}
        self.visited: set[Node] = set()
        self.res: list[Node | None] = [None] * len(self.queries)

        # [新增逻辑] 用于记录节点属于哪棵树（哪个连通分量）
        self.node_roots: dict[Node, Node] = {}

        # initialize union-find for all nodes
        for n in list(self.nodes):
            self.uf_parent[n] = n
            self.ancestor[n] = n

        # run Tarjan on each component (forest support)
        # 优先从根节点（入度为0）开始 DFS
        sorted_nodes = sorted(list(self.nodes), key=lambda n: in_degree.get(n, 0))
        
        for n in sorted_nodes:
            if n not in self.visited:
                # [修改逻辑] 传入当前分量的根节点 n 作为 root_id
                self.tarjan(n, None, n)
        
    # union-find's find
    def find(self, x):
        if x not in self.uf_parent:
            self.uf_parent[x] = x
            return x
        if self.uf_parent[x] != x:
            self.uf_parent[x] = self.find(self.uf_parent[x])
        return self.uf_parent[x]
    
    # union-find's union
    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        self.uf_parent[ry] = rx
    
    # [修改接口] 增加 root_id 参数，标记当前递归属于哪棵树
    def tarjan(self, u, p, root_id):
        # [新增逻辑] 记录当前节点所属的树根
        self.node_roots[u] = root_id

        self.ancestor[u] = u 
        
        for v in self.adj.get(u, []):
            if v == p: 
                continue
            if v in self.visited:
                continue
            
            # [修改逻辑] 递归传递 root_id
            self.tarjan(v, u, root_id)
            self.union(u, v)
            self.ancestor[self.find(u)] = u

        self.visited.add(u)

        for other, qi in self.query_map.get(u, []):
            # [修复核心] 只有当 other 也被访问过，且 other 属于同一棵树（同一个 root_id）时，才计算 LCA
            # 如果属于不同的树，说明不连通，LCA 保持为 None
            if other in self.visited:
                if self.node_roots.get(other) == root_id:
                    self.res[qi] = self.ancestor[self.find(other)]

    def lca(self) -> list[Node | None]:
        return self.res

class SemanticCluster:
    def __init__(self, hyperedges: list[Hyperedge], doc: LocalDoc, is_query: bool=True) -> None:
        self.hyperedges = hyperedges
        self.doc = doc
        self.vertices: list[Vertex] = []
        self.contained_hyperedges: dict[Vertex, list[Hyperedge]] = {}
        self.embedding: np.ndarray | None = None
        self.text_cache: str | None = None
        
        self.vertices_paths: dict[tuple[Vertex, Vertex], tuple[str, int]] = {}
        self.node_paths_cache: dict[tuple[Node, Node], tuple[str, int]] = {}
        
        self.is_query = is_query
        self._signature: tuple | None = None
        
    @staticmethod
    def likely_nodes(nodes1: list[Vertex], nodes2: list[Vertex]) -> dict[Vertex, set[Vertex]]:
        # node is likely if NLI label is entailment or share same pos
        likely_nodes: dict[Vertex, set[Vertex]] = {}
        text_pair_to_node_pairs: dict[tuple[str, str], tuple[Vertex, Vertex]] = {}
        for node1 in nodes1:
            for node2 in nodes2:
                text_pair_to_node_pairs[(node1.text(), node2.text())] = (node1, node2)
        text_pairs = list(text_pair_to_node_pairs.keys())
        labels = get_nli_labels_batch(text_pairs)
        for i, text_pair in enumerate(text_pairs):
            node_pair = text_pair_to_node_pairs[text_pair]
            label = labels[i]
            node1, node2 = node_pair
            if label == "entailment" or (label == "neutral" and node1.is_domain(node2)):
                if node1 not in likely_nodes:
                    likely_nodes[node1] = set()
                likely_nodes[node1].add(node2)
        return likely_nodes
    
    
    def is_subset_of(self, other: 'SemanticCluster') -> bool:
        self_edge_set = set(self.hyperedges)
        other_edge_set = set(other.hyperedges)
        return self_edge_set.issubset(other_edge_set)
    
    def get_contained_hyperedges(self, vertex: Vertex) -> list[Hyperedge]:
        if vertex in self.contained_hyperedges:
            return self.contained_hyperedges[vertex]
        contained_edges: list[Hyperedge] = []
        for he in self.hyperedges:
            if vertex in he.vertices:
                contained_edges.append(he)
        self.contained_hyperedges[vertex] = contained_edges
        return contained_edges
    
    def get_vertices(self) -> list[Vertex]:
        if len(self.vertices) > 0:
            return self.vertices
        id_set: set[int] = set()
        ordered_vertices: list[Vertex] = []
        for he in self.hyperedges:
            for v in he.vertices:
                if v.id in id_set:
                    continue
                id_set.add(v.id)
                ordered_vertices.append(v)
        self.vertices = ordered_vertices
        return self.vertices
    
    def get_paths_between_vertices(self, v1: Vertex, v2: Vertex) -> tuple[str, int]:
        key = (v1, v2)
        if key in self.vertices_paths:
            return self.vertices_paths[key]
        logger = getLogger("semantic_cluster")
        logger.debug(f"get_paths_between_vertices called for: '{v1.text()}' ↔ '{v2.text()}'")

        node_vertex: dict[Node, Vertex] = {}
        nodes_in_vertices: set[Node] = set()
        for he in self.hyperedges:
            for v in he.vertices:
                if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
                    continue
                nodes_in_vertices.add(he.current_node(v))
                node_vertex[he.current_node(v)] = v
            
        nodes_in_vertices_list = list(nodes_in_vertices)
        queries: list[tuple[Node, Node]] = []
        for i in range(len(nodes_in_vertices_list) - 1):
            for j in range(i + 1, len(nodes_in_vertices_list)):
                u = nodes_in_vertices_list[i]
                v = nodes_in_vertices_list[j]
                queries.append((u, v))
        
        edge_between_nodes: list[tuple[Node, Node]] = []
        saved_nodes: set[Node] = set()
        for he in self.hyperedges:
            root = he.current_node(he.root)
            for i in range(1, len(he.vertices)):
                node = he.current_node(he.vertices[i])
                edge_between_nodes.append((root, node))
                saved_nodes.add(node)
            head = root.head
            current = root
            while head:
                edge_between_nodes.append((head, current))
                if head in saved_nodes:
                    break
                current = head
                # >>> 新增：防止自环导致死循环 <<<
                if head.head == head:
                    logger.warning(f"Detected self-loop at node '{current.text}' during v→k trace. Breaking.")
                    break
                head = head.head
            saved_nodes.add(root)
        lca_results = TarjanLCA(edge_between_nodes, queries).lca()
        lca_map: dict[tuple[Node, Node], Node] = {}
        for i, (u, v) in enumerate(queries):
            lca_node = lca_results[i]
            if lca_node:
                lca_map[(u, v)] = lca_node
            
        node_paths: dict[tuple[Vertex, Vertex], list[tuple[str, int]]] = {}
        for (u, v), k in lca_map.items():
            vertex_u = node_vertex[u]
            vertex_v = node_vertex[v]
            
            if u == k:
                text = f"#A -{v.dep.name}-> #B"
                node_paths.setdefault((vertex_u, vertex_v), []).append((text, 1))
                continue
            elif v == k:
                text = f"#A <-{u.dep.name}- #B"
                node_paths.setdefault((vertex_u, vertex_v), []).append((text, 1))
                continue
            
            # === 修复1: 安全追溯 u -> k (移除 assert) ===
            node_cnt = 1
            path_items: list[Node] = []
            current = u
            current_trace: list[str] = [current.text]
            while current != k:
                if current in nodes_in_vertices:
                    node_cnt += 1
                    path_items.append(current)
                if current.head is None:
                    logger.warning(f"路径追溯失败 u→k: Node '{current.text}' (index={current.index}) has no head "
                            f"while tracing to LCA '{k.text}' (index={k.index}). "
                            f"Trace: {' → '.join(current_trace)}")
                    break
                # >>> 新增：防止自环导致死循环 <<<
                if current.head == current:
                    logger.warning(f"Detected self-loop at node '{current.text}' during u→k trace. Breaking.")
                    break
                current = current.head
                current_trace.append(current.text)
            else:
                # 仅当成功到达 LCA 时才继续
                path_items.append(k)
                
                # === 修复2: 安全追溯 v -> k (移除 assert) ===
                rev_path_items: list[Node] = []
                current = v
                current_trace = [current.text]
                while current != k:
                    if current in nodes_in_vertices:
                        node_cnt += 1
                        rev_path_items.append(current)
                    if current.head is None:
                        logger.warning(f"路径追溯失败 v→k: Node '{current.text}' (index={current.index}) has no head "
                                f"while tracing to LCA '{k.text}' (index={k.index}). "
                                f"Trace: {' → '.join(current_trace)}")
                        break
                    # >>> 新增：防止自环导致死循环 <<<
                    if current.head == current:
                        logger.warning(f"Detected self-loop at node '{current.text}' during v→k trace. Breaking.")
                        break
                    current = current.head
                    current_trace.append(current.text)
                else:
                    # 仅当两个方向都成功到达 LCA 时才构建路径
                    rev_path_items = rev_path_items[::-1]
                    path_items.extend(rev_path_items)
                    text = node_sequence_to_text(path_items)
                    text_inv = text.replace("#A", "#TEMP").replace("#B", "#A").replace("#TEMP", "#B")
                    
                    node_paths.setdefault((vertex_u, vertex_v), []).append((text, node_cnt))
                    node_paths.setdefault((vertex_v, vertex_u), []).append((text_inv, node_cnt))
                    continue  # 成功处理，跳过后续
        
        # 选择最短路径
        for (vertex_u, vertex_v), paths in node_paths.items():
            if paths:
                paths = sorted(paths, key=lambda x: x[1])
                self.vertices_paths[(vertex_u, vertex_v)] = paths[0]
        
        result = self.vertices_paths.get(key, ("", 0))
        logger.debug(f"get_paths_between_vertices result: count={result[1]}, sample='{result[0][:50]}...'")
        return result
    
    def text(self) -> str:
        if self.text_cache is not None:
            return self.text_cache

        if not self.hyperedges:
            return ""
        logger = getLogger("semantic_cluster")
        try:
            # Step 1: Build root_ancestors mapping
            root_ancestors = {}
            for e in self.hyperedges:
                root_node = e.current_node(e.root)
                if root_node is None:
                    logger.error(f"[text] Hyperedge {e} has invalid root node (None). Skipping.")
                    continue
                root_ancestors[root_node] = root_node

            # Resolve ancestor chains (with cycle detection)
            for e in self.hyperedges:
                root = e.current_node(e.root)
                if root is None:
                    continue
                node = root
                visited = set()
                while node.head is not None:
                    if node in visited:
                        logger.warning(f"[text] Detected cycle in ancestor chain starting from {root.text}. Breaking.")
                        break
                    visited.add(node)
                    if node.head in root_ancestors:
                        root_ancestors[root] = root_ancestors[node.head]
                        break
                    node = node.head

            # Step 2: Group nodes by ultimate root
            root_to_nodes: dict[Node, set[Node]] = {}
            for e in self.hyperedges:
                root = e.current_node(e.root)
                if root is None or root not in root_ancestors:
                    continue
                ultimate_root = root_ancestors[root]
                if ultimate_root not in root_to_nodes:
                    root_to_nodes[ultimate_root] = set()
                for vertex in e.vertices:
                    node = e.current_node(vertex)
                    if node is not None:
                        root_to_nodes[ultimate_root].add(node)

            # Step 3: Order sub-clusters by index
            sub_cluster_roots = set(root_ancestors.get(r, r) for r in root_to_nodes.keys())
            sub_clusters = sorted(list(sub_cluster_roots), key=lambda r: getattr(r, 'index', float('inf')))

            # Step 4: Generate text for each sub-cluster
            texts = []
            for root in sub_clusters:
                if root not in root_to_nodes:
                    continue
                nodes = list(root_to_nodes[root])
                if not nodes:
                    continue

                try:
                    start = min(getattr(node, 'index', 0) for node in nodes)
                    end = max(getattr(node, 'index', 0) for node in nodes) + 1
                except Exception as ex:
                    logger.error(f"[text] Failed to compute indices for root {root.text}: {ex}")
                    continue

                sentence_by_range = str(self.doc[start:end]) if self.doc else ""
                sentence_obj = getattr(root, 'sentence', None)
                sentence = str(sentence_obj) if sentence_obj else ""

                # Helper to extract prefix/suffix
                def calc_prefix_suffix(range_text, full_sentence):
                    start_idx = full_sentence.find(range_text)
                    if start_idx != -1:
                        prefix = full_sentence[:start_idx].strip()
                        suffix = full_sentence[start_idx + len(range_text):].strip()
                        return prefix, suffix
                    else:
                        return "", ""

                prefix, suffix = calc_prefix_suffix(sentence_by_range, sentence)

                # Build replacement list
                replacement = []
                for node in nodes:
                    if node == root:
                        continue
                    resolved_text = Vertex.resolved_text(node)
                    original_text = getattr(node, 'text', '')
                    replacement.append((original_text, resolved_text))

                # Add prefix/suffix removal
                if prefix:
                    replacement.append((prefix, ""))
                if suffix:
                    replacement.append((suffix, ""))

                # Apply replacements
                final_sentence = sentence
                for old, new in replacement:
                    if old in final_sentence:
                        final_sentence = final_sentence.replace(old, new)

                cleaned = final_sentence.strip()
                if cleaned:
                    texts.append(cleaned)

            # Step 5: Combine
            text = " ".join(texts).strip()
            self.text_cache = text
            return text

        except Exception as e:
            logger.exception(f"[text] Unexpected error in SemanticCluster.text(): {e}")
            fallback = " ".join(
                str(e.current_node(e.root).text) for e in self.hyperedges
                if e.current_node(e.root) and hasattr(e.current_node(e.root), 'text')
            ).strip()
            self.text_cache = fallback
            return fallback    
    
    def _build_signature(self) -> tuple:
        if not self.hyperedges:
            return ()
        items = []
        for he in self.hyperedges:
            root_id = he.root.id if he.root else -1
            items.append((root_id, he.start, he.end, he.desc))
        items.sort()
        return tuple(items)

    def signature(self) -> tuple:
        if self._signature is None:
            self._signature = self._build_signature()
        return self._signature

    def to_triple(self) -> list[tuple[str, list[str]]]:
        """
        将 SemanticCluster 抽象为三元组形式: (root, [node1, node2, ...])
        返回一个三元组列表，每个 hyperedge 对应一个三元组
        """
        triples = []
        for he in self.hyperedges:
            root_text = Vertex.resolved_text(he.current_node(he.root))
            
            # 收集非root的节点
            args = []
            for vertex in he.vertices:
                if vertex == he.root:
                    continue
                node = he.current_node(vertex)
                node_text = Vertex.resolved_text(node)
                args.append(node_text)
                
                if node.pos in {Pos.ADJ, Pos.ADV} and node.dep in {Dep.amod, Dep.advmod}:
                    head = node.head
                    if head and head.pos in {Pos.NOUN, Pos.PROPN, Pos.VERB}:
                        head_text = Vertex.resolved_text(head)
                        triples.append(("attr", [head_text, node_text]))
            
            triples.append((root_text, args))
        
        return triples
    
    def to_triple_text(self) -> str:
        """返回所有三元组的文本表示"""
        texts = []
        for root, args in self.to_triple():
            if len(args) == 0:
                texts.append(f"{root}()")
            else:
                texts.append(f"{root}({', '.join(args)})")
        return " & ".join(texts)

    def __hash__(self) -> int:
        return hash((self.is_query, self.signature()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticCluster):
            return False
        if self.is_query != other.is_query:
            return False
        return self.signature() == other.signature()

def calc_embedding_for_cluster_batch(clusters: list[SemanticCluster]) -> None:
    texts = [sc.text() for sc in clusters]
    embeddings = get_embedding_batch(texts)
    for i, sc in enumerate(clusters):
        sc.embedding = np.array(embeddings[i])


def compare_triples(triple1: tuple[str, list[str]], triple2: tuple[str, list[str]]) -> float:
    """比较两个三元组的相似度，返回 0-1 分数"""
    root1, args1 = triple1
    root2, args2 = triple2
    # 1. 比较root的相似度
    root_sim = get_similarity(root1, root2)
    
    # 2. 比较参数的相似度（使用NLI）
    if len(args1) == 0 and len(args2) == 0:
        return root_sim
    
    if len(args1) == 0 or len(args2) == 0:
        return root_sim * 0.5  # 参数数量不匹配，降低分数
    
    # 计算参数之间的最佳匹配
    arg_scores = []
    for arg1 in args1:
        best_score = 0.0
        for arg2 in args2:
            label = get_nli_label(arg1, arg2)
            if label == "entailment":
                score = 1.0
            elif label == "neutral":
                score = 0.5
            else:
                score = 0.0
            best_score = max(best_score, score)
        arg_scores.append(best_score)
    
    avg_arg_score = sum(arg_scores) / len(arg_scores) if arg_scores else 0.0
    
    # 综合分数：root占60%，参数占40%
    return root_sim * 0.6 + avg_arg_score * 0.4


def match_vertices_by_triple(
    q_edge: Hyperedge, 
    d_edge: Hyperedge,
    q_cluster: SemanticCluster,
    d_cluster: SemanticCluster
) -> dict[Vertex, Vertex]:
    """基于三元组结构匹配两个边中的顶点"""
    matches: dict[Vertex, Vertex] = {}
    
    # 获取三元组
    q_triples = q_cluster.to_triple()
    d_triples = d_cluster.to_triple()
    
    if not q_triples or not d_triples:
        return matches
    
    q_triple = q_triples[0]  # 单边cluster只有一个三元组
    d_triple = d_triples[0]
    
    q_root_text, q_args = q_triple
    d_root_text, d_args = d_triple
    
    # 1. 匹配root节点
    if get_similarity(q_root_text, d_root_text) > 0.6:
        matches[q_edge.root] = d_edge.root
    
    # 2. 匹配参数节点
    q_non_root_vertices = [v for v in q_edge.vertices if v != q_edge.root]
    d_non_root_vertices = [v for v in d_edge.vertices if v != d_edge.root]
    
    # 使用NLI匹配参数
    for q_vertex in q_non_root_vertices:
        q_text = Vertex.resolved_text(q_edge.current_node(q_vertex))
        best_match = None
        best_score = 0.0
        
        for d_vertex in d_non_root_vertices:
            if d_vertex in matches.values():
                continue  # 已经被匹配
            
            d_text = Vertex.resolved_text(d_edge.current_node(d_vertex))
            label = get_nli_label(q_text, d_text)
            
            if _legal_vertices(q_vertex, d_vertex):  # ← 统一入口
                if label == "entailment":
                    score = 1.0
                elif label == "neutral":
                    score = 0.7
                else:
                    score = 0.0
            else:
                score = 0.0
            
            if score > best_score:
                best_score = score
                best_match = d_vertex
        
        if best_match and best_score > 0.5:
            matches[q_vertex] = best_match
    
    return matches

def path_clean(paths: list[Path]) -> list[Path]:
    # remove paths that share same hyperedges
    # in a path, hyperedges are unique, if not, keep only one
    
    # 1. for each path, remove duplicate hyperedges
    cleaned_paths: list[Path] = []
    for path in paths:
        seen_hyperedges: set[int] = set()
        unique_hyperedges: list[Hyperedge] = []
        for he in path.hyperedges:
            he_id = id(he)
            if he_id not in seen_hyperedges:
                seen_hyperedges.add(he_id)
                unique_hyperedges.append(he)
        cleaned_paths.append(Path(unique_hyperedges))
    
    unique_paths: list[Path] = []
    seen_hyperedge_sets: set[frozenset[int]] = set()
    for path in cleaned_paths:
        hyperedge_ids = frozenset(id(e) for e in path.hyperedges)
        if hyperedge_ids not in seen_hyperedge_sets:
            seen_hyperedge_sets.add(hyperedge_ids)
            unique_paths.append(path)
    
    return unique_paths
    

def clean_semantic_cluster_pairs(pairs: list[tuple[SemanticCluster, SemanticCluster, float]]) -> list[tuple[SemanticCluster, SemanticCluster, float]]:
    # remove pairs where one cluster (qc, dc, score) if there exists another pair (qc', dc', score')
    # and score <= score' and (qc is subset of qc' or dc is subset of dc') 
    cleaned_pairs: list[tuple[SemanticCluster, SemanticCluster, float]] = []
    for i, (qc, dc, score) in enumerate(pairs):
        is_subset = False
        for j, (qc2, dc2, score2) in enumerate(pairs):
            if i == j:
                continue
            if score <= score2:
                if qc2.is_subset_of(qc) or dc2.is_subset_of(dc):
                    is_subset = True
                    break
        if not is_subset:
            cleaned_pairs.append((qc, dc, score))
    return cleaned_pairs
        

def _build_cluster_closure(
    initial_q_edges: set[Hyperedge],
    initial_d_edges: set[Hyperedge],
    query_hypergraph: Hypergraph,
    data_hypergraph: Hypergraph,
    matched_edges: list[tuple[Hyperedge, Hyperedge, float]],
    matched_vertices: dict[Vertex, set[Vertex]],
    edge_similarity_threshold: float = 0.7
) -> tuple[set[Hyperedge], set[Hyperedge]]:
    """
    递归地构建cluster闭包。
    如果一对匹配的边中有节点匹配了其他边，那就添加进去，直到没有新的边可以添加。
    """
    q_edges = set(initial_q_edges)
    d_edges = set(initial_d_edges)
    
    # 构建边到顶点的映射
    q_edge_to_vertices: dict[Hyperedge, set[Vertex]] = {}
    d_edge_to_vertices: dict[Hyperedge, set[Vertex]] = {}
    
    for edge in query_hypergraph.hyperedges:
        q_edge_to_vertices[edge] = set(edge.vertices)
    for edge in data_hypergraph.hyperedges:
        d_edge_to_vertices[edge] = set(edge.vertices)
    
    # 构建顶点到边的映射
    q_vertex_to_edges: dict[Vertex, set[Hyperedge]] = {}
    d_vertex_to_edges: dict[Vertex, set[Hyperedge]] = {}
    
    for edge in query_hypergraph.hyperedges:
        for vertex in edge.vertices:
            q_vertex_to_edges.setdefault(vertex, set()).add(edge)
    
    for edge in data_hypergraph.hyperedges:
        for vertex in edge.vertices:
            d_vertex_to_edges.setdefault(vertex, set()).add(edge)
    
    # 构建匹配边对的快速查找映射（仅保留高置信匹配）
    q_edge_to_matched_d_edges: dict[Hyperedge, list[tuple[Hyperedge, float]]] = {}
    d_edge_to_matched_q_edges: dict[Hyperedge, list[tuple[Hyperedge, float]]] = {}
    
    for q_edge, d_edge, score in matched_edges:
        if score >= edge_similarity_threshold:
            q_edge_to_matched_d_edges.setdefault(q_edge, []).append((d_edge, score))
            d_edge_to_matched_q_edges.setdefault(d_edge, []).append((q_edge, score))
    
    # 闭包迭代
    changed = True
    iteration = 0
    max_iterations = len(matched_edges)
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        # if iteration > 1:
            # with logging_redirect_tqdm():
            #     logger.info(f"闭包迭代 {iteration}: q_edges={len(q_edges)}, d_edges={len(d_edges)}")
        
        # 收集当前 cluster 中的所有顶点
        q_vertices_in_cluster = {v for e in q_edges for v in q_edge_to_vertices.get(e, ())}
        d_vertices_in_cluster = {v for e in d_edges for v in d_edge_to_vertices.get(e, ())}
        
        # 正向：从 query cluster 出发，通过匹配顶点扩展 data 边，再带回 query 边
        for q_vertex in q_vertices_in_cluster:
            for d_vertex in matched_vertices.get(q_vertex, ()):
                for d_edge in d_vertex_to_edges.get(d_vertex, ()):
                    if d_edge in d_edges:
                        continue
                    # 检查该 d_edge 是否与当前 data cluster 有顶点连接
                    if d_edge_to_vertices[d_edge] & d_vertices_in_cluster:
                        # 找到其匹配的 query 边
                        for q_edge, _ in d_edge_to_matched_q_edges.get(d_edge, []):
                            if q_edge not in q_edges and q_edge_to_vertices[q_edge] & q_vertices_in_cluster:
                                q_edges.add(q_edge)
                                d_edges.add(d_edge)
                                changed = True
        
        # 反向：从 data cluster 出发，通过匹配顶点扩展 query 边，再带回 data 边
        for d_vertex in d_vertices_in_cluster:
            # 高效反向查找匹配的 query 顶点（避免 O(n) 遍历）
            matched_q_vertices = [q_v for q_v, d_vs in matched_vertices.items() if d_vertex in d_vs]
            for q_vertex in matched_q_vertices:
                for q_edge in q_vertex_to_edges.get(q_vertex, ()):
                    if q_edge in q_edges:
                        continue
                    if q_edge_to_vertices[q_edge] & q_vertices_in_cluster:
                        for d_edge, _ in q_edge_to_matched_d_edges.get(q_edge, []):
                            if d_edge not in d_edges and d_edge_to_vertices[d_edge] & d_vertices_in_cluster:
                                q_edges.add(q_edge)
                                d_edges.add(d_edge)
                                changed = True

    return q_edges, d_edges

def get_semantic_cluster_pairs(query_hypergraph: Hypergraph, data_hypergraph: Hypergraph, sc_logger: logging.Logger) -> list[tuple[SemanticCluster, SemanticCluster, float]]:
    """
    新的实现：基于边和节点匹配的递归cluster构造。
    1. 匹配所有的边和节点
    2. 对于每一对匹配的边，递归地添加相关的边（如果边中的节点匹配了其他边）
    3. 求闭包，确保所有相关的边都被包含
    """
    # Step 1: 为所有单个边创建cluster并计算embedding
    single_cluster_q: list[SemanticCluster] = []
    edge_to_cluster_q: dict[Hyperedge, SemanticCluster] = {}
    for e in query_hypergraph.hyperedges:
        sc = SemanticCluster([e], query_hypergraph.doc)
        single_cluster_q.append(sc)
        edge_to_cluster_q[e] = sc
        
    texts_q = [sc.text() for sc in single_cluster_q]
    embeddings_q = get_embedding_batch(texts_q)
    for i, sc in enumerate(single_cluster_q):
        sc.embedding = np.array(embeddings_q[i])
    sc_logger.debug(f"embedding query for {texts_q}")
    single_cluster_d: list[SemanticCluster] = []
    edge_to_cluster_d: dict[Hyperedge, SemanticCluster] = {}
    for e in data_hypergraph.hyperedges:
        sc = SemanticCluster([e], data_hypergraph.doc)
        single_cluster_d.append(sc)
        edge_to_cluster_d[e] = sc
    sc_logger.debug(f"end hyperedge")
    texts_d = [sc.text() for sc in single_cluster_d]
    sc_logger.debug(f"before embedding: {texts_d}")
    embeddings_d = get_embedding_batch(texts_d)
    for i, sc in enumerate(single_cluster_d):
        sc.embedding = np.array(embeddings_d[i])
    sc_logger.debug(f"edge_to_cluster and embedding for {texts_d}")    
    # Step 2: 匹配所有的节点
    text_pair_to_node_pairs: dict[tuple[str, str], tuple[Vertex, Vertex]] = {}
    for node_q in sorted(query_hypergraph.vertices, key=_vertex_sort_key):
        for node_d in sorted(data_hypergraph.vertices, key=_vertex_sort_key):
            text_pair_to_node_pairs[(node_q.text(), node_d.text())] = (node_q, node_d)
    
    text_pairs = list(text_pair_to_node_pairs.keys())
    labels = get_nli_labels_batch(text_pairs)
    node_pair_to_label: dict[tuple[Vertex, Vertex], str] = {}
    for i, text_pair in enumerate(text_pairs):
        node_pair = text_pair_to_node_pairs[text_pair]
        node_pair_to_label[node_pair] = labels[i]
    sc_logger.debug(f"nli node_pair_to_label {labels}")
    matched_vertices: dict[Vertex, set[Vertex]] = {}
    for (node_q, node_d), label in node_pair_to_label.items():
        if label == "entailment" or (label == "neutral" and node_q.is_domain(node_d)):
            if node_q not in matched_vertices:
                matched_vertices[node_q] = set()
            matched_vertices[node_q].add(node_d)
    sc_logger.debug("nli matched_vertices")
    # Step 3: 匹配所有的边对（结合embedding和三元组相似度）
    matched_edges: list[tuple[Hyperedge, Hyperedge, float]] = []
    edge_similarity_threshold = 0.6  # 边相似度阈值
    
    for q_sc in single_cluster_q:
        if q_sc.embedding is None:
            continue
        for d_sc in single_cluster_d:
            if d_sc.embedding is None:
                continue
            # 1. Embedding相似度
            emb_score = cosine_similarity(q_sc.embedding, d_sc.embedding)
            # 2. 三元组相似度
            q_triples = q_sc.to_triple()
            d_triples = d_sc.to_triple()
            
            triple_score = 0.0
            if q_triples and d_triples:
                triple_score = compare_triples(q_triples[0], d_triples[0])
            
            # 3. 综合分数：embedding占70%，三元组占30%
            combined_score = emb_score * 0.7 + triple_score * 0.3
            
            if combined_score >= edge_similarity_threshold:
                q_edge = q_sc.hyperedges[0]
                d_edge = d_sc.hyperedges[0]
                matched_edges.append((q_edge, d_edge, combined_score))
                
                # 同时使用三元组匹配更新节点匹配
                vertex_matches = match_vertices_by_triple(q_edge, d_edge, q_sc, d_sc)
                for q_v, d_v in vertex_matches.items():
                    if q_v not in matched_vertices:
                        matched_vertices[q_v] = set()
                    matched_vertices[q_v].add(d_v)
                    
    sc_logger.info(f"边匹配完成: {len(matched_edges)}对 (阈值={edge_similarity_threshold})")
    
    # Step 4: 对于每一对匹配的边，递归地构建cluster闭包
    cluster_pairs: set[tuple[SemanticCluster, SemanticCluster, float]] = set()
    processed_pairs: set[tuple[frozenset[int], frozenset[int]]] = set()
    
    for q_edge, d_edge, initial_score in matched_edges:
        # 构建初始cluster
        initial_q_edges = {q_edge}
        initial_d_edges = {d_edge}
        
        # 递归地构建闭包
        q_edges_closure, d_edges_closure = _build_cluster_closure(
            initial_q_edges,
            initial_d_edges,
            query_hypergraph,
            data_hypergraph,
            matched_edges,
            matched_vertices,
            edge_similarity_threshold
        )
        # with logging_redirect_tqdm():
        #     logger.info(f"闭包迭代完成: q_edges={len(q_edges_closure)}, d_edges={len(d_edges_closure)}")
        # 检查是否已经处理过这个pair
        q_edge_ids = frozenset(id(e) for e in q_edges_closure)
        d_edge_ids = frozenset(id(e) for e in d_edges_closure)
        pair_key = (q_edge_ids, d_edge_ids)
        
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        
        # 创建cluster并计算最终相似度
        q_cluster = SemanticCluster(list(q_edges_closure), query_hypergraph.doc)
        d_cluster = SemanticCluster(list(d_edges_closure), data_hypergraph.doc)
        
        calc_embedding_for_cluster_batch([q_cluster, d_cluster])
        
        if q_cluster.embedding is not None and d_cluster.embedding is not None:
            final_score = cosine_similarity(q_cluster.embedding, d_cluster.embedding)
            cluster_pairs.add((q_cluster, d_cluster, final_score))
    
    # Step 5: 清理和去重
    ans_pairs = []
    seen_pairs: set[tuple[frozenset[int], frozenset[int]]] = set()
    for qc, dc, score in sorted(
        cluster_pairs,
        key=lambda pair: (_cluster_sort_key(pair[0]), _cluster_sort_key(pair[1]), -pair[2])
    ):
        qc_id_set = frozenset(id(e) for e in qc.hyperedges)
        dc_id_set = frozenset(id(e) for e in dc.hyperedges)
        pair_key = (qc_id_set, dc_id_set)
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            ans_pairs.append((qc, dc, score))

    unique_pairs: list[tuple[SemanticCluster, SemanticCluster, float]] = []
    seen_text_pairs: set[tuple[str, str]] = set()
    for qc, dc, score in ans_pairs:
        key = (qc.text(), dc.text())
        if key in seen_text_pairs:
            continue
        seen_text_pairs.add(key)
        unique_pairs.append((qc, dc, score))

    unique_pairs = sorted(unique_pairs, key=lambda pair: (pair[0].text(), pair[1].text(), -pair[2]))
    return unique_pairs

def node_sequence_to_text(nodes: list[Node]) -> str:
    if not nodes:
        return ""
    start, end = nodes[0], nodes[-1]
    nodes = sorted(nodes, key=lambda n: n.index)
    texts = []
    for node in nodes:
        if node == start:
            texts.append("#A")
        elif node == end:
            texts.append("#B")
        elif node.pos in {Pos.ADV, Pos.ADJ, Pos.DET}:
            continue
        elif node.pos in {Pos.NOUN, Pos.PROPN, Pos.PRON}:
            texts.append("some")
        else:
            texts.append(Vertex.resolved_text(node))
    return " ".join(texts)
        

def _formal_text_of(root: Node, node: Node) -> str:
    match (root.pos, node.dep):
        case (Pos.AUX, Dep.nsubj) | (Pos.AUX, Dep.nsubjpass):
            text = "#A is something"
        case (Pos.AUX, Dep.iobj) | (Pos.AUX, Dep.dobj):
            text = "#A is something"
        case (Pos.VERB, Dep.nsubj) | (Pos.VERB, Dep.nsubjpass):
            text = "#A does something"
        case (Pos.VERB, Dep.iobj) | (Pos.VERB, Dep.dobj):
            text = "Someone does #A"
        case _:
            text = f"#A -{node.dep.name}-> something"
    return text

def _better_path(s1: str, s2: str, s2_inv: str) -> bool:
    nli_labels = {"entailment": 3, "neutral": 2, "contradiction": 1}
    label1 = get_nli_label(s1, s2)
    label2 = get_nli_label(s1, s2_inv)
    if nli_labels[label1] > nli_labels[label2]: # s2 is better
        return True
    
    sim1 = get_similarity(s1, s2)
    sim2 = get_similarity(s1, s2_inv)
    return sim1 > sim2 
    

def _legal_vertices(v1: Vertex, v2: Vertex) -> bool:
    # Step 1: 语义兼容性（保留你原有的 is_domain 逻辑）
    label = get_nli_label(v1.text(), v2.text())
    if not (label == "entailment" or (label == "neutral" and v1.is_domain(v2))):
        # logger.info(f"节点语义不兼容: '{v1.text()}' vs '{v2.text()}' (NLI={label})")
        return False

    # Step 2: 【新增】句法角色（Dep）兼容性检查
    dep1 = v1.dep()
    dep2 = v2.dep()

    # 定义兼容的依存关系组
    SUBJECT_DEPS = {Dep.nsubj, Dep.nsubjpass, Dep.csubj, Dep.agent}
    OBJECT_DEPS = {Dep.dobj, Dep.iobj, Dep.pobj, Dep.attr}
    MODIFIER_DEPS = {Dep.amod, Dep.nmod, Dep.advmod, Dep.appos}

    # 同组内允许匹配
    if (dep1 in SUBJECT_DEPS and dep2 in SUBJECT_DEPS) or (dep1 in OBJECT_DEPS and dep2 in OBJECT_DEPS) or (dep1 in MODIFIER_DEPS and dep2 in MODIFIER_DEPS):
        return True

    # 允许常见 paraphrase 跨组（如 nmod ↔ dobj）
    if {dep1, dep2} <= {Dep.nmod, Dep.dobj}:
        return True

    # 其他情况拒绝（即使 is_domain 为真）
    # logger.info(f"依存关系不匹配: '{v1.text()}'({dep1.name}) vs '{v2.text()}'({dep2.name})")
    return False

def _path_score(s1: str, cnt1: int, s2: str, cnt2: int, path_score_cache: dict[tuple[str, str], float]) -> float:
    key = (s1, s2)
    if key in path_score_cache:
        return path_score_cache[key]
    sim = get_similarity(s1, s2)
    score = sim / (cnt1 + cnt2)
    path_score_cache[key] = score
    return score

def _get_matched_vertices(vertices1: list[Vertex], vertices2: list[Vertex]) -> dict[Vertex, set[Vertex]]: # 松紧可以调整
    matched_vertices: dict[Vertex, set[Vertex]] = {}
    text_pair_to_node_pairs: dict[tuple[str, str], tuple[Vertex, Vertex]] = {}
    for node1 in vertices1:
        for node2 in vertices2:
            text_pair_to_node_pairs[(node1.text(), node2.text())] = (node1, node2)
    text_pairs = list(text_pair_to_node_pairs.keys())
    labels = get_nli_labels_batch(text_pairs)
    for i, text_pair in enumerate(text_pairs):
        node_pair = text_pair_to_node_pairs[text_pair]
        label = labels[i]
        node1, node2 = node_pair
        if label == "entailment" or node1.is_domain(node2):
            if node1 not in matched_vertices:
                matched_vertices[node1] = set()
            matched_vertices[node1].add(node2)
    return matched_vertices

def get_d_match(sc1: SemanticCluster, sc2: SemanticCluster, score_threshold: float = 0.0) -> list[tuple[Vertex, Vertex, float]]:
    dm_logger = getLogger("d_match")

    # --- 提取 SC1 信息 ---
    sc1_vertices_all = sc1.get_vertices()
    sc1_vertices_noun = [v for v in sc1_vertices_all if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
    sc1_edges = sc1.hyperedges
    sc1_text = sc1.text()
    sc1_triples = sc1.to_triple() or []
    sc1_triple_repr = str(sc1_triples[0]) if sc1_triples else "(no triple)"

    # --- 提取 SC2 信息 ---
    sc2_vertices_all = sc2.get_vertices()
    sc2_vertices_noun = [v for v in sc2_vertices_all if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
    sc2_edges = sc2.hyperedges
    sc2_text = sc2.text()
    sc2_triples = sc2.to_triple() or []
    sc2_triple_repr = str(sc2_triples[0]) if sc2_triples else "(no triple)"

    # --- 完整日志：当前比较的语义簇对 ---
    dm_logger.info(
        f"=== D-Match 开始 (阈值={score_threshold}) ===\n"
        f"→ SC1:\n"
        f"   text='{sc1_text}'\n"
        f"   triple={sc1_triple_repr}\n"
        f"   nodes={len(sc1_vertices_all)} (noun={len(sc1_vertices_noun)}), edges={len(sc1_edges)}\n"
        f"→ SC2:\n"
        f"   text='{sc2_text}'\n"
        f"   triple={sc2_triple_repr}\n"
        f"   nodes={len(sc2_vertices_all)} (noun={len(sc2_vertices_noun)}), edges={len(sc2_edges)}"
    )
    matches: list[tuple[Vertex, Vertex]] = []
    # 如果两个边的节点很少，则输出结果会很少
    sc1_vertices = list(filter(lambda v: not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX)), sc1.get_vertices()))
    sc2_vertices = list(filter(lambda v: not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX)), sc2.get_vertices()))
    
    # logger.info(f"SC1 non-verb vertices: {[v.text() for v in sc1_vertices]}")
    # logger.info(f"SC2 non-verb vertices: {[v.text() for v in sc2_vertices]}")

    index_map: dict[Vertex, int] = {}
    for e in sc1.hyperedges:
        for v in e.vertices:
            if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
                continue
            if v not in index_map:
                index_map[v] = e.current_node(v).index
                
    for e in sc2.hyperedges:
        for v in e.vertices:
            if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
                continue
            if v not in index_map:
                index_map[v] = e.current_node(v).index
    
    sc1_edges: list[tuple[Vertex, Vertex]] = []
    for he in sc1.hyperedges:
        for i in range(len(he.vertices) - 1):
            for j in range(i + 1, len(he.vertices)):
                if he.have_no_link(he.vertices[i], he.vertices[j]):
                    continue
                if he.is_sub_vertex(he.vertices[i], he.vertices[j]):
                    sc1_edges.append((he.vertices[i], he.vertices[j]))
                else:
                    sc1_edges.append((he.vertices[j], he.vertices[i]))
    
    # logger.info(f"SC1 direct edges: {[(u.text(), v.text()) for u, v in sc1_edges]}")

    sc1_pairs : list[tuple[Vertex, Vertex]] = []
    # all (u, v) in sc1_edges are in sc1_pairs, and if (u, k), (k, v) in sc1_edges, then (u, v) is also in sc1_pairs
    # calculate then recursively
    added = True
    for u, v in sc1_edges:
        sc1_pairs.append((u, v))
    while added:
        added = False
        current_pairs = sc1_pairs.copy()
        for u1, v1 in current_pairs:
            for u2, v2 in current_pairs:
                if v1 == u2:
                    new_pair = (u1, v2)
                    if new_pair not in sc1_pairs:
                        sc1_pairs.append(new_pair)
                        added = True
    
    def _is_pair_in_vertices(u: Vertex, v: Vertex) -> bool:
        if u.pos_equal(Pos.VERB) or u.pos_equal(Pos.AUX):
            return False
        if v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX):
            return False
        return True
    
    sc1_pairs = list(filter(lambda pairs: _is_pair_in_vertices(pairs[0], pairs[1]), sc1_pairs))
    
    # logger.info(f"SC1 path pairs after closure and filter: {[(u.text(), v.text()) for u, v in sc1_pairs]}")

    sc1_paths: dict[tuple[Vertex, Vertex], tuple[str, int]] = {}
    for u, v in sc1_pairs:
        s, cnt = sc1.get_paths_between_vertices(u, v)
        if cnt == 0:
            continue
        sc1_paths[(u, v)] = (s, cnt)
    
    # logger.info(f"SC1 valid paths count: {len(sc1_paths)}")

    likely_nodes = _get_matched_vertices(sc1_vertices, sc2_vertices)
    
    # logger.info("Likely matched nodes: { " + ", ".join([f"{u.text()}→[{', '.join(v.text() for v in vs)}]" for u, vs in likely_nodes.items() if vs]) + " }")

    sc2_pairs: list[tuple[Vertex, Vertex]] = []
    sc2_paths: dict[tuple[Vertex, Vertex], tuple[str, int]] = {}
    
    # 核心匹配逻辑
    for u, u_prime in sc1_pairs:
        for v, v_prime in itertools.product(likely_nodes.get(u, set()), likely_nodes.get(u_prime, set())):
            if v == v_prime:
                continue
            s1, cnt1 = sc1_paths[(u, u_prime)]
            # logger.info(f"    Calling sc2.get_paths_between_vertices('{v.text()}', '{v_prime.text()}')")
            s2, cnt2 = sc2.get_paths_between_vertices(v, v_prime)
            # logger.info(f"    Forward path: count={cnt2}, sample='{s2[:50]}...'")

            # logger.info(f"    Calling sc2.get_paths_between_vertices('{v_prime.text()}', '{v.text()}')")
            s2_inv, cnt2_prime = sc2.get_paths_between_vertices(v_prime, v)
            # logger.info(f"    Backward path: count={cnt2_prime}, sample='{s2_inv[:50]}...'")
            
            # 处理单向路径缺失
            if cnt2 == 0 or s2 == "":
                if cnt2_prime > 0 and s2_inv:
                    sc2_pairs.append((v_prime, v))
                    sc2_paths[(v_prime, v)] = (s2_inv, cnt2_prime)
                continue
            elif cnt2_prime == 0 or s2_inv == "":
                sc2_pairs.append((v, v_prime))
                sc2_paths[(v, v_prime)] = (s2, cnt2)
                continue
            
            # === 修复3: 移除危险 assert，替换为防御性跳过 + 精准日志 ===
            if not s2 or not s2_inv:
                # logger.info(f"D-Match跳过: Empty paths for vertex pair '{v.text()}' ↔ '{v_prime.text()}' in cluster. s2='{s2}', s2_inv='{s2_inv}'")
                continue
            
            if _better_path(s1, s2, s2_inv):
                sc2_pairs.append((v, v_prime))
                sc2_paths[(v, v_prime)] = (s2, cnt2)
            else:
                sc2_pairs.append((v_prime, v))
                sc2_paths[(v_prime, v)] = (s2_inv, cnt2)

    # logger.info(f"SC2 inferred path pairs: {[(u.text(), v.text()) for u, v in sc2_pairs]}")
    # logger.info(f"SC2 paths count: {len(sc2_paths)}")

    # 让每一个节点和root做一次计算，通过此计算能得到一个分数。核心在于确定超边的子边方向
    match_scores: dict[tuple[Vertex, Vertex], float] = {}
    
    for u, v in itertools.product(sc1_vertices, sc2_vertices):
        if _legal_vertices(u, v):
            matches.append((u, v))
    
    # logger.info(f"Initial legal matches count: {len(matches)}")

    in_paths_of_sc1: dict[Vertex, list[tuple[str, int]]] = {}
    out_paths_of_sc1: dict[Vertex, list[tuple[str, int]]] = {}
    for u, v in sc1_pairs:
        if v not in in_paths_of_sc1:
            in_paths_of_sc1[v] = []
        in_paths_of_sc1[v].append(sc1_paths[(u, v)])
        if u not in out_paths_of_sc1:
            out_paths_of_sc1[u] = []
        out_paths_of_sc1[u].append(sc1_paths[(u, v)])
    
    # for vertex in sc1_vertices:
    #     if vertex in in_paths_of_sc1:
    #         logger.info(f"SC1 Vertex '{vertex.text()}' In Paths: {[s for s, _ in in_paths_of_sc1[vertex]]}")
    #     if vertex in out_paths_of_sc1:
    #         logger.info(f"SC1 Vertex '{vertex.text()}' Out Paths: {[s for s, _ in out_paths_of_sc1[vertex]]}")
    
    
    in_paths_of_sc2: dict[Vertex, list[tuple[str, int]]] = {}
    out_paths_of_sc2: dict[Vertex, list[tuple[str, int]]] = {}
    for u, v in sc2_pairs:
        if v not in in_paths_of_sc2:
            in_paths_of_sc2[v] = []
        in_paths_of_sc2[v].append(sc2_paths[(u, v)])
        if u not in out_paths_of_sc2:
            out_paths_of_sc2[u] = []
        out_paths_of_sc2[u].append(sc2_paths[(u, v)])
    
    # for vertex in sc2_vertices:
    #     if vertex in in_paths_of_sc2:
    #         logger.info(f"SC2 Vertex '{vertex.text()}' In Paths: {[s for s, _ in in_paths_of_sc2[vertex]]}")
    #     if vertex in out_paths_of_sc2:
    #         logger.info(f"SC2 Vertex '{vertex.text()}' Out Paths: {[s for s, _ in out_paths_of_sc2[vertex]]}")
    
    root_path_of_sc1: dict[Vertex, list[tuple[str, int]]] = {}
    for e in sc1.hyperedges:
        root = e.root
        root_node = e.current_node(root)
        if not (root_node.pos == Pos.VERB or root_node.pos == Pos.AUX):
            continue
        for v in e.vertices[1:]:
            v_node = e.current_node(v)
            if v_node.pos == Pos.VERB or v_node.pos == Pos.AUX:
                continue
            text = _formal_text_of(root_node, v_node)
            if v not in root_path_of_sc1:
                root_path_of_sc1[v] = []
            root_path_of_sc1[v].append((text, 2))
    root_path_of_sc2: dict[Vertex, list[tuple[str, int]]] = {}
    for e in sc2.hyperedges:
        root = e.root
        root_node = e.current_node(root)
        if not (root_node.pos == Pos.VERB or root_node.pos == Pos.AUX):
            continue
        for v in e.vertices[1:]:
            v_node = e.current_node(v)
            if v_node.pos == Pos.VERB or v_node.pos == Pos.AUX:
                continue
            text = _formal_text_of(root_node, v_node)
            if v not in root_path_of_sc2:
                root_path_of_sc2[v] = []
            root_path_of_sc2[v].append((text, 2))

    # logger.info(f"SC1 root paths count: {sum(len(ps) for ps in root_path_of_sc1.values())}")
    # logger.info(f"SC2 root paths count: {sum(len(ps) for ps in root_path_of_sc2.values())}")

    
    path_score_cache: dict[tuple[str, str], float] = {}
    path_pair_need_to_calc: set[tuple[str, str]] = set()
    for u, v in matches:
        for s1, cnt1 in in_paths_of_sc1.get(u, []):
            for s2, cnt2 in in_paths_of_sc2.get(v, []):
                if (s1, s2) not in path_score_cache:
                    path_pair_need_to_calc.add((s1, s2))
        for s1, cnt1 in out_paths_of_sc1.get(u, []):
            for s2, cnt2 in out_paths_of_sc2.get(v, []):
                if (s1, s2) not in path_score_cache:
                    path_pair_need_to_calc.add((s1, s2))
        for s1, cnt1 in root_path_of_sc1.get(u, []):
            for s2, cnt2 in root_path_of_sc2.get(v, []):
                if (s1, s2) not in path_score_cache:
                    path_pair_need_to_calc.add((s1, s2))

    # logger.info(f"Path similarity pairs to compute: {len(path_pair_need_to_calc)}")
    
    path_list_1: list[str] = []
    path_list_2: list[str] = []
    path_pair_need_to_calc_list = list(path_pair_need_to_calc)
    for s1, s2 in path_pair_need_to_calc_list:
        path_list_1.append(s1)
        path_list_2.append(s2)
    similarities = get_similarity_batch(path_list_1, path_list_2)
    for i, (s1, s2) in enumerate(path_pair_need_to_calc_list):
        path_score_cache[(s1, s2)] = similarities[i]
    
    # logger.info("Path similarity cache populated.")

    for u, v in matches:
        in_score = 0.0
        in_cnt = 0
        for s1, cnt1 in in_paths_of_sc1.get(u, []):
            for s2, cnt2 in in_paths_of_sc2.get(v, []):
                in_score += _path_score(s1, cnt1, s2, cnt2, path_score_cache)
                in_cnt += 1
        if in_cnt > 0:
            in_score /= in_cnt

        out_score = 0.0
        out_cnt = 0
        for s1, cnt1 in out_paths_of_sc1.get(u, []):
            for s2, cnt2 in out_paths_of_sc2.get(v, []):
                out_score += _path_score(s1, cnt1, s2, cnt2, path_score_cache)
                out_cnt += 1
        if out_cnt > 0:
            out_score /= out_cnt
            
        root_score = 0.0
        root_cnt = 0
        for s1, cnt1 in root_path_of_sc1.get(u, []):
            for s2, cnt2 in root_path_of_sc2.get(v, []):
                root_score += _path_score(s1, cnt1, s2, cnt2, path_score_cache)
                root_cnt += 1
        if root_cnt > 0:
            root_score /= root_cnt
        
        match_scores[(u, v)] = in_score + out_score + root_score
        
        # logger.info(f"Match score computed: '{u.text()}' ↔ '{v.text()}' = in({in_score:.3f}) + out({out_score:.3f}) + root({root_score:.3f}) = {match_scores[(u, v)]:.3f}")
        
    # filter by score_threshold
    matches = list(filter(lambda pair: match_scores.get(pair, 0.0) >= score_threshold, matches))
    # logger.info(f"D-Match过滤后: {len(matches)}个匹配 (阈值={score_threshold})")
    
    # delete the matches that if (u, v1) and (u, v2) in matches and v1 != v2, keep only the one with highest score
    final_matches: list[tuple[Vertex, Vertex, float]] = []
    matches_by_u: dict[Vertex, list[tuple[Vertex, float]]]  = {}
    for u, v in matches:
        score = match_scores.get((u, v), 0.0)
        if u not in matches_by_u:
            matches_by_u[u] = []
        matches_by_u[u].append((v, score))
    for u, v_scores in matches_by_u.items():
        v_scores = sorted(v_scores, key=lambda x: x[1], reverse=True)
        best_v, best_score = v_scores[0]
        final_matches.append((u, best_v, best_score))
        # if len(v_scores) > 1:
            # logger.info(f"Disambiguation for '{u.text()}': kept '{best_v.text()}' (score={best_score:.3f}), others: {[v.text() for v, s in v_scores[1:]]}")

    # === 新增：完整输出所有 D-Match 结果（不截断）===
    if final_matches:
        dm_logger.info("D-Match 完整结果:")
        for i, (u, v, score) in enumerate(final_matches, 1):
            dm_logger.info(
                f"  [{i}] Q{u.id}: '{u.text()}' "
                f"→ D{v.id}: '{v.text()}' "
                f"(score={score:.4f})"
            )
    else:
        dm_logger.info("D-Match 完整结果: 无匹配")

    return final_matches


# cosine判断
# 1. 文本本身之间的余弦相似性
# 2. root节点的文本相似性，可以考虑使用lemma，避免出现太多的根与根的匹配
# 3. 交叉 

# 结合四个通道取个值