import time
import os
from typing import Dict, List, Set, Tuple
from itertools import product
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex, Hyperedge
from hyper_simulation.hypergraph.dependency import Node
from simulation import Hypergraph as SimHypergraph, Hyperedge as SimHyperedge, Node as SimNode, Delta, DMatch
from hyper_simulation.component.semantic_cluster import SemanticCluster
from hyper_simulation.hypergraph.linguistic import Pos
from hyper_simulation.component.denial import get_matched_vertices, compute_allowed_pairs, compute_allowed_pairs_batch
from hyper_simulation.hypergraph.path import find_shortest_hyperpaths_bounded, find_shortest_hyperpaths_local_bounded
from hyper_simulation.component.nli import get_nli_labels_batch
import warnings
from tqdm import tqdm
from hyper_simulation.utils.log import getLogger
import logging

def _get_path_description_batch(
    hypergraph: LocalHypergraph,
    pairs: list[tuple[Vertex, Vertex]],
    hops: int,
    max_paths: int = 1000000,
) -> dict[tuple[Vertex, Vertex], str | None]:
    """
    在整个 hypergraph 上批量获取 (v1, v2) 的路径描述。

    逻辑与 SemanticCluster 的 group/intersection/within/across 思路一致：
    1) 按 hyperedge root 的 head 链构建 groups
    2) 计算 groups 间交集节点
    3) 组内路径：node -> 最近公共根 -> node
    4) 组间路径：先走 group 最短路径，再在每跳交集上做组合
    5) 得到 list[list[Node]] 后转成文本

    性能：
    - 使用函数内结构化缓存（group/交集/LCA/group shortest path）
    - 先用 bounded 最短 hyperpath 做 hops 过滤
    """
    if not pairs:
        return {}
    if hops < 0:
        return {pair: None for pair in pairs}

    # 函数内结构化缓存
    desc_cache: dict[tuple[int, int, int], str | None] = {}
    pair_lca_cache: dict[tuple[int, int], Node | None] = {}

    # 去重 pairs，避免重复计算
    unique_pairs = list(dict.fromkeys(pairs))

    # 先命中缓存（当前调用内）
    result: dict[tuple[Vertex, Vertex], str | None] = {}
    uncached_pairs: list[tuple[Vertex, Vertex]] = []
    for v1, v2 in unique_pairs:
        if v1 is None or v2 is None:
            result[(v1, v2)] = None
            continue
        key = (v1.id, v2.id, hops)
        if key in desc_cache:
            result[(v1, v2)] = desc_cache[key]
        else:
            uncached_pairs.append((v1, v2))

    # ---------- 0) 批量 hops 过滤（按经过 hyperedge 数） ----------
    shortest_map = find_shortest_hyperpaths_local_bounded(hypergraph, uncached_pairs, hops) if uncached_pairs else {}

    # ---------- 1) 构建 hyperedge groups（按 founder 分组） ----------
    # founder 定义：从 root.current_node 一直沿 head 向上，直到 None 或 self-loop 停止。
    founder_cache: dict[Node, Node] = {}

    def find_founder(node: Node | None) -> Node | None:
        if node is None:
            return None
        if node in founder_cache:
            return founder_cache[node]

        path: list[Node] = []
        cur = node
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            if cur in founder_cache:
                founder = founder_cache[cur]
                for p in path:
                    founder_cache[p] = founder
                return founder

            visited.add(cur)
            path.append(cur)
            nxt = cur.head
            if nxt is None or nxt == cur:
                founder = cur
                for p in path:
                    founder_cache[p] = founder
                return founder
            cur = nxt

        # 遇到环时，回退为当前节点作为 founder，避免无限循环。
        founder = cur if cur is not None else node
        for p in path:
            founder_cache[p] = founder
        return founder

    groups_dict: dict[Node, list[Hyperedge]] = {}
    for he in hypergraph.hyperedges:
        root_node = he.current_node(he.root)
        if root_node is None:
            continue
        founder = find_founder(root_node)
        if founder is None:
            continue
        groups_dict.setdefault(founder, []).append(he)

    groups = list(groups_dict.values())
    he_to_group = {}
    for gi, group in enumerate(groups):
        for he in group:
            he_to_group[he] = gi

    # 每个 group 的节点集合缓存
    group_nodes: list[set] = []
    for group in groups:
        nodes = set()
        for he in group:
            for vv in he.vertices:
                nn = he.current_node(vv)
                if nn is not None:
                    nodes.add(nn)
        group_nodes.append(nodes)

    # ---------- 辅助：组内路径 ----------
    ancestor_chain_cache: dict[Node, tuple[list[Node], set[Node]]] = {}

    def get_ancestor_chain_and_set(node: Node) -> tuple[list[Node], set[Node]]:
        cached = ancestor_chain_cache.get(node)
        if cached is not None:
            return cached

        chain: list[Node] = []
        chain_set: set[Node] = set()
        cur = node
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            chain.append(cur)
            chain_set.add(cur)
            if cur.head == cur:
                break
            cur = cur.head

        result = (chain, chain_set)
        ancestor_chain_cache[node] = result
        return result

    def nearest_common(node_a: Node, node_b: Node) -> Node | None:
        pair_key: tuple[int, int] = (id(node_a), id(node_b))
        if pair_key in pair_lca_cache:
            return pair_lca_cache[pair_key]

        _, ancestors_a = get_ancestor_chain_and_set(node_a)
        chain_b, _ = get_ancestor_chain_and_set(node_b)

        found = None
        for cur in chain_b:
            if cur in ancestors_a:
                found = cur
                break

        pair_lca_cache[pair_key] = found
        pair_lca_cache[(id(node_b), id(node_a))] = found
        return found

    def path_to_ancestor(node: Node, ancestor: Node) -> list[Node] | None:
        path: list[Node] = []
        cur = node
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            path.append(cur)
            if cur == ancestor:
                return path
            if cur.head == cur:
                break
            cur = cur.head
        return None

    def within_group_path(node_a: Node, node_b: Node, group_idx: int) -> list[Node] | None:
        if node_a not in group_nodes[group_idx] or node_b not in group_nodes[group_idx]:
            return None

        root = nearest_common(node_a, node_b)
        if root is None:
            return [node_a, node_b]

        if root == node_a:
            path = path_to_ancestor(node_b, node_a)
            if path:
                return list(reversed(path))
            return [node_a, node_b]

        if root == node_b:
            path = path_to_ancestor(node_a, node_b)
            if path:
                return path
            return [node_a, node_b]

        path_a: list[Node] = []
        cur = node_a
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            path_a.append(cur)
            if cur == root:
                break
            cur = cur.head
        if not path_a or path_a[-1] != root:
            return None

        path_b: list[Node] = []
        cur = node_b
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            path_b.append(cur)
            if cur == root:
                break
            cur = cur.head
        if not path_b or path_b[-1] != root:
            return None

        tail = list(reversed(path_b[:-1]))
        return path_a + tail

    # ---------- 3) 批量生成 list[list[Node]] 并转文本 ----------
    for v1, v2 in uncached_pairs:
        key = (v1.id, v2.id, hops)

        if v1 == v2:
            desc_cache[key] = v1.text()
            result[(v1, v2)] = desc_cache[key]
            continue

        shortest_hyperedges = shortest_map.get((v1, v2), [])
        # for he in shortest_hyperedges:
        #     he.assert_nodes_reach_root()
        if not shortest_hyperedges or len(shortest_hyperedges) > hops:
            desc_cache[key] = None
            result[(v1, v2)] = None
            continue

        all_segment_paths: list[list[list[Node]]] = []

        # 仅基于 shortest_hyperedges 的有序序列构造路径。
        # 将其切成连续的 group 段：[(gid, [he...]), ...]
        segments: list[tuple[int, list[Hyperedge]]] = []
        valid_shortest = True
        for he in shortest_hyperedges:
            gid = he_to_group.get(he)
            if gid is None:
                valid_shortest = False
                break
            if not segments or segments[-1][0] != gid:
                segments.append((gid, [he]))
            else:
                segments[-1][1].append(he)

        if not valid_shortest or not segments:
            desc_cache[key] = None
            result[(v1, v2)] = None
            continue
        
        def unique_nodes_from_segment(seg_edges: list[Hyperedge]) -> set[Node]:
            nodes: set[Node] = set()
            for he in seg_edges:
                for vv in he.vertices:
                    nn = he.current_node(vv)
                    if nn is not None:
                        nodes.add(nn)
            return nodes

        def vertex_nodes_in_segment(vertex: Vertex, seg_edges: list[Hyperedge]) -> list[Node]:
            nodes: list[Node] = []
            seen = set()
            for he in seg_edges:
                if vertex not in he.vertices:
                    continue
                nn = he.current_node(vertex)
                if nn is None or nn in seen:
                    continue
                seen.add(nn)
                nodes.append(nn)
            return nodes

        # 特殊情况：只有一个 segment 时（单 hops 情况）
        # 直接在该 group 内寻找路径，无需段间连接
        
        # print segments for debugging
        # print(f"Segments for pair ({v1.text()}, {v2.text()}):")
        # for gid, seg_edges in segments:
        #     print(f"  Group {gid}: Hyperedges {[he.text() for he in seg_edges]}")
        
        if len(segments) == 1:
            gid, seg_edges = segments[0]
            start_candidates = vertex_nodes_in_segment(v1, seg_edges)
            end_candidates = vertex_nodes_in_segment(v2, seg_edges)
            
            if not start_candidates or not end_candidates:
                desc_cache[key] = None
                result[(v1, v2)] = None
                continue
            
            # 尝试所有组合
            for start_node in start_candidates:
                for end_node in end_candidates:
                    seg_path = within_group_path(start_node, end_node, gid)
                    if seg_path:
                        all_segment_paths.append([seg_path])
            
            # 同样处理直连候选
            shared_hes = [he for he in shortest_hyperedges if (v1 in he.vertices and v2 in he.vertices)]
            for he in shared_hes:
                n1_direct = he.current_node(v1)
                n2_direct = he.current_node(v2)
                if n1_direct is not None and n2_direct is not None:
                    gid_direct = he_to_group.get(he)
                    if gid_direct is not None:
                        direct_seg_path = within_group_path(n1_direct, n2_direct, gid_direct)
                        if direct_seg_path:
                            all_segment_paths.append([direct_seg_path])
                            continue
                    all_segment_paths.append([[n1_direct, n2_direct]])
            
            if not all_segment_paths:
                desc_cache[key] = None
                result[(v1, v2)] = None
                continue
        else:
            # 多 segment 情况：计算相邻段交集
            connector_lists: list[list[Node]] = []
            segment_node_sets = [unique_nodes_from_segment(seg_edges) for _, seg_edges in segments]
            
            connectors_valid = True
            for i in range(len(segments) - 1):
                inter_nodes = list(segment_node_sets[i] & segment_node_sets[i + 1])
                if not inter_nodes:
                    connectors_valid = False
                    break
                connector_lists.append(inter_nodes)


            if not connectors_valid:
                desc_cache[key] = None
                result[(v1, v2)] = None
                continue

            # 多segment情况的start/end候选
            start_candidates = vertex_nodes_in_segment(v1, segments[0][1])
            end_candidates = vertex_nodes_in_segment(v2, segments[-1][1])
            if not start_candidates or not end_candidates:
                desc_cache[key] = None
                result[(v1, v2)] = None
                continue

            connector_combos = list(product(*connector_lists)) if connector_lists else [()]

            # 一跳直连候选优先保留（仍只来源于 shortest_hyperedges）
            shared_hes = [he for he in shortest_hyperedges if (v1 in he.vertices and v2 in he.vertices)]
            for he in shared_hes:
                n1_direct = he.current_node(v1)
                n2_direct = he.current_node(v2)
                if n1_direct is not None and n2_direct is not None:
                    gid_direct = he_to_group.get(he)
                    if gid_direct is not None:
                        direct_seg_path = within_group_path(n1_direct, n2_direct, gid_direct)
                        if direct_seg_path:
                            all_segment_paths.append([direct_seg_path])
                            continue
                    all_segment_paths.append([[n1_direct, n2_direct]])

            for start_node in start_candidates:
                for end_node in end_candidates:
                    for combo in connector_combos:
                        chain_nodes = [start_node] + list(combo) + [end_node]
                        if len(chain_nodes) != len(segments) + 1:
                            continue

                        seg_paths_for_candidate: list[list[Node]] = []
                        ok = True
                        for si, (gid, _) in enumerate(segments):
                            n_from = chain_nodes[si]
                            n_to = chain_nodes[si + 1]
                            seg_path = within_group_path(chain_nodes[si], chain_nodes[si + 1], gid)
                            if not seg_path:
                                ok = False
                                break
                            seg_paths_for_candidate.append(seg_path)

                        if not ok:
                            continue
                        all_segment_paths.append(seg_paths_for_candidate)

        if not all_segment_paths:
            desc_cache[key] = None
            result[(v1, v2)] = None
            continue

        def candidate_node_cost(seg_paths: list[list[Node]]) -> int:
            cost = 0
            for i, seg in enumerate(seg_paths):
                if i == 0:
                    cost += len(seg)
                else:
                    cost += max(0, len(seg) - 1)
            return cost

        min_cost = min(candidate_node_cost(p) for p in all_segment_paths)
        shortest_paths = [p for p in all_segment_paths if candidate_node_cost(p) == min_cost]
        best_segments = shortest_paths[0]

        # 统一按 node.index 排序，保证 (v,v') 与 (v',v) 的描述构造一致
        best_nodes = [n for seg in best_segments for n in seg if getattr(n, "text", None)]
        best_nodes = sorted(best_nodes, key=lambda n: n.index)
        desc = " ".join(n.text for n in best_nodes)
        # print(f"({v1.text()}, {v2.text()}): {desc}")

        desc_cache[key] = desc if desc else None
        result[(v1, v2)] = desc_cache[key]

    # 补齐重复输入 pair 的返回
    return {pair: result.get(pair) for pair in pairs}

def post_detection(
    query: LocalHypergraph,
    data: LocalHypergraph,
    simulation: list[tuple[Vertex, Vertex]],
    hops: int = 10,
    require_all_neighbors: bool = False,
) -> list[tuple[Vertex, Vertex]]:
    """
    对 simulation 进行后处理检查和构造一致性下的映射。
    
     参数：
     - require_all_neighbors: bool
        * False (默认): 对于(u, v) in match，针对任意同边邻接点 u'：
                   若存在 (u', _) in match，则必须存在 (u', v') in match
                   且 (u, u') 与 (v, v') 匹配；否则删除(u, v)
        * True: 对于(u, v) in match，u 在边中的所有邻接节点 u' 都必须有 (u', _) in match，
                  否则删除(u, v)
    
     步骤：
     1) 枚举 query 超边内的节点对 (u, u')
         - 如果 (u, v), (u', v') 都在 simulation，获取它们的描述
         - 用 NLI 检查 (u, u') 描述 和 (v, v') 描述是否不矛盾
         - 记录匹配关系
     2) 初始化 match = simulation
     3) 进行不动点计算（根据 require_all_neighbors 选择策略）
    """
    if not simulation:
        return []

    debug_postprocess = os.environ.get("POSTPROCESS_DEBUG", "0") not in {"0", "", "false", "False"}
    debug_focus = {
        term.strip().lower()
        for term in os.environ.get("POSTPROCESS_DEBUG_FOCUS", "").split(",")
        if term.strip()
    }
    trace_causal = os.environ.get("POSTPROCESS_TRACE_CAUSAL", "0") not in {"0", "", "false", "False"}
    
    # 初始化
    filtered_simulation = [(u, v) for u, v in simulation if u is not None and v is not None]
    match: set[tuple[Vertex, Vertex]] = set(filtered_simulation)

    # 初始 simulation 的多值索引：u -> {v1, v2, ...}
    simulation_by_u: dict[Vertex, set[Vertex]] = {}
    for u, v in filtered_simulation:
        simulation_by_u.setdefault(u, set()).add(v)

    def _should_debug_pair(u: Vertex, v: Vertex) -> bool:
        if not debug_postprocess:
            return False
        if not debug_focus:
            return True
        u_text = u.text().strip().lower()
        v_text = v.text().strip().lower()
        return u_text in debug_focus or v_text in debug_focus
    
    # -------- 第一阶段：枚举 query 超边，验证一致性 --------
    # 收集所有在超边内配对且都在 simulation 中的四元组 (u, u', v, v')
    # 同时构建邻接关系：每个节点在 query 超边中的邻接节点
    edge_neighbors: dict[Vertex, set[Vertex]] = {}
    uu_vv_quads: list[tuple[Vertex, Vertex, Vertex, Vertex]] = []
    uu_desc_cache: dict[tuple[int, int], str] = {}
    quad_evidence: dict[tuple[int, int, int, int], dict[str, str]] = {}

    def _quad_key(u: Vertex, u_prime: Vertex, v: Vertex, v_prime: Vertex) -> tuple[int, int, int, int]:
        return (u.id, u_prime.id, v.id, v_prime.id)

    def _get_quad_evidence(u: Vertex, u_prime: Vertex, v: Vertex, v_prime: Vertex) -> dict[str, str]:
        direct = quad_evidence.get(_quad_key(u, u_prime, v, v_prime))
        if direct is not None:
            return direct
        reverse = quad_evidence.get(_quad_key(u_prime, u, v_prime, v))
        if reverse is not None:
            return reverse
        return {"reason": "unknown"}

    def _uu_cache_key(u: Vertex, u_prime: Vertex) -> tuple[int, int]:
        return (u.id, u_prime.id) if u.id <= u_prime.id else (u_prime.id, u.id)

    def _path_to_ancestor(node: Node, ancestor: Node) -> list[Node] | None:
        path: list[Node] = []
        cur = node
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            path.append(cur)
            if cur == ancestor:
                return path
            if cur.head == cur:
                break
            cur = cur.head
        return None

    def _nearest_common_query_node(node_a: Node, node_b: Node) -> Node | None:
        ancestors_a: set[Node] = set()
        cur = node_a
        visited: set[Node] = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            ancestors_a.add(cur)
            if cur.head == cur:
                break
            cur = cur.head

        cur = node_b
        visited = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            if cur in ancestors_a:
                return cur
            if cur.head == cur:
                break
            cur = cur.head
        return None

    def _render_query_node_text(node: Node) -> str | None:
        text = getattr(node, "text", None)
        if not text:
            return None
        if not node.is_query:
            return text
        type_str = node.type_str()
        if type_str:
            return f"The {type_str}"
        return "The ATTRIBUTE"

    def _build_uu_desc_from_hyperedge(he: Hyperedge, u: Vertex, u_prime: Vertex) -> str | None:
        n1 = he.current_node(u)
        n2 = he.current_node(u_prime)
        if n1 is None or n2 is None:
            return None

        lca = _nearest_common_query_node(n1, n2)
        if lca is None:
            return None

        p1 = _path_to_ancestor(n1, lca)
        p2 = _path_to_ancestor(n2, lca)
        if not p1 or not p2:
            return None

        seq = p1 + list(reversed(p2[:-1]))
        # 统一按 node.index 排序，保证 (u,u') 与 (u',u) 生成相同描述
        seq = sorted(seq, key=lambda n: n.index)
        tokens = [_render_query_node_text(n) for n in seq]
        desc = " ".join(t for t in tokens if t)
        return desc if desc else None
    
    for he in query.hyperedges:
        vertices = list(he.vertices)
        # 构建邻接关系
        for v in vertices:
            if v not in edge_neighbors:
                edge_neighbors[v] = set()
            for v_other in vertices:
                if v_other != v:
                    edge_neighbors[v].add(v_other)
        # 收集四元组（支持同一 u 的多目标映射）
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, u_prime = vertices[i], vertices[j]

                cache_key = _uu_cache_key(u, u_prime)
                if cache_key not in uu_desc_cache:
                    desc = _build_uu_desc_from_hyperedge(he, u, u_prime)
                    if desc is not None:
                        uu_desc_cache[cache_key] = desc

                v_set = simulation_by_u.get(u)
                v_prime_set = simulation_by_u.get(u_prime)
                if not v_set or not v_prime_set:
                    continue
                for v in v_set:
                    for v_prime in v_prime_set:
                        uu_vv_quads.append((u, u_prime, v, v_prime))
    
    # 对于每个四元组，获取 (v, v') 的路径描述，进行 NLI 检查
    # 记录 (u, u') 可接受的 (v, v')：label != contradiction
    uu_to_vv_match: dict[tuple[Vertex, Vertex], set[tuple[Vertex, Vertex]]] = {}
    
    def _truncate_desc_between_vertices(desc: str, v: Vertex, v_prime: Vertex) -> str | None:
        """
        在描述文本中查找 v 和 v_prime，若两者都存在，则截断：
        - 删去 v 第一次出现的左侧部分
        - 删去 v_prime 最后一次出现的右侧部分
        保留中间部分以减少噪声对 NLI 的影响。
        """
        if not desc:
            return None
        
        v_text = v.text()
        v_prime_text = v_prime.text()
        
        if not v_text or not v_prime_text or v_text == v_prime_text:
            return desc
        
        # 找 v_text 第一次出现的位置
        first_v_pos = desc.find(v_text)
        if first_v_pos == -1:
            return desc
        
        # 找 v_prime_text 最后一次出现的位置
        last_v_prime_pos = desc.rfind(v_prime_text)
        if last_v_prime_pos == -1:
            return desc
        
        # 如果 v 在 v_prime 之后，无法截断，返回原值
        if first_v_pos >= last_v_prime_pos:
            return desc
        
        # 截断：从 v_text 开始到 v_prime_text 结束
        truncated = desc[first_v_pos:last_v_prime_pos + len(v_prime_text)]
        return truncated if truncated else desc
    
    if uu_vv_quads:
        # 获取所有 (v, v') 的路径描述
        v_v_pairs = [(quad[2], quad[3]) for quad in uu_vv_quads]
        path_descs = _get_path_description_batch(data, v_v_pairs, hops)
        # 构造 NLI 检查的文本对（4 种组合：原构造×2方向 + 截断构造×2方向）
        nli_pairs_list: list[tuple[str, str, str, str, int, int]] = []  # 4 个 desc + 两个 quad indices
        valid_quads: list[tuple[Vertex, Vertex, Vertex, Vertex]] = []
        valid_quad_keys: list[tuple[int, int, int, int]] = []
        
        for u, u_prime, v, v_prime in uu_vv_quads:
            # (u, u') 仅使用 query 超边路径描述；无法构造时跳过该四元组
            uu_desc = uu_desc_cache.get(_uu_cache_key(u, u_prime))
            # (v, v') 的描述从路径获取
            vv_desc_original = path_descs.get((v, v_prime))

            qkey = _quad_key(u, u_prime, v, v_prime)
            if uu_desc is None:
                quad_evidence[qkey] = {
                    "reason": "missing_query_path_desc",
                    "detail": f"(u,u')=({u.text()},{u_prime.text()})",
                }
                continue
            if vv_desc_original is None:
                quad_evidence[qkey] = {
                    "reason": "missing_data_path_desc",
                    "detail": f"(v,v')=({v.text()},{v_prime.text()})",
                }
                continue

            if uu_desc is not None and vv_desc_original is not None:
                # 生成截断版本
                vv_desc_truncated = _truncate_desc_between_vertices(vv_desc_original, v, v_prime)
                if vv_desc_truncated is None:
                    vv_desc_truncated = vv_desc_original
                
                quad_idx = len(valid_quads)
                nli_pairs_list.append((vv_desc_original, uu_desc, vv_desc_truncated, uu_desc, quad_idx, quad_idx))
                valid_quads.append((u, u_prime, v, v_prime))
                valid_quad_keys.append(qkey)
        
        # 获取 NLI 标签：4 种组合 (原构造×2方向 + 截断构造×2方向)
        if nli_pairs_list:
            nli_pairs: list[tuple[str, str]] = []
            nli_pair_to_quad_idx: list[tuple[int, str]] = []  # (quad_idx, desc_type)
            
            for vv_orig, uu_desc, vv_trunc, _, quad_idx, _ in nli_pairs_list:
                # 原构造：A->B 和 B->A
                nli_pairs.append((vv_orig, uu_desc))
                nli_pair_to_quad_idx.append((quad_idx, "original_ab"))
                
                nli_pairs.append((uu_desc, vv_orig))
                nli_pair_to_quad_idx.append((quad_idx, "original_ba"))
                
                # 截断构造：A->B 和 B->A
                nli_pairs.append((vv_trunc, uu_desc))
                nli_pair_to_quad_idx.append((quad_idx, "truncated_ab"))
                
                nli_pairs.append((uu_desc, vv_trunc))
                nli_pair_to_quad_idx.append((quad_idx, "truncated_ba"))

            labels = get_nli_labels_batch(nli_pairs)
            
            # 按 quad 聚合 4 个标签
            quad_idx_to_labels: dict[int, dict[str, str]] = {}
            for pair_idx, (quad_idx, desc_type) in enumerate(nli_pair_to_quad_idx):
                if quad_idx not in quad_idx_to_labels:
                    quad_idx_to_labels[quad_idx] = {}
                quad_idx_to_labels[quad_idx][desc_type] = labels[pair_idx]
            
            # 判断矛盾：4 种都是 contradiction 才认为矛盾
            for idx, (u, u_prime, v, v_prime) in enumerate(valid_quads):
                qkey = valid_quad_keys[idx]
                label_dict = quad_idx_to_labels.get(idx, {})
                
                original_ab = label_dict.get("original_ab", "unknown")
                original_ba = label_dict.get("original_ba", "unknown")
                truncated_ab = label_dict.get("truncated_ab", "unknown")
                truncated_ba = label_dict.get("truncated_ba", "unknown")
                
                # 只有 4 个都是 contradiction，才认为是矛盾
                is_all_contradiction = (
                    original_ab == 'contradiction' and 
                    original_ba == 'contradiction' and 
                    truncated_ab == 'contradiction' and 
                    truncated_ba == 'contradiction'
                )
                
                if is_all_contradiction:
                    quad_evidence[qkey] = {
                        "reason": "nli_contradiction",
                        "detail": f"original_ab={original_ab}|original_ba={original_ba}|truncated_ab={truncated_ab}|truncated_ba={truncated_ba}",
                    }
                    # print(f"[NLI 4-way contradiction] ({u.text()}, {u_prime.text()}) <-> ({v.text()}, {v_prime.text()})")
                else:
                    quad_evidence[qkey] = {
                        "reason": "nli_non_contradiction",
                        "detail": f"original_ab={original_ab}|original_ba={original_ba}|truncated_ab={truncated_ab}|truncated_ba={truncated_ba}",
                    }

            # 按新规则更新 uu_to_vv_match
            for idx, (u, u_prime, v, v_prime) in enumerate(valid_quads):
                qkey = valid_quad_keys[idx]
                label_dict = quad_idx_to_labels.get(idx, {})
                
                # 只要不是全都矛盾，就认为非矛盾
                original_ab = label_dict.get("original_ab", "unknown")
                original_ba = label_dict.get("original_ba", "unknown")
                truncated_ab = label_dict.get("truncated_ab", "unknown")
                truncated_ba = label_dict.get("truncated_ba", "unknown")
                
                is_non_contradict = not (
                    original_ab == 'contradiction' and 
                    original_ba == 'contradiction' and 
                    truncated_ab == 'contradiction' and 
                    truncated_ba == 'contradiction'
                )
                
                # 记录非矛盾的匹配
                if is_non_contradict:
                    uu_to_vv_match.setdefault((u, u_prime), set()).add((v, v_prime))
                    uu_to_vv_match.setdefault((u_prime, u), set()).add((v_prime, v))
                    # if debug_postprocess and (_should_debug_pair(u, v) or _should_debug_pair(u_prime, v_prime)):
                    #     print(
                    #         f"[POSTPROCESS DEBUG] NLI ok (4-way): ({u.text()}, {u_prime.text()}) <-> ({v.text()}, {v_prime.text()})"
                    #     )
                # elif debug_postprocess and (_should_debug_pair(u, v) or _should_debug_pair(u_prime, v_prime)):
                #     print(
                #         f"[POSTPROCESS DEBUG] NLI 4-way contradiction: ({u.text()}, {u_prime.text()}) <-> ({v.text()}, {v_prime.text()})"
                #     )
    
    # -------- 第二阶段：worklist 不动点计算 --------
    # 仅重检受影响的 u，避免每轮全量扫描 match。

    # u -> 当前仍在 match 中的 (u, v) 集合
    match_by_u: dict[Vertex, set[tuple[Vertex, Vertex]]] = {}
    for u, v in match:
        match_by_u.setdefault(u, set()).add((u, v))

    # 反向邻接：当某个 u 被删时，哪些中心节点会受影响
    reverse_neighbors: dict[Vertex, set[Vertex]] = {}
    for center_u, neighbors in edge_neighbors.items():
        for neighbor_u in neighbors:
            reverse_neighbors.setdefault(neighbor_u, set()).add(center_u)

    deleted_pair_causes: dict[tuple[int, int], dict[str, object]] = {}
    deleted_pairs_by_u: dict[Vertex, list[tuple[str, str]]] = {}

    def _print_pair_debug_header(u: Vertex, v: Vertex) -> None:
        print(f"\n[POSTPROCESS DEBUG] pair=({u.text()}, {v.text()})")
        print(f"  query neighbors: {[n.text() for n in sorted(edge_neighbors.get(u, set()), key=lambda x: x.text())]}")
        print("  current neighbors in match:")
        for neighbor in sorted(edge_neighbors.get(u, set()), key=lambda x: x.text()):
            neighbor_pairs = match_by_u.get(neighbor, set())
            formatted_pairs = [
                (u2.text(), v2.text())
                for u2, v2 in sorted(neighbor_pairs, key=lambda p: (p[0].text(), p[1].text()))
            ]
            print(f"    - {neighbor.text()}: {formatted_pairs}")

    def should_remove_pair(u: Vertex, v: Vertex) -> tuple[bool, dict[str, object] | None]:
        debug_this_pair = _should_debug_pair(u, v)
        # if debug_this_pair:
            # _print_pair_debug_header(u, v)

        if require_all_neighbors:
            # 模式2：u 的所有同边邻接节点都必须存在 (u', _) in match
            for u_neighbor in edge_neighbors.get(u, set()):
                if not match_by_u.get(u_neighbor):
                    reason: dict[str, object] = {
                        "reason": "missing_neighbor_pairs",
                        "neighbor": u_neighbor.text(),
                    }
                    if trace_causal:
                        recent_deleted = deleted_pairs_by_u.get(u_neighbor, [])[-5:]
                        if recent_deleted:
                            reason["upstream_deleted"] = [
                                f"({u_neighbor.text()}, {v_text}): {r_text}"
                                for v_text, r_text in recent_deleted
                            ]
                    # if debug_this_pair:
                    #     print(f"  [REMOVE] missing neighbor support: {u_neighbor.text()} has no surviving match")
                    #     if trace_causal:
                    #         print(f"  [CAUSE] {reason}")
                    return True, reason
            return False, None

        # 模式1：对任意同边邻接点 u'，若存在 (u', _) in match，
        # 必须存在 v' 使得 (v, v') 与 (u, u') 的关系非矛盾。
        for u_prime in edge_neighbors.get(u, set()):
            u_prime_pairs = match_by_u.get(u_prime)
            if not u_prime_pairs:
                # if debug_this_pair:
                #     print(f"  [SKIP] neighbor {u_prime.text()} has no surviving pairs")
                continue

            allowed_vv = uu_to_vv_match.get((u, u_prime), set())
            has_support = False
            candidate_failures: list[str] = []
            for _, v_prime in u_prime_pairs:
                if (v, v_prime) in allowed_vv:
                    has_support = True
                    # if debug_this_pair:
                    #     print(f"  [SUPPORT] neighbor={u_prime.text()} supports via ({v.text()}, {v_prime.text()})")
                    break
                evidence = _get_quad_evidence(u, u_prime, v, v_prime)
                candidate_failures.append(
                    f"({v.text()}, {v_prime.text()}): {evidence.get('reason', 'unknown')}"
                )
            if not has_support:
                reason: dict[str, object] = {
                    "reason": "no_support_from_neighbor",
                    "neighbor": u_prime.text(),
                    "candidate_failures": candidate_failures[:12],
                }
                # if debug_this_pair:
                #     print(f"  [REMOVE] no support from neighbor {u_prime.text()}")
                #     print(
                #         f"  allowed pairs for ({u.text()}, {u_prime.text()}): {[(v1.text(), v2.text()) for v1, v2 in sorted(allowed_vv, key=lambda p: (p[0].text(), p[1].text()))]}"
                #     )
                #     print(
                #         f"  surviving pairs for {u_prime.text()}: {[(u2.text(), v2.text()) for u2, v2 in sorted(u_prime_pairs, key=lambda p: (p[0].text(), p[1].text()))]}"
                #     )
                #     if trace_causal:
                #         print(f"  [CAUSE] {reason}")
                return True, reason
        return False, None

    # 初始将所有可能受约束的 u 入队
    dirty_u: set[Vertex] = set(match_by_u.keys())
    
    while dirty_u:
        to_remove: dict[tuple[Vertex, Vertex], dict[str, object]] = {}

        # 只扫描脏节点对应的映射对
        for u in dirty_u:
            for pair in list(match_by_u.get(u, set())):
                pu, pv = pair
                should_remove, cause = should_remove_pair(pu, pv)
                if should_remove:
                    to_remove[pair] = cause or {"reason": "unknown"}

        if not to_remove:
            # if debug_postprocess:
            #     print(f"[POSTPROCESS DEBUG] fixed point reached with {len(match)} pairs")
            break

        next_dirty_u: set[Vertex] = set()
        for (u, v), cause in to_remove.items():
            if (u, v) not in match:
                continue
            match.remove((u, v))
            cause_payload = dict(cause)
            deleted_pair_causes[(u.id, v.id)] = cause_payload
            deleted_pairs_by_u.setdefault(u, []).append((v.text(), str(cause.get("reason", "unknown"))))
            # if trace_causal and (debug_postprocess and _should_debug_pair(u, v)):
            #     print(f"[POSTPROCESS CAUSAL] removed ({u.text()}, {v.text()}) -> {cause}")

            if u in match_by_u:
                match_by_u[u].discard((u, v))
                if not match_by_u[u]:
                    del match_by_u[u]

            # 删除 (u, v) 会影响依赖 u 的中心节点；u 自身也可能仍有其它候选需要重检
            next_dirty_u.update(reverse_neighbors.get(u, set()))
            if u in match_by_u:
                next_dirty_u.add(u)

        dirty_u = next_dirty_u
    
    # if debug_postprocess:
    #     print(f"[POSTPROCESS DEBUG] final match size: {len(match)}")

    return list(match)

def get_simulation_slice(query: LocalHypergraph, data: LocalHypergraph, simulation: list[tuple[Vertex, Vertex]], num: int) -> list[list[tuple[Vertex, Vertex]]]:
    """
    基于 Vertex 的 provenance 信息，将 simulation 切割为各个原始 hypergraph 下的切片。
    
    u 来自 query（单一来源），v 来自 data（可能属于多个原始 hypergraph）。
    对于 simulation 中的每个 (u, v) 对，根据 v 的 provenance 确定该对属于哪些原始 hypergraph。
    
    参数：
    - query: 查询的 LocalHypergraph
    - data: 数据的 LocalHypergraph
    - simulation: Vertex 对的匹配列表
    - num: 原始 hypergraph 的总数量（从1到num）
    
    返回：
    - list[list[tuple[Vertex, Vertex]]]: 长度为 num 的列表，
      其中索引 i 对应第 (i+1) 个原始 hypergraph 的 simulation 切片
    """
    # 初始化结果：每个原始 hypergraph 对应一个空列表
    slices = [[] for _ in range(num)]
    
    # 遍历 simulation 中的每个 (u, v) 对
    for u, v in simulation:
        if u is None or v is None:
            continue
        
        # 获取 v 的 provenance（所属的原始 hypergraph id 集合）
        v_provenance = v.get_provenance()
        
        # 将该对添加到 v 所属的所有原始 hypergraph 对应的切片中
        for hg_id in v_provenance:
            # hg_id 从1开始，数组索引从0开始，所以需要减1
            slices[hg_id - 1].append((u, v))
    
    return slices

def check_slice_consistency(query: LocalHypergraph, simulation_slice: list[tuple[Vertex, Vertex]], vertex_ids: set[int]) -> bool:
    """
    检查 simulation_slice 中的 (u, v) 对是否满足一致性要求：
对于 query 中 id 在 vertex_ids 内的每个 vertex u，simulation_slice 中至少存在一个 v 使得 (u, v) 在其中。
    
    返回 True 如果满足一致性
    """
    
    vertex_map: dict[Vertex, set[Vertex]] = {}
    for u, v in simulation_slice:
        if u not in vertex_map:
            vertex_map[u] = set()
        vertex_map[u].add(v)
    
    vertex_needs: set[Vertex] = set()
    
    for u in query.vertices:
        if u.id in vertex_ids:
            # print(f"- [{u.id}] {u.text()}")
            vertex_needs.add(u)
    # returns True if for all u in vertex_needs, there exists a v such that (u, v) in simulation_slice
    hit_cnt = 0
    for u in vertex_needs:
        if u in vertex_map and len(vertex_map[u]) > 0:
            hit_cnt += 1
    return hit_cnt == len(vertex_needs)

def refine_simulation_slices(query: LocalHypergraph, simulation_slices: list[list[tuple[Vertex, Vertex]]], answer: str) -> list[list[tuple[Vertex, Vertex]]]:
    # 基于 answer 对 simulation_slices 进行进一步过滤
    # 独立操作每个 slice。
    
    def _match(v_text: str, answer: str) -> bool:
        # 简单的文本匹配函数，判断 v_text 是否与 answer 匹配
        # 这里可以使用更复杂的匹配逻辑，例如包含关系、同义词等
        return v_text.strip().lower() == answer.strip().lower()
    
    refined_slices: list[list[tuple[Vertex, Vertex]]] = []
    for slice in simulation_slices:
        new_slice: list[tuple[Vertex, Vertex]] = []
        # 首先检查 slice 内是否有 (u, v) 满足 v.text() 和 answer 匹配
        # 若存在匹配，则 保留 (u, v), 而删除其他的 (u, _)
        # 否则不进行修改
        matched_map: dict[Vertex, Vertex] = {}
        for u, v in slice:
            if v is not None and _match(v.text(), answer):
                matched_map[u] = v
                
        for u, v in matched_map.items():
            new_slice.append((u, v))

        for u, v in slice:
            if u in matched_map:
                continue
            new_slice.append((u, v))
                
        refined_slices.append(new_slice)
    return refined_slices

def ranking_slices(query: LocalHypergraph, simulation_slices: list[list[tuple[Vertex, Vertex]]], vertex_ids: set[int], k: int) -> list[int]:
    """
    对 simulation_slices 做 soft ranking。

    评分定义（精确整数比较，避免浮点误差）：
    - score = hit_cnt / len(vertex_needs)
    - 因为 len(vertex_needs) 对所有 slice 恒定，排序可等价为按 hit_cnt 排序
    - hit_cnt 为该 slice 中命中的 query 目标顶点数量
    - vertex_needs 为 query 中 id 在 vertex_ids 内的顶点集合

    返回：
    - 按得分从高到低排序后的 slice 索引列表
    - 取前 k 时若截断同分项，则保留所有与第 k 名同分的 slice（可能超过 k）
    """
    if k <= 0 or not simulation_slices:
        return []

    vertex_needs: set[Vertex] = {u for u in query.vertices if u.id in vertex_ids}
    need_cnt = len(vertex_needs)

    scored_indices: list[tuple[int, int]] = []
    for idx, simulation_slice in enumerate(simulation_slices):
        present_u: set[Vertex] = {u for u, _ in simulation_slice if u is not None}
        if need_cnt == 0:
            hit_cnt = 1
        else:
            hit_cnt = sum(1 for u in vertex_needs if u in present_u)
        scored_indices.append((idx, hit_cnt))

    # 先按命中数降序，再按 index 升序，保证同分时结果稳定。
    scored_indices.sort(key=lambda x: (-x[1], x[0]))

    if len(scored_indices) <= k:
        return [idx for idx, _ in scored_indices]

    kth_hit_cnt = scored_indices[k - 1][1]
    return [idx for idx, hit_cnt in scored_indices if hit_cnt >= kth_hit_cnt]
    