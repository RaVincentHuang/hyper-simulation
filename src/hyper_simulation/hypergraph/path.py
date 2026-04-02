from collections import deque, defaultdict
from typing import Dict, List, Tuple, Union
from hyper_simulation.hypergraph.hypergraph import Hyperedge, Hypergraph, Vertex


def find_shortest_hyperpaths(
    hypergraph: Hypergraph,
    pairs: List[Tuple[Vertex, Vertex]]
) -> Dict[Tuple[Vertex, Vertex], List[Hyperedge]]:
    """返回每对 (u, v) 的一条最短超路径（以超边列表表示）。
    - []    : u == v 或 u 与 v 不可达
    - [e…]  : 按从 u 到 v 顺序排列的最短超边序列
    注：返回所有最短路径的条数在最坏情况下指数级增长（同一节点可属于多条超边），
    故此处只保证返回一条最短路径。
    """

    # 1. 预处理：按起点 u 对查询进行分组，避免重复执行相同起点的 BFS
    queries_by_source = defaultdict(list)
    for u, v in pairs:
        queries_by_source[u].append(v)

    result = {}

    # 2. 对每个唯一的起点执行 BFS V_souces(V_d+E_d)
    for u, targets in queries_by_source.items():
        # distances 用于避免重复访问，parents 用于回溯路径
        distances = {u: 0}
        parents: Dict[Union[Vertex, Hyperedge], Union[Vertex, Hyperedge, None]] = {u: None}
        queue = deque([u])

        # 在隐式二分图上运行 BFS
        while queue:
            curr = queue.popleft()

            if isinstance(curr, Vertex):
                # 如果当前是节点，扩展其所属的所有超边
                for edge in hypergraph.contained_edges.get(curr, []):
                    if edge not in distances:
                        distances[edge] = distances[curr] + 1
                        # parents[edge] = curr
                        parents[edge] = curr
                        queue.append(edge)
            elif isinstance(curr, Hyperedge):
                # 如果当前是超边，扩展其包含的所有节点
                for vertex in curr.vertices:
                    if vertex not in distances:
                        distances[vertex] = distances[curr] + 1
                        parents[vertex] = curr
                        queue.append(vertex)

        # 3. 为该起点的所有目标查询回溯路径
        for v in targets:
            if v not in distances:
                # 节点不可达
                result[(u, v)] = []
            elif u == v:
                # 起点和终点相同，路径长度为0 (不包含任何超边)
                result[(u, v)] = []
            else:
                path = []
                curr_node = v
                # 从终点回溯到起点
                while curr_node is not None:
                    # 二分图路径中，交替出现 Vertex 和 Hyperedge
                    # 我们只需要提取出 Hyperedge 作为最终结果
                    if isinstance(curr_node, Hyperedge):
                        path.append(curr_node)
                    curr_node = parents[curr_node]
                
                # 反转路径，使其变为从起点到终点的顺序
                path.reverse()
                result[(u, v)] = path

    return result

def find_shortest_hyperpaths_local(
    hypergraph: Hypergraph,
    pairs: List[Tuple[Vertex, Vertex]]
) -> Dict[Tuple[Vertex, Vertex], List[Hyperedge]]:
    """求局部最短超路径：路径上的所有超边必须具有相同的 hypergraph_id。

    返回值语义与 find_shortest_hyperpaths 一致：
    - []   : u == v 或不可达
    - [e…] : 一条最短的局部路径（所有 e.hypergraph_id 相同）
    """

    queries_by_source = defaultdict(list)
    for u, v in pairs:
        queries_by_source[u].append(v)

    result: Dict[Tuple[Vertex, Vertex], List[Hyperedge]] = {}

    for u, targets in queries_by_source.items():
        # 状态 = (节点/超边, 组件ID)。组件ID在走第一条边后确定并保持不变。
        unset_component = object()
        start_state = (u, unset_component)

        distances: dict[tuple[Union[Vertex, Hyperedge], object], int] = {start_state: 0}
        parents: dict[
            tuple[Union[Vertex, Hyperedge], object],
            tuple[Union[Vertex, Hyperedge], object] | None,
        ] = {start_state: None}
        queue = deque([start_state])

        # 记录每个目标顶点首次被发现的有效状态（组件已锁定），用于回溯路径。
        wanted = {v for v in targets if v != u}
        best_target_state: dict[Vertex, tuple[Union[Vertex, Hyperedge], object]] = {}

        while queue:
            curr_obj, curr_component = queue.popleft()

            if isinstance(curr_obj, Vertex):
                for edge in hypergraph.contained_edges.get(curr_obj, []):
                    edge_component = edge.hypergraph_id
                    if curr_component is not unset_component and edge_component != curr_component:
                        continue

                    next_state = (edge, edge_component)
                    if next_state in distances:
                        continue
                    distances[next_state] = distances[(curr_obj, curr_component)] + 1
                    parents[next_state] = (curr_obj, curr_component)
                    queue.append(next_state)
            else:
                # 当前是超边，只能走到同组件下的顶点状态
                for vertex in curr_obj.vertices:
                    next_state = (vertex, curr_component)
                    if next_state in distances:
                        continue
                    distances[next_state] = distances[(curr_obj, curr_component)] + 1
                    parents[next_state] = (curr_obj, curr_component)
                    queue.append(next_state)

                    # BFS 首次到达即该组件下最短，且在扩展状态图中也是全局最短。
                    if vertex in wanted and vertex not in best_target_state:
                        best_target_state[vertex] = next_state

            if len(best_target_state) == len(wanted):
                break

        for v in targets:
            if u == v:
                result[(u, v)] = []
                continue

            terminal_state = best_target_state.get(v)
            if terminal_state is None:
                result[(u, v)] = []
                continue

            path: list[Hyperedge] = []
            state: tuple[Union[Vertex, Hyperedge], object] | None = terminal_state
            while state is not None:
                obj, _ = state
                if isinstance(obj, Hyperedge):
                    path.append(obj)
                state = parents[state]

            path.reverse()
            result[(u, v)] = path

    return result


def find_shortest_hyperpaths_bounded(
    hypergraph: Hypergraph,
    pairs: List[Tuple[Vertex, Vertex]],
    max_hops: int,
) -> Dict[Tuple[Vertex, Vertex], List[Hyperedge]]:
    """返回每对 (u, v) 在 max_hops 内的一条最短超路径。

    - []   : u == v、不可达、或最短路径跳数 > max_hops
    - [e…] : 一条最短路径（边数即跳数）
    """
    if max_hops < 0:
        return {(u, v): [] for u, v in pairs}

    queries_by_source = defaultdict(list)
    for u, v in pairs:
        queries_by_source[u].append(v)

    result: Dict[Tuple[Vertex, Vertex], List[Hyperedge]] = {}

    for u, targets in queries_by_source.items():
        # 仅在 Vertex 层做 BFS，单次过边 hop+1
        dist_hops: dict[Vertex, int] = {u: 0}
        parent: dict[Vertex, tuple[Vertex, Hyperedge] | None] = {u: None}
        queue = deque([u])

        wanted = {v for v in targets if v != u}
        found: set[Vertex] = set()

        while queue:
            curr = queue.popleft()
            curr_hops = dist_hops[curr]

            if curr_hops >= max_hops:
                continue

            for edge in hypergraph.contained_edges.get(curr, []):
                for nxt in edge.vertices:
                    if nxt == curr:
                        continue
                    nxt_hops = curr_hops + 1
                    if nxt_hops > max_hops:
                        continue
                    if nxt not in dist_hops:
                        dist_hops[nxt] = nxt_hops
                        parent[nxt] = (curr, edge)
                        queue.append(nxt)
                        if nxt in wanted:
                            found.add(nxt)

            if len(found) == len(wanted):
                break

        for v in targets:
            if u == v:
                result[(u, v)] = []
                continue
            if v not in dist_hops:
                result[(u, v)] = []
                continue

            path: list[Hyperedge] = []
            cur = v
            while cur != u:
                prev_info = parent.get(cur)
                if prev_info is None:
                    path = []
                    break
                prev, edge = prev_info
                path.append(edge)
                cur = prev
            path.reverse()
            result[(u, v)] = path

    return result


def find_shortest_hyperpaths_local_bounded(
    hypergraph: Hypergraph,
    pairs: List[Tuple[Vertex, Vertex]],
    max_hops: int,
) -> Dict[Tuple[Vertex, Vertex], List[Hyperedge]]:
    """返回每对 (u, v) 在 max_hops 内的一条局部最短超路径（同 hypergraph_id 约束）。

    - []   : u == v、不可达、或最短路径跳数 > max_hops
    - [e…] : 一条满足同组件约束的最短路径
    """
    if max_hops < 0:
        return {(u, v): [] for u, v in pairs}

    queries_by_source = defaultdict(list)
    for u, v in pairs:
        queries_by_source[u].append(v)

    result: Dict[Tuple[Vertex, Vertex], List[Hyperedge]] = {}

    for u, targets in queries_by_source.items():
        unset_component = object()
        start_state = (u, unset_component)

        dist_hops: dict[tuple[Vertex, object], int] = {start_state: 0}
        parent: dict[tuple[Vertex, object], tuple[tuple[Vertex, object], Hyperedge] | None] = {start_state: None}
        queue = deque([start_state])

        wanted = {v for v in targets if v != u}
        found_state: dict[Vertex, tuple[Vertex, object]] = {}

        while queue:
            curr_v, curr_comp = queue.popleft()
            curr_hops = dist_hops[(curr_v, curr_comp)]

            if curr_hops >= max_hops:
                continue

            for edge in hypergraph.contained_edges.get(curr_v, []):
                edge_comp = edge.hypergraph_id
                if curr_comp is not unset_component and edge_comp != curr_comp:
                    continue
                next_comp = edge_comp
                for nxt in edge.vertices:
                    if nxt == curr_v:
                        continue
                    next_state = (nxt, next_comp)
                    nxt_hops = curr_hops + 1
                    if nxt_hops > max_hops:
                        continue
                    if next_state in dist_hops:
                        continue
                    dist_hops[next_state] = nxt_hops
                    parent[next_state] = ((curr_v, curr_comp), edge)
                    queue.append(next_state)
                    if nxt in wanted and nxt not in found_state:
                        found_state[nxt] = next_state

            if len(found_state) == len(wanted):
                break

        for v in targets:
            if u == v:
                result[(u, v)] = []
                continue
            terminal = found_state.get(v)
            if terminal is None:
                result[(u, v)] = []
                continue

            path: list[Hyperedge] = []
            state = terminal
            while state != start_state:
                prev_info = parent.get(state)
                if prev_info is None:
                    path = []
                    break
                prev_state, edge = prev_info
                path.append(edge)
                state = prev_state
            path.reverse()
            result[(u, v)] = path

    return result
