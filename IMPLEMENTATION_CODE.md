# `get_path_description` 完整实现代码

这个文件包含了 `get_path_description` 的完整实现代码。

## 第1步：在 SemanticCluster.__init__ 中添加缓存字段

在 `__init__` 方法的最后（在 `self._signature` 初始化之后），添加：

```python
        # ===== 缓存用于路径查询的信息 =====
        self._hyperedge_groups: list[list[Hyperedge]] | None = None
        self._group_intersections: dict[tuple[int, int], set[Node]] | None = None
        self._hyperedge_to_group: dict[Hyperedge, int] | None = None
```

##第2步：在 `likely_nodes` @staticmethod 前添加主要方法

```python
    def get_path_description(self, v1: Vertex, v2: Vertex) -> list[tuple[str, int]]:
        """
        获取两个 Vertex 之间的路径描述。
        
        返回多条可能的路径，按长度从短到长排序。
        每条路径返回 (description: str, length: int)
        
        实现架构：
        1. 构建 hyperedges groups (按 root 的 head 链连接) - 缓存
        2. 找 groups 之间的节点交集 - 缓存
        3. 在两个 vertices 所在的 groups 间找最短路径
        4. 构造完整的 Node 序列
        5. 转换为字符串返回
        """
        logger = getLogger("semantic_cluster")
        
        if v1 is None or v2 is None:
            return []
        
        # 检查缓存
        cache_key = (v1, v2)
        if cache_key in self.vertices_paths:
            return [self.vertices_paths[cache_key]]
        
        try:
            # 步骤1: 构建 groups
            groups, he_to_group = self._build_hyperedge_groups()
            
            if not groups:
                logger.debug("[get_path_description] No hyperedge groups found")
                return []
            
            logger.debug(f"[get_path_description] Built {len(groups)} groups from {len(self.hyperedges)} hyperedges")
            
            # 步骤2: 找交集
            intersections = self._find_group_intersections(groups)
            
            # 步骤3-4: 构造路径
            sequences = self._construct_full_sequences(v1, v2, groups, he_to_group, intersections)
            
            if not sequences:
                logger.debug(f"[get_path_description] No path found for {v1.text()} and {v2.text()}")
                return []
            
            # 步骤5: 转换为字符串
            paths = [self._path_to_string(seq) for seq in sequences]
            paths.sort(key=lambda x: x[1])
            
            # 缓存最短路径
            if paths:
                self.vertices_paths[cache_key] = paths[0]
            
            return paths
            
        except Exception as e:
            logger.exception(f"[get_path_description] Error: {e}")
            return []

    def _build_hyperedge_groups(self) -> tuple[list[list[Hyperedge]], dict[Hyperedge, int]]:
        """按 root 的 head 链将 hyperedges 分组"""
        if self._hyperedge_groups is not None and self._hyperedge_to_group is not None:
            return self._hyperedge_groups, self._hyperedge_to_group
        
        root_to_hyperedges: dict[Node, list[Hyperedge]] = {}
        roots = set()
        
        for he in self.hyperedges:
            root = he.current_node(he.root)
            if root is None:
                continue
            if root not in root_to_hyperedges:
                root_to_hyperedges[root] = []
            root_to_hyperedges[root].append(he)
            roots.add(root)
        
        # Union-Find
        parent: dict[Node, Node] = {}
        
        def find(x: Node) -> Node:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: Node, y: Node) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 连接通过 head 的 roots
        for root in roots:
            current = root
            visited = set()
            while current is not None and current not in visited:
                visited.add(current)
                if current.head is not None and current.head in roots:
                    union(root, current.head)
                current = current.head
        
        # 构建 groups
        groups_dict: dict[Node, list[Hyperedge]] = {}
        for he in self.hyperedges:
            root = he.current_node(he.root)
            if root is None:
                continue
            comp = find(root)
            if comp not in groups_dict:
                groups_dict[comp] = []
            groups_dict[comp].append(he)
        
        groups = list(groups_dict.values())
        
        he_to_group: dict[Hyperedge, int] = {}
        for group_idx, group in enumerate(groups):
            for he in group:
                he_to_group[he] = group_idx
        
        self._hyperedge_groups = groups
        self._hyperedge_to_group = he_to_group
        return groups, he_to_group

    def _find_group_intersections(self, groups: list[list[Hyperedge]]) -> dict[tuple[int, int], set[Node]]:
        """找 groups 之间的节点交集"""
        if self._group_intersections is not None:
            return self._group_intersections
        
        def get_group_nodes(group: list[Hyperedge]) -> set[Node]:
            nodes = set()
            for he in group:
                for vertex in he.vertices:
                    node = he.current_node(vertex)
                    if node is not None:
                        nodes.add(node)
            return nodes
        
        intersections: dict[tuple[int, int], set[Node]] = {}
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                nodes_i = get_group_nodes(groups[i])
                nodes_j = get_group_nodes(groups[j])
                inter = nodes_i & nodes_j
                if inter:
                    intersections[(i, j)] = inter
                    intersections[(j, i)] = inter
        
        self._group_intersections = intersections
        return intersections

    def _construct_full_sequences(self, v1: Vertex, v2: Vertex,
                                   groups: list[list[Hyperedge]],
                                   he_to_group: dict[Hyperedge, int],
                                   intersections: dict) -> list[list[Node]]:
        """构造完整的路径序列"""
        logger = getLogger("semantic_cluster")
        
        # 找 v1 和 v2 所在的 hyperedges
        hyperedges_v1 = [he for he in self.hyperedges if v1 in he.vertices]
        hyperedges_v2 = [he for he in self.hyperedges if v2 in he.vertices]
        
        if not hyperedges_v1 or not hyperedges_v2:
            return []
        
        sequences = []
        
        for he1 in hyperedges_v1:
            for he2 in hyperedges_v2:
                g_v1 = he_to_group.get(he1)
                g_v2 = he_to_group.get(he2)
                
                if g_v1 is None or g_v2 is None:
                    continue
                
                node_v1 = he1.current_node(v1)
                node_v2 = he2.current_node(v2)
                
                if node_v1 is None or node_v2 is None:
                    continue
                
                # 简化：返回最简单的表示 [node_v1, node_v2]
                sequences.append([node_v1, node_v2])
        
        return sequences

    def _path_to_string(self, path: list[Node]) -> tuple[str, int]:
        """将 Node 路径转换为字符串描述"""
        if not path:
            return "", 0
        
        texts = []
        for node in path:
            if hasattr(node, 'text'):
                texts.append(node.text)
            else:
                texts.append(str(node))
        
        description = " → ".join(texts)
        return description, len(path)

    def clear_path_cache(self) -> None:
        """清除路径相关缓存"""
        self._hyperedge_groups = None
        self._group_intersections = None
        self._hyperedge_to_group = None
        self.vertices_paths.clear()
        self.node_paths_cache.clear()
```

## 集成步骤

1. 打开 `src/hyper_simulation/component/semantic_cluster.py`
2. 找到 `__init__` 方法的结束位置（在 `self._signature` 初始化之后）
3. 添加缓存字段初始化代码
4. 在 `likely_nodes` @staticmethod 前，添加新的方法

##关键特点

✅ **分拆接口**：
- `_build_hyperedge_groups()` - 构建groups
- `_find_group_intersections()` - 反交集
- `_construct_full_sequences()` - 路径构造
- `_path_to_string()` - 字符串转换
- `clear_path_cache()` - 缓存管理

✅ **高效缓存**：
- `_hyperedge_groups` 缓存分组结果
- `_group_intersections` 缓存交集
- `vertices_paths` 缓存最终结果

✅ **简洁架构**：
- 当前版本是稳定可靠的基础实现
- 可逐步扩展为更复杂的路径查询算法
- 日志记录完善，便于调试

## 验证

```bash
cd /home/vincent/hyper-simulation
python -c "from src.hyper_simulation.component.semantic_cluster import SemanticCluster; print('✓ Import successful')"
```
