# `get_path_description` 实现文档

## 概述

`get_path_description(v1: Vertex, v2: Vertex) -> list[tuple[str, int]]` 是 `SemanticCluster` 类的核心方法，用于找出两个 Vertex 之间的所有可能路径，并返回其文本描述和长度。

## 算法设计

### 核心概念

1. **Hyperedge Group**: 一组通过 root 的 head 链相连的 hyperedges，形成一个连通分量。
   - 同一 group 内的 hyperedges 的 roots 通过 head 关系连接
   - 使用 Union-Find 算法高效计算

2. **桥梁节点 (Bridge Nodes)**: 两个不同 groups 之间的公共节点
   - 这些节点是 groups 间的唯一通道
   - 缓存所有可能的桥梁以提高效率

3. **路径构造**: 从 v1 → Group1 → Bridge → Group2 → ... → GroupN → v2 的两层次路径
   - **大步**: 跨越不同 groups 的转移
   - **小步**: 在同一 group 内通过 head 链的追溯

### 算法流程

```
get_path_description(v1, v2)
  ↓
1. _build_hyperedge_groups()  [缓存]
   → 按 root 连接关系分组 hyperedges
   → 返回: (groups, he_to_group_map)
  ↓
2. _find_group_intersections()  [缓存]
   → 计算所有 groups 之间的节点交集
   → 返回: dict[(g1, g2)] = intersection_nodes
  ↓
3. _construct_full_sequences(v1, v2)
   ├─ 找 v1, v2 所在的 hyperedges 和 groups
   ├─ 对每个 (g_v1, g_v2) 对:
   │  ├─ _find_shortest_inter_group_path(g_v1, g_v2)
   │  │  → BFS 找 groups 间最短路径
   │  ├─ 如果同一 group:
   │  │  └─ _trace_node_path(node_v1, node_v2)
   │  │     → 通过公共 ancestor 追溯
   │  └─ 如果不同 groups:
   │     └─ _construct_inter_group_path()
   │        → 通过桥梁节点连接
   └─ 返回: [list[Node], ...]
  ↓
4. _path_to_string(node_path)  ×N
   → 将 Node 序列转换为字符串描述
   → 返回: (description: str, length: int)
  ↓
5. 按长度排序并返回
```

## 实现细节

### 1. `_build_hyperedge_groups()` → `(list[list[Hyperedge]], dict[Hyperedge, int])`

**目的**: 按 root 的 head 连接关系将 hyperedges 分组

**实现**:
- 对每个 hyperedge，提取其 root 节点
- 构建 root_to_hyperedges 映射
- 使用 Union-Find 连接通过 head 关联的 roots
- 按 component 分组并缓存

**复杂度**: O(N × H × D log D)
- N: hyperedges 数量
- H: head 链深度
- D: distinct roots 数量

**缓存**: `self._hyperedge_groups`, `self._hyperedge_to_group`

### 2. `_find_group_intersections()` → `dict[tuple[int, int], set[Node]]`

**目的**: 找所有 groups 之间的公共节点

**实现**:
- 对每个 group，收集其所有节点
- 对每对 groups，计算节点交集
- 双向存储 (i,j) 和 (j,i) 以支持双向查询

**复杂度**: O(G² × N/G) = O(G × N)
- G: groups 数量
- N: 总节点数

**缓存**: `self._group_intersections`

### 3. `_find_shortest_inter_group_path()` → `list[int]`

**目的**: 用 BFS 找两个 groups 之间的最短 group 序列

**实现**:
- 从 g_from 开始 BFS
- 每次只探索有交集的相邻 groups
- 第一次到达 g_to 时即返回（BFS 保证最短）

**复杂度**: O(G + E)
- G: groups 数量
- E: group 之间的边数（受约束）

### 4. `_trace_node_path()` → `list[Node] | None`

**目的**: 在同一 group 内追溯两个 nodes 之间的路径

**实现**:
- 从两个 nodes 分别向上追溯至 root，收集所有 ancestors
- 找公共 ancestor
- 构造路径: node_v1 → ... → ancestor → ... → node_v2

**假设**: 存在公共 ancestor（同一 group 保证）

**复杂度**: O(D) 其中 D 为 head 链深度

### 5. `_construct_inter_group_path()` → `list[Node] | None`

**目的**: 构造跨越多个 groups 的完整路径

**实现**:
```
对 group_path 中的每个相邻 group 对:
  1. 从当前最后节点到该对间的某个桥梁节点
  2. 使用 _trace_node_path 跨越
最后一段: 从最后桥梁到 node_to
```

**关键点**: 在每个转移点选择可达的桥梁节点

### 6. `_path_to_string()` → `tuple[str, int]`

**目的**: 将 Node 序列转换为可读的字符串描述

**实现**:
- 遍历每个 node，提取其 text 属性
- 用 " → " 连接
- 返回 (description, path_length)

## 使用示例

```python
from hyper_simulation.hypergraph.hypergraph import Vertex
from hyper_simulation.component.semantic_cluster import SemanticCluster

# 假设已有 cluster 实例
cluster: SemanticCluster = ...

# 获取两个顶点间的路径
v1: Vertex = ...  # 第一个顶点
v2: Vertex = ...  # 第二个顶点

paths = cluster.get_path_description(v1, v2)
# paths = [
#     ("word1 → word2 → word3", 3),
#     ("word1 → word4 → word3", 3),
#     ("word1 → word5 → word2 → word3", 4),
#     ...
# ]
# 按长度排序，最短的在前
```

## 性能优化

### 缓存策略

| 缓存字段 | 内容 | 清除条件 |
|---------|------|--------|
| `_hyperedge_groups` | 分组后的 hyperedges | hyperedges 修改后 |
| `_group_intersections` | groups 间的交集 | `_hyperedge_groups` 变化 |
| `_hyperedge_to_group` | hyperedge → group_id 映射 | `_hyperedge_groups` 变化 |
| `vertices_paths` | v1→v2 的最短路径 | 调用 `clear_path_cache()` |

### 缓存清除

```python
cluster.clear_path_cache()  # 在 hyperedges 修改后调用
```

### 时间复杂度分析

| 操作 | 第一次调用 | 之后调用 |
|-----|----------|--------|
| 路径查询 | O(G² × N + D × P) | O(1) 缓存命中 |
| group 构建 | O(E log E) | O(1) 缓存 |
| 交集计算 | O(G² × N) | O(1) 缓存 |

- G: groups 数量（通常较小）
- N: 总节点数
- E: hyperedges 数量
- D: head 链深度
- P: 路径长度

## 错误处理

方法使用日志记录处理：

```python
logger = getLogger("semantic_cluster")
logger.debug()   # 调试信息
logger.warning() # 警告（如循环检测）
logger.exception()# 异常信息
```

可能遇到的情况：

1. **无路径**: 两个 vertices 不在任何连通的 groups 中
2. **循环**: head 链存在自环或循环
3. **缺失数据**: hyperedge 或 vertex 数据不完整

所有情况都会返回空列表 `[]`，并记录相应日志。

## 接口总结

```python
class SemanticCluster:
    # 主入口 API
    def get_path_description(self, v1: Vertex, v2: Vertex) -> list[tuple[str, int]]:
        """获取两个 vertices 之间的所有可能路径"""
    
    # 缓存管理
    def clear_path_cache(self) -> None:
        """清除路径相关缓存"""
    
    # 内部接口（不推荐外部使用）
    def _build_hyperedge_groups(self) -> tuple[list[list[Hyperedge]], dict[Hyperedge, int]]:
    def _find_group_intersections(self) -> dict[tuple[int, int], set[Node]]:
    def _find_shortest_inter_group_path(self, g_from: int, g_to: int, 
                                        intersections: dict) -> list[int]:
    def _construct_full_sequences(self, v1: Vertex, v2: Vertex,
                                  groups: list[list[Hyperedge]]) -> list[list[Node]]:
    def _trace_node_path(self, node_from: Node, node_to: Node) -> list[Node] | None:
    def _construct_inter_group_path(self, node_from: Node, node_to: Node,
                                     group_path: list[int],
                                     intersections: dict,
                                     groups: list) -> list[Node] | None:
    def _path_to_string(self, path: list[Node]) -> tuple[str, int]:
```

## 设计特点

1. ✅ **分离关注点**: 每个方法单一职责
2. ✅ **高效缓存**: 避免重复计算 groups 和交集
3. ✅ **清晰分层**: 大步（group）和小步（node）分离
4. ✅ **完整异常处理**: 日志记录所有边界情况
5. ✅ **灵活扩展**: 易于添加新的路径选择策略
6. ✅ **测试友好**: 内部接口可独立测试

## 已知限制

1. 假设 hyperedge.vertices[0] 是 root（首个元素）
2. 假设 head 链不存在循环（已添加循环检测）
3. 性能在 groups 数量很多时可能下降（BFS 复杂度）
4. 不支持加权路径（暂时只考虑最短路由）
