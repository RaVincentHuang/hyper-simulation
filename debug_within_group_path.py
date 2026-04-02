#!/usr/bin/env python3
"""
调试脚本：测试 _get_path_description_batch 和 within_group_path
"""
import sys
import os

# 添加 src 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from hyper_simulation.hypergraph.hypergraph import Hypergraph
from hyper_simulation.component.postprocess import _get_path_description_batch

# 加载测试数据
query_file = "data/hypergraph/query_hypergraph.pkl"
data_file = "data/hypergraph/data_hypergraph.pkl"

print("Loading hypergraphs...")
data_hypergraph = Hypergraph.load(data_file)

# 从 data_hypergraph 中随机选择几对顶点来测试
vertices = list(data_hypergraph.vertices)
print(f"Total vertices in data hypergraph: {len(vertices)}")

# 选择一些测试对
test_pairs = []
for i in range(min(5, len(vertices))):
    for j in range(i + 1, min(i + 3, len(vertices))):
        test_pairs.append((vertices[i], vertices[j]))

print(f"\nTesting {len(test_pairs)} pairs...")
print("=" * 80)

# 调用 _get_path_description_batch
results = _get_path_description_batch(data_hypergraph, test_pairs, hops=4)

print("\n" + "=" * 80)
print("\nResults:")
for (v1, v2), desc in results.items():
    print(f"({v1.text()}, {v2.text()}): {desc}")
