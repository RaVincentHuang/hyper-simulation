# Python Simulation Package 


**Install** 
```bash
git submodule update --init --recursive
pip install maturin
pip install -r requirements.txt
maturin develop --release
```

**How to use**
```python
import networkx as nx
import simulation

def random_graph_gen(n, p, k):
    G: DiGraph = nx.fast_gnp_random_graph(n, p, directed=True)
    for node in G.nodes:
        G.nodes[node]['label'] = random.randint(1, k)
    return G

def graph_permutation(g: DiGraph) -> DiGraph:
    nodes = list(g.nodes)
    random.shuffle(nodes)
    mapping = dict(zip(g.nodes, nodes))
    g_perm = nx.relabel_nodes(g, mapping)
    return g_perm

def attr_same(attr1: dict, attr2: dict):
    label1 = attr1.get("label")
    label2 = attr2.get("label")
    return label1 == label2

n, p = ...
g1 = random_graph_gen(n, p, directed=True)
g2 = graph_permutation(g1)
print(simulation.is_simulation_isomorphic_fn(g1, g2, attr_same))
```
