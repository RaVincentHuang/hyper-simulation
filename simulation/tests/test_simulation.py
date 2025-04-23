from math import e
import time
import simulation
import networkx as nx
import os

# Get the directory where the current file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

simulation_test_path = os.path.join(current_dir, "lib/graph-simulation/data/label_graph/simulation_test")
# Read all files in simulation_test_path
files = []
if os.path.exists(simulation_test_path):
    for file_name in os.listdir(simulation_test_path):
        file_path = os.path.join(simulation_test_path, file_name)
        if os.path.isfile(file_path):
            files.append(file_path)

def attr_same(attr1: dict, attr2: dict):
    label1 = attr1.get("label")
    label2 = attr2.get("label")
    return label1 == label2

times_cost = 0

for path in files:
    with open(path, "r") as f:
        lines = f.readlines()
        # Get the first line of the file
        is_isomorphic = True if lines[0].strip() == "t" else False
        
        index = 1
        # Get the first graph
        graph1 = nx.DiGraph()
        graph1_nodes, graph1_edges, graph1_labels = map(int, lines[index].strip().split())
        index += 1
        for i in range(index, index + graph1_nodes):
            idx, label = lines[i].strip().split()
            graph1.add_node(idx, label=label)
            
        index += graph1_nodes
        
        for i in range(index, index + graph1_edges):
            u, v = map(int, lines[i].strip().split())
            graph1.add_edge(u, v)
        
        index += graph1_edges
        
        # Get the second graph
        
        graph2 = nx.DiGraph()
        
        graph2_nodes, graph2_edges, graph2_labels = map(int, lines[index].strip().split())
        index += 1
        for i in range(index, index + graph2_nodes):
            idx, label = lines[i].strip().split()
            graph2.add_node(idx, label=label)
            
        index += graph2_nodes
        
        for i in range(index, index + graph2_edges):
            u, v = map(int, lines[i].strip().split())
            graph2.add_edge(u, v)
            
        
        start = time.time()
        is_simulation = simulation.is_simulation_isomorphic_fn(graph1, graph2, attr_same)
        end = time.time()
        times_cost += end - start
        
        if is_isomorphic != is_simulation and is_isomorphic:
            print("Test failed for file", path)


print("Time taken:", times_cost)