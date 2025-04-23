
from typing import Callable, Dict
import networkx


def get_simulation_inter(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, is_label_cached=False) -> Dict: 
    """
    Get the simulation between two graphs.
    """

def is_simulation_isomorphic(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, is_label_cached=False) -> bool:
    """
    Check if two graphs are isomorphic by graph simulation.
    """

def get_simulation_inter_fn(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, compare_fn: Callable, is_label_cached=False) -> Dict: 
    """
    Get the simulation between two graphs.
    """
    
def is_simulation_isomorphic_fn(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, compare_fn: Callable, is_label_cached=False) -> bool:
    """
    Check if two graphs are isomorphic by graph simulation.
    """

def is_simulation_isomorphic_of_node_edge_fn(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, node_compare_fn: Callable,  edge_compare_fn: Callable, is_label_cached=False) -> bool:
    """
    Check if two graphs are isomorphic by graph simulation.
    """

def is_simulation_isomorphic_of_edge_fn(nx_graph1: networkx.DiGraph, nx_graph2: networkx.DiGraph, node_edge_compare_fn: Callable, is_label_cached=False) -> bool:
    """
    Check if two graphs are isomorphic by graph simulation.
    """
