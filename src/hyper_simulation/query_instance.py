
from dataclasses import dataclass
from typing import Any, List, Union

@dataclass
class QueryInstance:
    query: str
    data: list[str]
    answers: list[str]
    ground_truth: list[tuple[bool, Any]]  # list of (has_contradiction, evidence)
    fixed_data: Union[List[str], None] = None
    query_decomposition: Union[List[str], None] = None
    
    simulation_logs: Union[List[str], None] = None
    denial_logs: Union[List[str], None] = None
    semantic_cluster_logs: Union[List[str], None] = None
    d_match_logs: Union[List[str], None] = None
    
    def add_simulation_log(self, log: str, data_id: int) -> None:
        if self.simulation_logs is None:
            self.simulation_logs = []
        self.simulation_logs.append(f"[{data_id}] {log}\n")
    
    def add_denial_log(self, log: str, data_id: int) -> None:
        if self.denial_logs is None:
            self.denial_logs = []
        self.denial_logs.append(f"[{data_id}] {log}\n")
        
    def add_semantic_cluster_log(self, log: str, data_id: int) -> None:
        if self.semantic_cluster_logs is None:
            self.semantic_cluster_logs = []
        self.semantic_cluster_logs.append(f"[{data_id}] {log}\n")
    
    def add_d_match_log(self, log: str, data_id: int) -> None:
        if self.d_match_logs is None:
            self.d_match_logs = []
        self.d_match_logs.append(f"[{data_id}] {log}\n")