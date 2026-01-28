
from dataclasses import dataclass
from typing import Any, List, Union

@dataclass
class QueryInstance:
    query: str
    data: list[str]
    answers: list[str]
    ground_truth: list[tuple[bool, Any]]  # list of (has_contradiction, evidence)
    fixed_data: Union[List[str], None] = None
