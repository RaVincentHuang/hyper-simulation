import hashlib
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from hyper_simulation.query_instance import QueryInstance
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.linguistic import Entity, Pos, Dep
from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.embedding import get_embedding_batch, cosine_similarity
from hyper_simulation.utils.log import getLogger
from tqdm import tqdm
from hyper_simulation.utils.log import current_query_id
from hyper_simulation.hypergraph.union import MultiHopFusion
from hyper_simulation.component.nli import init_nli_model
from hyper_simulation.component.embedding import init_embedding_model
from hyper_simulation.question_answer.decompose import decompose_question
import json
import time


TASK_ALIASES = {
    "dconli": "docnli",
    "contrac_nli": "contract_nli",
}

TASK_DEFAULT_INSTANCE_ROOTS = {
    "musique": Path("data/debug/musique/sample1417"),
    "econ": Path("data/debug/econ/sample"),
    "docnli": Path("data/debug/docnli/sample50"),
    "contract_nli": Path("data/debug/contract_nli/sample65"),
}


def normalize_task(task: str) -> str:
    return TASK_ALIASES.get(task, task)


def load_instance_metadata(instance_path: str) -> dict[str, Any]:
    instance_dir = Path(instance_path)
    metadata_path = instance_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in instance directory: {instance_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected metadata.json to contain an object: {metadata_path}")
    return metadata


def resolve_instance_dir(task: str, instance_path: str | None) -> Path:
    task = normalize_task(task)
    candidate_path = Path(instance_path) if instance_path is not None else TASK_DEFAULT_INSTANCE_ROOTS.get(task)
    if candidate_path is None:
        raise ValueError(f"Unsupported task: {task}")

    metadata_path = candidate_path / "metadata.json"
    if metadata_path.exists():
        return candidate_path

    if candidate_path.is_dir():
        instance_dirs = sorted(
            child for child in candidate_path.iterdir() if child.is_dir() and (child / "metadata.json").exists()
        )
        if len(instance_dirs) == 1:
            return instance_dirs[0]
        if len(instance_dirs) > 1:
            print(f"Warning: using first instance directory under {candidate_path}: {instance_dirs[0].name}")
            return instance_dirs[0]

    raise FileNotFoundError(f"No instance directory with metadata.json found at: {candidate_path}")


def extract_query_text(metadata: dict[str, Any], task: str) -> str:
    task = normalize_task(task)
    question = str(metadata.get("question") or "").strip()
    if question:
        return question

    if task == "musique":
        final_answer = str(metadata.get("final_answer") or metadata.get("answer") or "").strip()
        if final_answer:
            return final_answer

    raise ValueError("Unable to extract query text from metadata.json")


def extract_valid_texts(metadata: dict[str, Any], task: str) -> list[str]:
    task = normalize_task(task)

    data_entries = metadata.get("data_entries")
    if isinstance(data_entries, list):
        texts = []
        for entry in data_entries:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text") or "").strip()
            if text:
                texts.append(text)
        if texts:
            return texts

    if task == "musique":
        paragraphs = metadata.get("paragraphs")
        if isinstance(paragraphs, list):
            texts = []
            for paragraph in paragraphs:
                if not isinstance(paragraph, dict):
                    continue
                text = str(paragraph.get("text") or "").strip()
                if text:
                    texts.append(text)
            if texts:
                return texts

    premise_chunks = metadata.get("premise_chunks") or metadata.get("context_docs")
    if isinstance(premise_chunks, list):
        texts = [str(text).strip() for text in premise_chunks if str(text).strip()]
        if texts:
            return texts

    if task == "musique":
        files = metadata.get("files")
        data_files = files.get("data") if isinstance(files, dict) else None
        if isinstance(data_files, list):
            texts = []
            for entry in data_files:
                text = str(entry).strip()
                if text:
                    texts.append(text)
            if texts:
                return texts

    raise ValueError("Unable to extract valid texts from metadata.json")


def load_musique_case(json_path: str) -> tuple[str, list[str], list[int]]:
    """
    Load one MuSiQue item and map fields:
    - query <- question
    - dataset <- paragraphs[*].paragraph_text
    - supports <- question_decomposition[*].paragraph_support_idx
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"MuSiQue input file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    item = raw[0] if isinstance(raw, list) else raw
    if not isinstance(item, dict):
        raise ValueError("Expected a JSON object (or a list whose first element is object).")

    query = item.get("question", "")

    paragraphs = item.get("paragraphs", [])
    dataset = [p.get("paragraph_text", "") for p in paragraphs if isinstance(p, dict)]

    supports_set: set[int] = set()
    for step in item.get("question_decomposition", []):
        if not isinstance(step, dict):
            continue
        paragraph_idx = step.get("paragraph_support_idx")
        if paragraph_idx is None:
            continue
        try:
            supports_set.add(int(paragraph_idx))
        except (TypeError, ValueError):
            continue

    supports = sorted(supports_set)
    return query, dataset, supports

def query_fixup(instance_path: str | None = None, task: str = 'musique'):
    """
    基于 hyper simulation 的一致性修复
    """
    task = normalize_task(task)

    # Load query and data hypergraphs
    # f"{path}/query_hypergraph.pkl"
    # f"{path}/data_hypergraph0.pkl", f"{path}/data_hypergraph1.pkl", ...
    instance_dir = resolve_instance_dir(task, instance_path)
    query_path = instance_dir / "query_hypergraph.pkl"
    if not query_path.exists():
        alt_query_path = instance_dir / "query.pkl"
        if alt_query_path.exists():
            query_path = alt_query_path
    query_hg = LocalHypergraph.load(str(query_path))
    data_hgs_by_index: dict[int, LocalHypergraph] = {}
    for file_path in instance_dir.glob("data_hypergraph*.pkl"):
        match = re.fullmatch(r"data_hypergraph(\d+)\.pkl", file_path.name)
        if match is None:
            continue
        idx = int(match.group(1))
        data_hgs_by_index[idx] = LocalHypergraph.load(str(file_path))

    if not data_hgs_by_index:
        for file_path in instance_dir.glob("data_*.pkl"):
            match = re.fullmatch(r"data_(\d+)\.pkl", file_path.name)
            if match is None:
                continue
            idx = int(match.group(1))
            data_hgs_by_index[idx] = LocalHypergraph.load(str(file_path))
    
    valid_indices = sorted(data_hgs_by_index.keys())
    valid_hgs = [data_hgs_by_index[i] for i in valid_indices]
    
    metadata = load_instance_metadata(str(instance_dir))
    metadata_task = normalize_task(str(metadata.get("task") or task))
    if metadata_task != task:
        print(f"Warning: task argument {task!r} does not match metadata task {metadata_task!r}")

    query_text = extract_query_text(metadata, task)
    valid_texts = extract_valid_texts(metadata, task)
    
    print(f"Query: {query_text}\n")
    
    for i, v in enumerate(query_hg.vertices):
        t = v.type()
        if t:
            print(f" -[{i}]{v.text()} [{t.name}]")
        else:
            print(f" -[{i}]{v.text()}")

    print(f"Valid Texts:")
    for i, text in enumerate(valid_texts):
        print(f"{i}. {text}\n")
        for j, v in enumerate(valid_hgs[i].vertices):
            t = v.type()
            if t:
                print(f" -[{j}]{v.text()} [{t.name}]")
            else:
                print(f" -[{j}]{v.text()}")
    
    
    
    if task in {"docnli", "econ"}: # single-hop tasks, do not need fusion
        mapping, q_map, d_map = compute_hyper_simulation(query_hg, valid_hgs[0])
        print(f"Query doc:{query_text}\n")
        print(f"Data doc:\n{valid_texts[0]}\n")
        print(f"Hyper Simulation Matches:")
        for q_id, d_ids in mapping.items():
            for d_id in d_ids:
                u = q_map[q_id]
                v = d_map[d_id]
                if u.is_verb() or v.is_verb():
                    continue
                print(f"    - {u.text()} <-> {v.text()}")
        return
    
        
    
    
    fusion = MultiHopFusion()
    # decomposes = decompose_question(query_text, query_hg)
    # print("\n=== Decomposed Sub-questions ===")
    # for i, (sub_q, vertex_ids) in enumerate(decomposes):
    #     print(f"[{i + 1}] {sub_q} (related vertex IDs: {sorted(vertex_ids)})")
    
        # merge + reverse trace
        
    context = fusion.process(query_hg, valid_hgs, valid_texts)
    
    # print("\n=== Fusion Result ===")
    # print(context)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyper simulation fix-up with configurable task and instance path.")
    parser.add_argument(
        "--task",
        default="musique",
        choices=["musique", "econ", "docnli", "dconli", "contract_nli", "contrac_nli"],
        help="Task label used to resolve the instance layout",
    )
    parser.add_argument(
        "--instance-path",
        default=None,
        help="Path to an instance directory containing metadata.json and hypergraph files",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Legacy alias for the instance directory or task root",
    )
    args = parser.parse_args()

    instance_path = args.instance_path or args.path

    init_nli_model()
    init_embedding_model()
    time1 = time.time()
    query_fixup(instance_path=instance_path, task=args.task)
    time2 = time.time()
    print(f"Total time taken: {time2 - time1:.2f} seconds")
