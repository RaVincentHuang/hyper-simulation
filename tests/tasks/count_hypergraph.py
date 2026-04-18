from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import fmean

from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph


def _sorted_index_from_name(path: Path) -> int:
    match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
    if match_obj is not None:
        return int(match_obj.group(1))
    match_obj = re.fullmatch(r"data_(\d+)\.pkl", path.name)
    if match_obj is not None:
        return int(match_obj.group(1))
    return 10**9


def _find_query_file(instance_dir: Path) -> Path | None:
    for name in ("query_hypergraph.pkl", "query.pkl"):
        candidate = instance_dir / name
        if candidate.exists():
            return candidate
    return None


def _list_data_files(instance_dir: Path) -> list[Path]:
    all_files: dict[str, Path] = {}
    for pattern in ("data_hypergraph*.pkl", "data_*.pkl"):
        for path in instance_dir.glob(pattern):
            all_files[path.name] = path
    return sorted(all_files.values(), key=lambda path: (_sorted_index_from_name(path), path.name))


def _collect_instance_dirs(root: Path) -> list[Path]:
    instance_dirs: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if _find_query_file(path) is None:
            continue
        instance_dirs.append(path)
    return sorted(instance_dirs)


def _count_instance(instance_dir: Path) -> dict[str, int] | None:
    query_path = _find_query_file(instance_dir)
    if query_path is None:
        return None

    data_paths = _list_data_files(instance_dir)

    try:
        query_hg = LocalHypergraph.load(str(query_path))
    except Exception as exc:
        print(f"[Skip] {instance_dir}: failed to load query hypergraph ({type(exc).__name__})")
        return None

    data_vertices_sum = 0
    data_hyperedges_sum = 0
    loaded_data_count = 0
    for data_path in data_paths:
        try:
            data_hg = LocalHypergraph.load(str(data_path))
            data_vertices_sum += len(data_hg.vertices)
            data_hyperedges_sum += len(data_hg.hyperedges)
            loaded_data_count += 1
        except Exception as exc:
            print(f"[Warn] {instance_dir}: failed to load {data_path.name} ({type(exc).__name__})")

    return {
        "query_vertices": len(query_hg.vertices),
        "query_hyperedges": len(query_hg.hyperedges),
        "query_hypergraph_count": 1,
        "data_vertices_sum": data_vertices_sum,
        "data_hyperedges_sum": data_hyperedges_sum,
        "data_hypergraph_count": loaded_data_count,
    }


def count_hypergraphs(root_dir: str) -> None:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    instance_dirs = _collect_instance_dirs(root)
    if not instance_dirs:
        raise FileNotFoundError(f"No instance directory with query hypergraph found under: {root}")

    per_instance_stats: list[dict[str, int]] = []
    for instance_dir in instance_dirs:
        stat = _count_instance(instance_dir)
        if stat is not None:
            per_instance_stats.append(stat)

    if not per_instance_stats:
        raise RuntimeError("No valid instance could be loaded.")

    avg_query_vertices = fmean(stat["query_vertices"] for stat in per_instance_stats)
    avg_query_hyperedges = fmean(stat["query_hyperedges"] for stat in per_instance_stats)
    avg_query_hg_count = fmean(stat["query_hypergraph_count"] for stat in per_instance_stats)
    avg_data_vertices_sum = fmean(stat["data_vertices_sum"] for stat in per_instance_stats)
    avg_data_hyperedges_sum = fmean(stat["data_hyperedges_sum"] for stat in per_instance_stats)
    avg_data_hg_count = fmean(stat["data_hypergraph_count"] for stat in per_instance_stats)

    print("=" * 60)
    print(f"Root: {root.resolve()}")
    print(f"Total instance dirs found: {len(instance_dirs)}")
    print(f"Valid instances counted: {len(per_instance_stats)}")
    print("=" * 60)
    print("Average per instance:")
    print(f"  |V_Q| (query vertices):       {avg_query_vertices:.4f}")
    print(f"  |E_Q| (query hyperedges):     {avg_query_hyperedges:.4f}")
    print(f"  query_hypergraph count:       {avg_query_hg_count:.4f}")
    print(f"  |V| (data vertices sum):      {avg_data_vertices_sum:.4f}")
    print(f"  |E| (data hyperedges sum):    {avg_data_hyperedges_sum:.4f}")
    print(f"  data_hypergraph count:        {avg_data_hg_count:.4f}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count hypergraph statistics for all instances under a root folder. "
            "For each instance, query_hypergraph is one graph, while data_hypergraph uses the sum over all data graphs."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root folder that contains instance subfolders.",
    )
    args = parser.parse_args()

    count_hypergraphs(args.root_dir)


if __name__ == "__main__":
    main()
