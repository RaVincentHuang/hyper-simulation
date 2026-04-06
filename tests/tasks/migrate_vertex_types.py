"""
Migrate vertex type labels from refined hypergraphs (brk) to newly generated hypergraphs (sample1000).

Logic:
  - Scan source (refined hypergraphs in brk)
  - Build a mapping: vertex.text() -> vertex.type()
  - Apply mapping to target (new hypergraphs in sample1000)
  - Save updated hypergraphs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from hyper_simulation.hypergraph.entity import ENT
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph


def build_instance_mapping(source_root: str, instance_id: str) -> dict[str, ENT]:
    """
    Build text -> type mapping for a specific instance from source.
    Scans both query and data hypergraphs for the given instance.
    
    Args:
        source_root: Path to refined instances (e.g., data/debug/musique/brk)
        instance_id: The specific instance ID to load
    
    Returns:
        Dictionary mapping vertex text to ENT type for this instance
    """
    instance_dir = Path(source_root) / instance_id
    if not instance_dir.exists():
        return {}

    mapping: dict[str, ENT] = {}

    # Process query hypergraph
    query_path = instance_dir / "query_hypergraph.pkl"
    if query_path.exists():
        try:
            query_hg = LocalHypergraph.load(str(query_path))
            for vertex in query_hg.vertices:
                vertex_type = vertex.type()
                if vertex_type is not None:
                    text = vertex.text()
                    if text:
                        mapping[text] = vertex_type
        except Exception as e:
            print(f"  Warning: Failed to load query from {query_path}: {e}")

    # Process data hypergraphs
    data_paths = sorted(instance_dir.glob("data_hypergraph*.pkl"))
    for data_path in data_paths:
        try:
            data_hg = LocalHypergraph.load(str(data_path))
            for vertex in data_hg.vertices:
                vertex_type = vertex.type()
                if vertex_type is not None:
                    text = vertex.text()
                    if text:
                        mapping[text] = vertex_type
        except Exception as e:
            print(f"  Warning: Failed to load {data_path.name}: {e}")

    return mapping


def migrate_instance(
    source_root: str,
    target_root: str,
    instance_id: str,
    save: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """
    Migrate types for a single instance from source to target.
    
    Args:
        source_root: Root directory of refined instances
        target_root: Root directory of new instances
        instance_id: The specific instance ID to process
        save: Whether to save updated hypergraphs
        force: Force overwrite even if types already exist
    
    Returns:
        Dictionary with instance migration stats
    """
    # Build mapping from source instance
    source_mapping = build_instance_mapping(source_root, instance_id)
    if not source_mapping:
        return {
            "instance_id": instance_id,
            "status": "skipped",
            "reason": "source_instance_not_found_or_empty",
            "applied": 0,
            "skipped": 0,
        }

    # Find target instance
    target_dir = Path(target_root) / instance_id
    if not target_dir.exists():
        return {
            "instance_id": instance_id,
            "status": "skipped",
            "reason": "target_instance_not_found",
            "applied": 0,
            "skipped": 0,
        }

    instance_applied = 0
    instance_skipped = 0

    # Process target query hypergraph
    target_query_path = target_dir / "query_hypergraph.pkl"
    if target_query_path.exists():
        try:
            target_query_hg = LocalHypergraph.load(str(target_query_path))
            for vertex in target_query_hg.vertices:
                text = vertex.text()
                if text in source_mapping:
                    old_type = vertex.type()
                    new_type = source_mapping[text]
                    if old_type is None or force:
                        vertex.type_cache = new_type
                        instance_applied += 1
                    else:
                        instance_skipped += 1
                else:
                    instance_skipped += 1

            if save:
                target_query_hg.save(str(target_query_path))
        except Exception as e:
            print(f"  Error processing target query {target_query_path}: {e}")

    # Process target data hypergraphs
    target_data_paths = sorted(target_dir.glob("data_hypergraph*.pkl"))
    for target_data_path in target_data_paths:
        try:
            target_data_hg = LocalHypergraph.load(str(target_data_path))
            for vertex in target_data_hg.vertices:
                text = vertex.text()
                if text in source_mapping:
                    old_type = vertex.type()
                    new_type = source_mapping[text]
                    if old_type is None or force:
                        vertex.type_cache = new_type
                        instance_applied += 1
                    else:
                        instance_skipped += 1
                else:
                    instance_skipped += 1

            if save:
                target_data_hg.save(str(target_data_path))
        except Exception as e:
            print(f"  Error processing target data {target_data_path}: {e}")

    return {
        "instance_id": instance_id,
        "status": "completed",
        "applied": instance_applied,
        "skipped": instance_skipped,
    }


def migrate_all_instances(
    source_root: str,
    target_root: str,
    save: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """
    Migrate types for all instances by processing each instance individually.
    
    Args:
        source_root: Root directory of refined instances
        target_root: Root directory of new instances
        save: Whether to save updated hypergraphs
        force: Force overwrite even if types already exist
    
    Returns:
        Summary dictionary with statistics
    """
    target_root_path = Path(target_root)
    if not target_root_path.exists():
        raise FileNotFoundError(f"Target root not found: {target_root}")

    target_instance_dirs = sorted([d for d in target_root_path.iterdir() if d.is_dir()])
    if not target_instance_dirs:
        raise FileNotFoundError(f"No instance directories found in {target_root}")

    print(f"Processing {len(target_instance_dirs)} instances...")

    summary = {
        "source_root": str(Path(source_root).resolve()),
        "target_root": str(target_root_path.resolve()),
        "total_instances": len(target_instance_dirs),
        "completed": 0,
        "skipped": 0,
        "total_applied": 0,
        "total_skipped": 0,
        "instance_stats": [],
    }

    for target_dir in tqdm(target_instance_dirs, desc="Migrating instances"):
        instance_id = target_dir.name
        result = migrate_instance(
            source_root=source_root,
            target_root=target_root,
            instance_id=instance_id,
            save=save,
            force=force,
        )
        
        summary["instance_stats"].append(result)
        if result["status"] == "completed":
            summary["completed"] += 1
            summary["total_applied"] += result["applied"]
            summary["total_skipped"] += result["skipped"]
        else:
            summary["skipped"] += 1

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate vertex type labels from refined to new hypergraphs (instance-by-instance)."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        required=True,
        help="Root directory of refined instances (e.g., data/debug/musique/brk)",
    )
    parser.add_argument(
        "--target-root",
        type=str,
        required=True,
        help="Root directory of new instances (e.g., data/debug/musique/sample1000)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Dry run: don't save changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing types",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Instance-by-Instance Type Migration")
    print("=" * 60)
    summary = migrate_all_instances(
        source_root=args.source_root,
        target_root=args.target_root,
        save=not args.no_save,
        force=args.force,
    )

    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total instances: {summary['total_instances']}")
    print(f"Completed: {summary['completed']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Total applied: {summary['total_applied']}")
    print(f"Total skipped: {summary['total_skipped']}")
    
    # Print detailed stats if there are failures
    failed = [s for s in summary['instance_stats'] if s['status'] != 'completed']
    if failed:
        print(f"\nSkipped instances ({len(failed)}):")
        for stat in failed[:10]:  # Show first 10
            print(f"  {stat['instance_id']}: {stat['reason']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    print("\nDetailed JSON:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
