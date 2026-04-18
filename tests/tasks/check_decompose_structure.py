from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = ROOT / "tests" / "tasks"
SRC_DIR = ROOT / "src"

for candidate in (str(SRC_DIR), str(TASKS_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from refine_hypergraph import load_dataset_index


DEFAULT_INSTANCES_ROOT = "data/debug/musique/sample1000"
DEFAULT_DATASET_PATH = "/home/vincent/.dataset/musique/sample1000/musique_answerable.jsonl"
DEFAULT_OUTPUT_PATH = "data/debug/musique/decompose_structure_issues.json"


def _extract_original_sub_count(item: dict[str, Any]) -> int:
    decomposition = item.get("question_decomposition", []) or []
    if not isinstance(decomposition, list):
        return 0
    count = 0
    for step in decomposition:
        if not isinstance(step, dict):
            continue
        q = (step.get("question") or "").strip()
        if q:
            count += 1
    return count


def _extract_generated_sub_count(decompose_obj: dict[str, Any]) -> int:
    subqs = decompose_obj.get("decomposed_subquestions", []) or []
    if not isinstance(subqs, list):
        return 0
    return len(subqs)


def _extract_empty_vertex_subquestions(decompose_obj: dict[str, Any]) -> list[dict[str, Any]]:
    subqs = decompose_obj.get("decomposed_subquestions", []) or []
    if not isinstance(subqs, list):
        return []

    empty_items: list[dict[str, Any]] = []
    for idx, sub in enumerate(subqs):
        if not isinstance(sub, dict):
            continue
        vertex_ids = sub.get("vertex_ids", [])
        if not isinstance(vertex_ids, list):
            vertex_ids = []
        if len(vertex_ids) == 0:
            empty_items.append(
                {
                    "sub_index": idx + 1,
                    "sub_question": (sub.get("question") or "").strip(),
                }
            )
    return empty_items


def check_decompose_structure(
    instances_root: str = DEFAULT_INSTANCES_ROOT,
    dataset_path: str = DEFAULT_DATASET_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    limit_instances: int | None = None,
) -> dict[str, Any]:
    root = Path(instances_root)
    if not root.exists():
        raise FileNotFoundError(f"Instances root not found: {root}")

    instance_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if limit_instances is not None and limit_instances > 0:
        instance_dirs = instance_dirs[:limit_instances]

    target_ids = {p.name for p in instance_dirs}
    dataset_index = load_dataset_index(dataset_path=dataset_path, target_ids=target_ids)

    issues: list[dict[str, Any]] = []
    checked = 0
    missing_decompose = 0
    missing_dataset = 0

    for instance_dir in instance_dirs:
        instance_id = instance_dir.name
        item = dataset_index.get(instance_id)
        if item is None:
            missing_dataset += 1
            continue

        checked += 1
        original_count = _extract_original_sub_count(item)

        decompose_path = instance_dir / "decompose.json"
        if not decompose_path.exists():
            missing_decompose += 1
            continue

        try:
            decompose_obj = json.loads(decompose_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(
                {
                    "instance_id": instance_id,
                    "issue": "decompose_json_invalid",
                    "detail": f"{type(exc).__name__}",
                }
            )
            continue

        generated_count = _extract_generated_sub_count(decompose_obj)
        if generated_count < original_count:
            issues.append(
                {
                    "issue": "subquestion_count_less_than_original",
                    "instance_id": instance_id,
                    "original_subquestion_count": original_count,
                    "generated_subquestion_count": generated_count,
                    "delta": original_count - generated_count,
                    "decompose_path": str(decompose_path),
                    "question": (item.get("question") or "").strip(),
                }
            )

        empty_vertex_subs = _extract_empty_vertex_subquestions(decompose_obj)
        if empty_vertex_subs:
            issues.append(
                {
                    "issue": "subquestion_has_empty_vertex_ids",
                    "instance_id": instance_id,
                    "original_subquestion_count": original_count,
                    "generated_subquestion_count": generated_count,
                    "empty_vertex_subquestions": empty_vertex_subs,
                    "decompose_path": str(decompose_path),
                    "question": (item.get("question") or "").strip(),
                }
            )

    report = {
        "summary": {
            "instances_root": str(root.resolve()),
            "dataset_path": str(Path(dataset_path).resolve()),
            "total_instance_dirs": len(instance_dirs),
            "checked_with_dataset": checked,
            "missing_dataset_item": missing_dataset,
            "missing_decompose_json": missing_decompose,
            "structural_issue_count": len(issues),
        },
        "issues": issues,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check decompose structural issues: generated sub-question count < original count."
    )
    parser.add_argument("--instances-root", type=str, default=DEFAULT_INSTANCES_ROOT)
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit-instances", type=int, default=0)
    args = parser.parse_args()

    report = check_decompose_structure(
        instances_root=args.instances_root,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        limit_instances=args.limit_instances or None,
    )

    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
