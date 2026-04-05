from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from refine_hypergraph import (
	DEFAULT_BASE_URL,
	DEFAULT_MODEL,
	DashScopeChatClient,
	load_dataset_index,
	refine_instance_directory,
)


def _pick_instances(root: Path, limit: int) -> list[Path]:
	instance_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
	filtered: list[Path] = []
	for instance_dir in instance_dirs:
		if not (instance_dir / "metadata.json").exists():
			continue
		if not (instance_dir / "query_hypergraph.pkl").exists():
			continue
		filtered.append(instance_dir)
		if len(filtered) >= limit:
			break
	return filtered


def run_smoke_test(
	instances_root: str,
	dataset_path: str,
	limit_instances: int,
	max_data_graphs: int,
	save: bool,
	model: str,
	base_url: str,
) -> dict[str, Any]:
	root = Path(instances_root)
	if not root.exists():
		raise FileNotFoundError(f"Instances root not found: {root}")

	instance_dirs = _pick_instances(root=root, limit=limit_instances)
	if not instance_dirs:
		raise FileNotFoundError(f"No valid instance dirs found under: {root}")

	target_ids = {path.name for path in instance_dirs}
	dataset_index = load_dataset_index(dataset_path=dataset_path, target_ids=target_ids)
	client = DashScopeChatClient(model=model, base_url=base_url)

	results: list[dict[str, Any]] = []
	for instance_dir in instance_dirs:
		item = dataset_index.get(instance_dir.name)
		if item is None:
			results.append(
				{
					"instance_id": instance_dir.name,
					"status": "skipped",
					"reason": "dataset_item_not_found",
				}
			)
			continue

		updated_metadata = refine_instance_directory(
			instance_dir=instance_dir,
			item=item,
			client=client,
			model=model,
			base_url=base_url,
			dataset_path=dataset_path,
			max_data_graphs=max_data_graphs,
			save=save,
		)
		results.append(
			{
				"instance_id": instance_dir.name,
				"status": "updated",
				"query": updated_metadata.get("refine_summary", {}).get("query", {}),
				"data_count": len(updated_metadata.get("data_hypergraphs", [])),
			}
		)

	return {
		"instances_root": str(root.resolve()),
		"dataset_path": str(Path(dataset_path).resolve()),
		"model": model,
		"base_url": base_url,
		"save": save,
		"limit_instances": limit_instances,
		"max_data_graphs": max_data_graphs,
		"processed": len([item for item in results if item.get("status") == "updated"]),
		"skipped": len([item for item in results if item.get("status") == "skipped"]),
		"results": results,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Smoke test for DashScope-based hypergraph refine pipeline.")
	parser.add_argument("--instances-root", type=str, default="data/debug/musique/sample1000")
	parser.add_argument(
		"--dataset-path",
		type=str,
		default="/home/vincent/.dataset/musique/sample1000/musique_answerable.jsonl",
	)
	parser.add_argument("--limit-instances", type=int, default=1)
	parser.add_argument("--max-data-graphs", type=int, default=2)
	parser.add_argument("--save", action="store_true", help="Persist refined pkl and metadata.")
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
	parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
	args = parser.parse_args()

	summary = run_smoke_test(
		instances_root=args.instances_root,
		dataset_path=args.dataset_path,
		limit_instances=args.limit_instances,
		max_data_graphs=args.max_data_graphs,
		save=args.save,
		model=args.model,
		base_url=args.base_url,
	)
	print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()