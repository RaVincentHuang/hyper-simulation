from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph
from hyper_simulation.question_answer.decompose import decompose_question_with_subs_batch
from refine_hypergraph import load_dataset_index


DEFAULT_INSTANCES_ROOT = "data/debug/musique/sample1000"
DEFAULT_DATASET_PATH = "/home/vincent/.dataset/musique/sample1000/musique_answerable.jsonl"
DEFAULT_OUTPUT_PATH = "data/debug/musique/decompose_with_subs.json"


def _extract_sub_questions(item: dict[str, Any]) -> list[str]:
	decomposition = item.get("question_decomposition", []) or []
	if not isinstance(decomposition, list):
		return []

	# Keep stable order if ids exist.
	if decomposition and all(isinstance(step, dict) and "id" in step for step in decomposition):
		decomposition = sorted(decomposition, key=lambda step: step.get("id"))

	subs: list[str] = []
	for step in decomposition:
		if not isinstance(step, dict):
			continue
		q = (step.get("question") or "").strip()
		if q:
			subs.append(q)
	return subs


def _sorted_index_from_name(path: Path) -> int:
	match = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
	if match is None:
		return 10**9
	return int(match.group(1))


def run_batch_decompose_with_subs(
	instances_root: str = DEFAULT_INSTANCES_ROOT,
	dataset_path: str = DEFAULT_DATASET_PATH,
	output_path: str = DEFAULT_OUTPUT_PATH,
	limit_instances: int | None = None,
	batch_size: int = 8,
) -> dict[str, Any]:
	root = Path(instances_root)
	if not root.exists():
		raise FileNotFoundError(f"Instances root not found: {root}")

	instance_dirs = sorted(
		[path for path in root.iterdir() if path.is_dir() and (path / "query_hypergraph.pkl").exists()]
	)
	if limit_instances is not None and limit_instances > 0:
		instance_dirs = instance_dirs[:limit_instances]
	if not instance_dirs:
		raise FileNotFoundError(f"No valid instance directories found under: {root}")

	target_ids = {path.name for path in instance_dirs}
	dataset_index = load_dataset_index(dataset_path=dataset_path, target_ids=target_ids)

	samples: list[dict[str, Any]] = []
	for instance_dir in instance_dirs:
		item = dataset_index.get(instance_dir.name)
		if item is None:
			samples.append(
				{
					"instance_id": instance_dir.name,
					"status": "skipped",
					"reason": "dataset_item_not_found",
				}
			)
			continue

		query_path = instance_dir / "query_hypergraph.pkl"
		if not query_path.exists():
			samples.append(
				{
					"instance_id": instance_dir.name,
					"status": "skipped",
					"reason": "query_hypergraph_missing",
				}
			)
			continue

		try:
			query_hg = LocalHypergraph.load(str(query_path))
		except Exception as exc:
			samples.append(
				{
					"instance_id": instance_dir.name,
					"status": "skipped",
					"reason": f"query_hypergraph_load_failed: {type(exc).__name__}",
				}
			)
			continue

		question = (item.get("question") or "").strip()
		subs = _extract_sub_questions(item)
		samples.append(
			{
				"instance_id": instance_dir.name,
				"status": "pending",
				"question": question,
				"subs": subs,
				"query_hg": query_hg,
			}
		)

	pending = [s for s in samples if s.get("status") == "pending"]
	results_by_id: dict[str, dict[str, Any]] = {}

	pbar = tqdm(total=len(pending), desc="Decompose with subs", unit="inst")
	for start in range(0, len(pending), batch_size):
		chunk = pending[start : start + batch_size]
		questions = [s["question"] for s in chunk]
		subs_batch = [s["subs"] for s in chunk]
		queries = [s["query_hg"] for s in chunk]

		batch_outputs = decompose_question_with_subs_batch(
			questions=questions,
			subs_batch=subs_batch,
			queries=queries,
		)

		for sample, output in zip(chunk, batch_outputs):
			sub_results = [
				{
					"index": idx + 1,
					"question": sub_q,
					"vertex_ids": sorted(list(vertex_ids)),
				}
				for idx, (sub_q, vertex_ids) in enumerate(output)
			]
			results_by_id[sample["instance_id"]] = {
				"instance_id": sample["instance_id"],
				"status": "ok",
				"question": sample["question"],
				"input_subquestions": sample["subs"],
				"decomposed_subquestions": sub_results,
			}
		pbar.update(len(chunk))
	pbar.close()

	final_results: list[dict[str, Any]] = []
	ok_count = 0
	skip_count = 0
	for sample in samples:
		if sample.get("status") != "pending":
			final_results.append(sample)
			skip_count += 1
			continue
		one = results_by_id.get(sample["instance_id"])
		if one is None:
			final_results.append(
				{
					"instance_id": sample["instance_id"],
					"status": "skipped",
					"reason": "batch_output_missing",
				}
			)
			skip_count += 1
			continue
		final_results.append(one)
		ok_count += 1

	output = {
		"summary": {
			"instances_root": str(root.resolve()),
			"dataset_path": str(Path(dataset_path).resolve()),
			"total": len(samples),
			"ok": ok_count,
			"skipped": skip_count,
			"batch_size": batch_size,
		},
		"results": final_results,
	}

	out_path = Path(output_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
	return output


def main() -> None:
	parser = argparse.ArgumentParser(description="Batch decompose MuSiQue questions with provided sub-questions.")
	parser.add_argument("--instances-root", type=str, default=DEFAULT_INSTANCES_ROOT)
	parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
	parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--limit-instances", type=int, default=0)
	parser.add_argument("--batch-size", type=int, default=8)
	args = parser.parse_args()

	report = run_batch_decompose_with_subs(
		instances_root=args.instances_root,
		dataset_path=args.dataset_path,
		output_path=args.output_path,
		limit_instances=args.limit_instances or None,
		batch_size=max(1, args.batch_size),
	)
	print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
