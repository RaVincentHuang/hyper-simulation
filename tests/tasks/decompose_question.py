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


DEFAULT_INSTANCES_ROOT = "data/debug/musique/sample1417"
DEFAULT_DATASET_PATH = "/home/vincent/.dataset/musique/rest/musique_answerable.jsonl"
DEFAULT_OUTPUT_PATH = "data/debug/musique/decompose_with_subs.json"


def _build_vertex_text_map(query_hg: LocalHypergraph) -> dict[int, str]:
	return {vertex.id: vertex.text() for vertex in query_hg.vertices}


def _load_issue_instance_ids(issues_file: str) -> set[str]:
	path = Path(issues_file)
	if not path.exists():
		raise FileNotFoundError(f"issues file not found: {path}")
	obj = json.loads(path.read_text(encoding="utf-8"))
	issues = obj.get("issues", []) if isinstance(obj, dict) else []
	if not isinstance(issues, list):
		return set()
	ids: set[str] = set()
	for issue in issues:
		if not isinstance(issue, dict):
			continue
		instance_id = str(issue.get("instance_id", "")).strip()
		if instance_id:
			ids.add(instance_id)
	return ids


def _extract_sub_questions(item: dict[str, Any]) -> list[str]:
	decomposition = item.get("question_decomposition", []) or []
	if not isinstance(decomposition, list):
		return []

	subs: list[str] = []
	for step in decomposition:
		if not isinstance(step, dict):
			continue
		q = (step.get("question") or "").strip()
		if q:
			subs.append(q)
	return subs


# def _sorted_index_from_name(path: Path) -> int:
# 	match = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
# 	if match is None:
# 		return 10**9
# 	return int(match.group(1))


def run_batch_decompose_with_subs(
	instances_root: str = DEFAULT_INSTANCES_ROOT,
	dataset_path: str = DEFAULT_DATASET_PATH,
	output_path: str = DEFAULT_OUTPUT_PATH,
	limit_instances: int | None = None,
	batch_size: int = 8,
	show_live_details: bool = True,
	issue_instance_ids: set[str] | None = None,
) -> dict[str, Any]:
	root = Path(instances_root)
	if not root.exists():
		raise FileNotFoundError(f"Instances root not found: {root}")

	instance_dirs = sorted(
		[path for path in root.iterdir() if path.is_dir() and (path / "query_hypergraph.pkl").exists()]
	)
	if issue_instance_ids is not None:
		instance_dirs = [path for path in instance_dirs if path.name in issue_instance_ids]
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
			vertex_text_map = _build_vertex_text_map(sample["query_hg"])
			sub_results = [
				{
					"index": idx + 1,
					"question": sub_q,
					"vertex_ids": sorted(list(vertex_ids)),
					"vertices": [
						{
							"id": vid,
							"text": vertex_text_map.get(vid, "<UNKNOWN_VERTEX>"),
						}
						for vid in sorted(list(vertex_ids))
					],
				}
				for idx, (sub_q, vertex_ids) in enumerate(output)
			]
			instance_result = {
				"instance_id": sample["instance_id"],
				"status": "ok",
				"question": sample["question"],
				"input_subquestions": sample["subs"],
				"decomposed_subquestions": sub_results,
			}
			results_by_id[sample["instance_id"]] = instance_result

			# Also persist a per-instance copy for downstream tools.
			instance_output_path = root / sample["instance_id"] / "decompose.json"
			instance_output_path.write_text(
				json.dumps(instance_result, indent=2, ensure_ascii=False),
				encoding="utf-8",
			)

			if show_live_details:
				tqdm.write(f"\n[instance] {sample['instance_id']}")
				tqdm.write(f"[question] {sample['question']}")
				for sub in sub_results:
					tqdm.write(f"  ({sub['index']}) {sub['question']}")
					for vertex in sub["vertices"]:
						tqdm.write(f"      - [{vertex['id']}] {vertex['text']}")
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
	parser.add_argument("--disable-live-details", action="store_true")
	parser.add_argument("--issues-file", type=str, default="")
	args = parser.parse_args()

	issue_instance_ids: set[str] | None = None
	if args.issues_file:
		issue_instance_ids = _load_issue_instance_ids(args.issues_file)

	report = run_batch_decompose_with_subs(
		instances_root=args.instances_root,
		dataset_path=args.dataset_path,
		output_path=args.output_path,
		limit_instances=args.limit_instances or None,
		batch_size=max(1, args.batch_size),
		show_live_details=not args.disable_live_details,
		issue_instance_ids=issue_instance_ids,
	)
	print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
