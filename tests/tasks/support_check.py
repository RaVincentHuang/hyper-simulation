from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = ROOT / "tests" / "tasks"
SRC_DIR = ROOT / "src"

for candidate in (str(SRC_DIR), str(TASKS_DIR)):
	if candidate not in sys.path:
		sys.path.insert(0, candidate)

from tqdm import tqdm

from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.postprocess import post_detection
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.union import MultiHopFusion
from hyper_simulation.utils.log import current_query_id
from refine_hypergraph import load_dataset_index


def format_vertex(vertex: Vertex) -> str:
    nodes = "\n".join(
        f"    - '{node.text}' (pos={node.pos.name}, dep={node.dep.name}, ent={node.ent.name}, ENT={node.entity.name if node.entity else 'None'})"
        for node in vertex.nodes
    )
    return f"[{vertex.id}] '{vertex.text()}'\n{nodes}"

DEFAULT_INSTANCES_ROOT = "data/debug/musique/sample1000"
DEFAULT_DATASET_PATH = "/home/vincent/.dataset/musique/sample1000/musique_answerable.jsonl"


def _sorted_index_from_name(path: Path) -> int:
	match = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
	if match is None:
		return 10**9
	return int(match.group(1))


def _support_indices_from_item(item: dict[str, Any]) -> list[int]:
	support_indices: set[int] = set()

	for step in item.get("question_decomposition", []) or []:
		if not isinstance(step, dict):
			continue
		paragraph_idx = step.get("paragraph_support_idx")
		if paragraph_idx is None:
			continue
		try:
			support_indices.add(int(paragraph_idx))
		except (TypeError, ValueError):
			continue

	if support_indices:
		return sorted(support_indices)

	for idx, paragraph in enumerate(item.get("paragraphs", []) or []):
		if not isinstance(paragraph, dict):
			continue
		if bool(paragraph.get("is_supporting", False)):
			support_indices.add(idx)

	return sorted(support_indices)


def _load_instance_graphs(instance_dir: Path, item: dict[str, Any], max_data_graphs: int | None = None) -> tuple[LocalHypergraph | None, list[dict[str, Any]]]:
	query_path = instance_dir / "query_hypergraph.pkl"
	if not query_path.exists():
		return None, []

	query_hg = LocalHypergraph.load(str(query_path))
	paragraphs = item.get("paragraphs", []) or []
	data_paths = sorted(instance_dir.glob("data_hypergraph*.pkl"), key=_sorted_index_from_name)
	if max_data_graphs is not None:
		data_paths = data_paths[:max_data_graphs]

	evidence_items: list[dict[str, Any]] = []
	for data_path in data_paths:
		match = re.fullmatch(r"data_hypergraph(\d+)\.pkl", data_path.name)
		if match is None:
			continue
		data_idx = int(match.group(1))
		if data_idx >= len(paragraphs):
			continue
		paragraph = paragraphs[data_idx]
		if not isinstance(paragraph, dict):
			continue
		paragraph_text = (paragraph.get("paragraph_text") or "").strip()
		if not paragraph_text:
			continue
		try:
			data_hg = LocalHypergraph.load(str(data_path))
		except Exception:
			continue
		evidence_items.append(
			{
				"index": data_idx,
				"path": str(data_path),
				"hypergraph": data_hg,
				"text": paragraph_text,
			}
		)

	return query_hg, evidence_items


def _compute_query_metrics(ids: set[int], supports: set[int]) -> dict[str, float | int]:
	tp = len(ids & supports)
	fp = len(ids - supports)
	fn = len(supports - ids)
	precision = tp / len(ids) if ids else 0.0
	recall = tp / len(supports) if supports else 1.0
	f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
	jaccard = tp / len(ids | supports) if (ids | supports) else 1.0
	noise_ratio = fp / len(ids) if ids else 0.0
	clean_score = 1.0 - noise_ratio if ids else 0.0

	return {
		"predicted_ids": len(ids),
		"support_ids": len(supports),
		"tp": tp,
		"fp": fp,
		"fn": fn,
		"coverage_recall": recall,
		"coverage_precision": precision,
		"coverage_f1": f1,
		"coverage_jaccard": jaccard,
		"support_hit": 1 if tp > 0 else 0,
		"all_support_covered": 1 if fn == 0 else 0,
		"noise_count": fp,
		"noise_ratio": noise_ratio,
		"clean_score": clean_score,
	}


def evaluate_support_batch(
	instances_root: str = DEFAULT_INSTANCES_ROOT,
	dataset_path: str = DEFAULT_DATASET_PATH,
	limit_instances: int | None = None,
	max_data_graphs: int | None = None,
	show_live_score: bool = True,
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
	

	results: list[dict[str, Any]] = []
	running_metric_keys = [
		"coverage_recall",
		"coverage_precision",
		"coverage_f1",
		"noise_ratio",
		"clean_score",
	]
	running_sums: dict[str, float] = {key: 0.0 for key in running_metric_keys}
	running_ok = 0
	running_skipped = 0

	pbar = tqdm(instance_dirs, desc="Support check")
	for instance_dir in pbar:
		item = dataset_index.get(instance_dir.name)
		if item is None:
			running_skipped += 1
			results.append(
				{
					"instance_id": instance_dir.name,
					"status": "skipped",
					"reason": "dataset_item_not_found",
				}
			)
			pbar.set_postfix(ok=running_ok, skip=running_skipped)
			continue

		query_hg, evidence_items = _load_instance_graphs(instance_dir, item, max_data_graphs=max_data_graphs)
		if query_hg is None or not evidence_items:
			running_skipped += 1
			results.append(
				{
					"instance_id": instance_dir.name,
					"status": "skipped",
					"reason": "missing_graphs",
				}
			)
			pbar.set_postfix(ok=running_ok, skip=running_skipped)
			continue

		current_query_id.set(instance_dir.name)
		valid_hgs = [entry["hypergraph"] for entry in evidence_items]
		supports = set(_support_indices_from_item(item))
		fusion = MultiHopFusion()
		# print query and data graph info for debugging
		# print(f"Processing instance: {instance_dir.name}")
		# print(f"Query hypergraph: {query_hg}")
		# for entry in evidence_items:
		# 	# print the text
		# 	print(f"{entry['text']}")
		# 	print(f"Data hypergraph (index {entry['index']}): {entry['hypergraph']}")
		# 	hypergraph = entry["hypergraph"]
		# 	for vertex in hypergraph.vertices:
		# 		print(format_vertex(vertex))
		merged_hg, _provenance = fusion.merge_hypergraphs(valid_hgs)
		mapping, q_map, d_map = compute_hyper_simulation(query_hg, merged_hg)
		simulation = [
			(q_map[q_id], d_map[d_id])
			for q_id, d_ids in mapping.items()
			for d_id in d_ids
			if q_id in q_map and d_id in d_map
		]
		final_matches = post_detection(query_hg, merged_hg, simulation)

		ids: set[int] = set()
		matched_nodes: list[dict[str, Any]] = []
		for query_vertex, data_vertex in final_matches:
			if query_vertex is None or data_vertex is None:
				continue
			provenance_ids = sorted(data_vertex.get_provenance())
			ids.update(provenance_ids)
			matched_nodes.append(
				{
					"query": query_vertex.text(),
					"data": data_vertex.text(),
					"provenance_ids": provenance_ids,
				}
			)

		metrics = _compute_query_metrics(ids=ids, supports=supports)
		running_ok += 1
		for key in running_metric_keys:
			running_sums[key] += float(metrics[key])

		running_means = {
			key: (running_sums[key] / running_ok if running_ok else 0.0)
			for key in running_metric_keys
		}
		pbar.set_postfix(
			ok=running_ok,
			skip=running_skipped,
			recall=f"{running_means['coverage_recall']:.3f}",
			precision=f"{running_means['coverage_precision']:.3f}",
			f1=f"{running_means['coverage_f1']:.3f}",
			noise=f"{running_means['noise_ratio']:.3f}",
			clean=f"{running_means['clean_score']:.3f}",
		)

		if show_live_score:
			tqdm.write(
				(
					f"[{instance_dir.name}] "
					f"recall={float(metrics['coverage_recall']):.3f}, "
					f"precision={float(metrics['coverage_precision']):.3f}, "
					f"f1={float(metrics['coverage_f1']):.3f}, "
					f"noise={float(metrics['noise_ratio']):.3f}, "
					f"running_f1={running_means['coverage_f1']:.3f}, "
					f"running_noise={running_means['noise_ratio']:.3f}"
				)
			)

		results.append(
			{
				"instance_id": instance_dir.name,
				"status": "ok",
				"question": (item.get("question") or "").strip(),
				"support_ids": sorted(supports),
				"ids": sorted(ids),
				"matched_nodes": matched_nodes,
				"metrics": metrics,
			}
		)

	ok_results = [item for item in results if item.get("status") == "ok"]
	metric_keys = [
		"coverage_recall",
		"coverage_precision",
		"coverage_f1",
		"coverage_jaccard",
		"support_hit",
		"all_support_covered",
		"noise_count",
		"noise_ratio",
		"clean_score",
	]

	summary = {
		"instances_root": str(root.resolve()),
		"dataset_path": str(Path(dataset_path).resolve()),
		"processed": len(ok_results),
		"skipped": len(results) - len(ok_results),
		"means": {
			key: fmean(float(item["metrics"][key]) for item in ok_results) if ok_results else 0.0
			for key in metric_keys
		},
	}

	return {
		"summary": summary,
		"results": results,
	}


def _print_report(report: dict[str, Any]) -> None:
	summary = report["summary"]
	means = summary["means"]
	print(json.dumps(summary, indent=2, ensure_ascii=False))
	print()
	print("Batch averages:")
	print(f"  processed: {summary['processed']}")
	print(f"  skipped: {summary['skipped']}")
	print(f"  coverage_recall: {means['coverage_recall']:.4f}")
	print(f"  coverage_precision: {means['coverage_precision']:.4f}")
	print(f"  coverage_f1: {means['coverage_f1']:.4f}")
	print(f"  coverage_jaccard: {means['coverage_jaccard']:.4f}")
	print(f"  support_hit: {means['support_hit']:.4f}")
	print(f"  all_support_covered: {means['all_support_covered']:.4f}")
	print(f"  noise_count: {means['noise_count']:.4f}")
	print(f"  noise_ratio: {means['noise_ratio']:.4f}")
	print(f"  clean_score: {means['clean_score']:.4f}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Support coverage check for MuSiQue hypergraph matches.")
	parser.add_argument("--instances-root", type=str, default=DEFAULT_INSTANCES_ROOT)
	parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
	parser.add_argument("--limit-instances", type=int, default=0)
	parser.add_argument("--max-data-graphs", type=int, default=0)
	parser.add_argument("--output-path", type=str, default="")
	parser.add_argument("--disable-live-score", action="store_true")
	args = parser.parse_args()

	report = evaluate_support_batch(
		instances_root=args.instances_root,
		dataset_path=args.dataset_path,
		limit_instances=args.limit_instances or None,
		max_data_graphs=args.max_data_graphs or None,
		show_live_score=not args.disable_live_score,
	)

	if args.output_path:
		output_path = Path(args.output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

	_print_report(report)


if __name__ == "__main__":
	main()
