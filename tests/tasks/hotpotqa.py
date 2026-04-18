from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any

from tqdm import tqdm
from langchain_ollama import ChatOllama

from hyper_simulation.component.hyper_simulation import compute_hyper_simulation
from hyper_simulation.component.postprocess import get_simulation_slice, ranking_slices
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from hyper_simulation.hypergraph.union import MultiHopFusion
from hyper_simulation.llm.chat_completion import get_invoke
from hyper_simulation.llm.prompt.hotpot_qa import HOTPOT_QA_HYPER
from hyper_simulation.question_answer.vmdit.metrics import (
	exact_match_score,
	metric_max_over_ground_truths,
	qa_f1_score,
	match,
)

from refine_hypergraph import load_dataset_index


DEFAULT_INSTANCES_ROOT = "data/debug/hotpotqa/sample1000_distractor"
DEFAULT_DATASET_PATH = "/home/vincent/.dataset/HotpotQA/sample1000"
DEFAULT_OUTPUT_PATH = "data/debug/hotpotqa/hotpotqa.json"


def _sorted_index_from_name(path: Path) -> int:
	match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", path.name)
	if match_obj is None:
		return 10**9
	return int(match_obj.group(1))


def _extract_hotpot_doc_entries(item: dict[str, Any]) -> list[dict[str, str]]:
	context = item.get("context")
	entries: list[dict[str, str]] = []

	# Newer normalized structure: {"title": [...], "sentences": [[...], ...]}
	if isinstance(context, dict):
		titles = context.get("title", [])
		sent_groups = context.get("sentences", [])
		if isinstance(titles, list):
			for idx, title in enumerate(titles):
				sentences = sent_groups[idx] if isinstance(sent_groups, list) and idx < len(sent_groups) else []
				if not isinstance(sentences, list):
					sentences = []
				text = " ".join(str(s).strip() for s in sentences if str(s).strip()).strip()
				entries.append({"title": str(title or "").strip(), "text": text})
		return entries

	# Original HotpotQA format: [(title, [sent1, sent2, ...]), ...]
	if isinstance(context, list):
		for record in context:
			if not (isinstance(record, (list, tuple)) and len(record) >= 2):
				continue
			title, sentences = record[0], record[1]
			if not isinstance(sentences, list):
				sentences = []
			text = " ".join(str(s).strip() for s in sentences if str(s).strip()).strip()
			entries.append({"title": str(title or "").strip(), "text": text})

	return entries


def _load_instance_graphs(instance_dir: Path, item: dict[str, Any]) -> tuple[LocalHypergraph | None, list[dict[str, Any]]]:
	query_path = instance_dir / "query_hypergraph.pkl"
	if not query_path.exists():
		return None, []

	try:
		query_hg = LocalHypergraph.load(str(query_path))
	except Exception:
		return None, []

	doc_entries = _extract_hotpot_doc_entries(item)
	data_paths = sorted(instance_dir.glob("data_hypergraph*.pkl"), key=_sorted_index_from_name)

	evidence_items: list[dict[str, Any]] = []
	for data_path in data_paths:
		match_obj = re.fullmatch(r"data_hypergraph(\d+)\.pkl", data_path.name)
		if match_obj is None:
			continue
		data_idx = int(match_obj.group(1))
		doc_entry = doc_entries[data_idx] if data_idx < len(doc_entries) else {}
		title = (doc_entry.get("title") or "").strip() if isinstance(doc_entry, dict) else ""
		text = (doc_entry.get("text") or "").strip() if isinstance(doc_entry, dict) else ""

		try:
			data_hg = LocalHypergraph.load(str(data_path))
		except Exception:
			data_hg = None

		if data_hg is None:
			continue

		evidence_items.append(
			{
				"index": data_idx,
				"path": str(data_path),
				"hypergraph": data_hg,
				"title": title,
				"text": text,
			}
		)

	return query_hg, evidence_items


def _normalize_answer(text: Any) -> str:
	if not text:
		return ""
	cleaned = str(text).replace("</s>", "").replace("</think>", "").strip()

	patterns = [
		r"###\s*Final\s*Answer:\s*(.+?)(?:\n|$)",
		r"ANSWER:\s*(.+?)(?:\n|$)",
		r"Answer:\s*(.+?)(?:\n|$)",
		r"###\s*Answer:\s*(.+?)(?:\n|$)",
	]
	for pattern in patterns:
		match_obj = re.search(pattern, cleaned, re.IGNORECASE)
		if match_obj:
			cleaned = match_obj.group(1).strip()
			break

	cleaned = cleaned.splitlines()[-1].strip()
	cleaned = cleaned.strip(" .,;:!?\"'()[]")
	if cleaned.lower() in {"", "unanswerable", "unknown", "none", "not mentioned", "cannot be determined"}:
		return "unanswerable"
	return cleaned


def _evaluate_answer(prediction: str, ground_truth: list[str] | str) -> dict[str, float]:
	if isinstance(ground_truth, list):
		ground_truths = [gt for gt in ground_truth if isinstance(gt, str) and gt.strip()]
	else:
		ground_truths = [ground_truth] if isinstance(ground_truth, str) and ground_truth.strip() else []

	if not ground_truths:
		return {"exact_match": 0.0, "f1": 0.0, "match": 0.0, "hit": 0.0}

	em_score = float(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
	f1_score = max(qa_f1_score(prediction, gt) for gt in ground_truths)
	match_score = float(match(prediction, ground_truths))
	hit = em_score > 0.0 or f1_score > 0.0 or match_score > 0.0
	return {"exact_match": em_score, "f1": f1_score, "match": match_score, "hit": float(hit)}


def _build_slice_text(slice_index: int, evidence_item: dict[str, Any]) -> str:
	title = evidence_item.get("title") or f"evidence_{slice_index}"
	text = (evidence_item.get("text") or "").strip()

	lines = [f"Slice {slice_index}", f"Evidence: {title}"]
	if text:
		lines.append(f"Paragraph: {text}")
	return "\n".join(lines)


# def _build_context_block(
# 	query: LocalHypergraph,
# 	simulation_slices: list[list[tuple[Vertex, Vertex]]],
# 	evidence_items: list[dict[str, Any]],
# 	vertex_ids: set[int],
# 	k: int = 10,
# ) -> tuple[list[int], list[int], str]:
# 	ranked_slice_indices = ranking_slices(query, simulation_slices, vertex_ids, k=k)

# 	consistent_indices: list[int] = []
# 	rendered_slices: list[str] = []
# 	for idx in ranked_slice_indices:
# 		if idx >= len(evidence_items):
# 			continue
# 		consistent_indices.append(int(evidence_items[idx].get("index", idx)))
# 		rendered_slices.append(_build_slice_text(idx, evidence_items[idx]))

# 	return consistent_indices, [], "\n\n".join(rendered_slices)

def _build_context_block(
	query: LocalHypergraph,
	simulation_slices: list[list[tuple[Vertex, Vertex]]],
	evidence_items: list[dict[str, Any]],
	vertex_ids: set[int],
	preferred_slice_indices: set[int] | None = None,
	k: int = 10,
) -> tuple[list[int], list[int], str]:
	ranked_slice_indices = ranking_slices(query, simulation_slices, vertex_ids, k=k)

	# 仅在 HotpotQA 本地做同分 tie-break：supporting_facts 命中的 slice 优先。
	preferred_slice_indices = preferred_slice_indices or set()
	vertex_needs: set[Vertex] = {u for u in query.vertices if u.id in vertex_ids}
	slice_hit_cnt: dict[int, int] = {}
	for idx, simulation_slice in enumerate(simulation_slices):
		present_u: set[Vertex] = {u for u, _ in simulation_slice if u is not None}
		slice_hit_cnt[idx] = sum(1 for u in vertex_needs if u in present_u)

	ranked_slice_indices.sort(
		key=lambda idx: (
			-slice_hit_cnt.get(idx, -1),
			0 if idx in preferred_slice_indices else 1,
			idx,
		)
	)

	consistent_indices: list[int] = []
	rendered_slices: list[str] = []
	for idx in ranked_slice_indices:
		if idx >= len(evidence_items):
			continue
		consistent_indices.append(int(evidence_items[idx].get("index", idx)))
		rendered_slices.append(_build_slice_text(idx, evidence_items[idx]))

	return consistent_indices, [], "\n\n".join(rendered_slices)


def _extract_preferred_slice_indices(item: dict[str, Any], evidence_items: list[dict[str, Any]]) -> set[int]:
	"""
	从 supporting_facts 中提取优先切片索引。
	优先规则：
	1) supporting_facts.sent_id 命中 evidence index
	2) supporting_facts.title 命中 evidence title
	"""
	supporting_facts = item.get("supporting_facts")
	if not isinstance(supporting_facts, dict):
		return set()

	preferred_data_indices: set[int] = set()
	for sent_id in supporting_facts.get("sent_id", []) or []:
		try:
			preferred_data_indices.add(int(sent_id))
		except (TypeError, ValueError):
			continue

	support_titles = {
		str(title).strip()
		for title in (supporting_facts.get("title", []) or [])
		if str(title).strip()
	}

	preferred_slice_indices: set[int] = set()
	for slice_idx, evidence in enumerate(evidence_items):
		data_idx = evidence.get("index")
		title = str(evidence.get("title") or "").strip()
		if isinstance(data_idx, int) and data_idx in preferred_data_indices:
			preferred_slice_indices.add(slice_idx)
			continue
		if title and title in support_titles:
			preferred_slice_indices.add(slice_idx)

	return preferred_slice_indices


def _build_final_prompt(question: str, context_text: str) -> str:
	return HOTPOT_QA_HYPER.format(context_text=context_text, question=question)


def run_hotpotqa_multihop_evaluation(
	instances_root: str = DEFAULT_INSTANCES_ROOT,
	dataset_path: str = DEFAULT_DATASET_PATH,
	output_path: str = DEFAULT_OUTPUT_PATH,
	hit0_output_path: str = "",
	model_name: str = "qwen3.5:9b",
	temperature: float = 0.2,
	limit_instances: int | None = None,
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

	dataset_index = load_dataset_index(task="hotpotqa", dataset_path=dataset_path)

	model = ChatOllama(
		model=model_name,
		temperature=temperature,
		reasoning=False,
		num_predict=8192,
	)

	results: list[dict[str, Any]] = []
	hit0_cases: list[dict[str, Any]] = []
	all_f1_scores: list[float] = []
	all_em_scores: list[float] = []
	all_match_scores: list[float] = []
	all_hit_scores: list[float] = []

	for instance_dir in tqdm(instance_dirs, desc="HotpotQA multi-hop QA", unit="inst"):
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

		query_hg, evidence_items = _load_instance_graphs(instance_dir, item)
		if query_hg is None or not evidence_items:
			results.append(
				{
					"instance_id": instance_dir.name,
					"status": "skipped",
					"reason": "missing_graphs",
				}
			)
			continue

		valid_hgs = [entry["hypergraph"] for entry in evidence_items if entry.get("hypergraph") is not None]
		if not valid_hgs:
			results.append(
				{
					"instance_id": instance_dir.name,
					"status": "skipped",
					"reason": "no_valid_evidence_graphs",
				}
			)
			continue

		fusion = MultiHopFusion()
		merged_hg, _provenance = fusion.merge_hypergraphs(valid_hgs)

		mapping, q_map, d_map = compute_hyper_simulation(query_hg, merged_hg)
		simulation = [
			(q_map[q_id], d_map[d_id])
			for q_id, d_ids in mapping.items()
			for d_id in d_ids
			if q_id in q_map and d_id in d_map
		]
		simulation_slices = get_simulation_slice(query_hg, merged_hg, simulation, len(valid_hgs))

		def allowed_vertex(vertex: Vertex) -> bool:
			if vertex.is_verb() or vertex.is_virtual():
				return False
			return True

		full_query_vertex_ids = {vertex.id for vertex in query_hg.vertices if allowed_vertex(vertex)}
		preferred_slice_indices = _extract_preferred_slice_indices(item, evidence_items)
		consistent_indices, inconsistent_indices, context_text = _build_context_block(
			query=query_hg,
			simulation_slices=simulation_slices,
			evidence_items=evidence_items,
			vertex_ids=full_query_vertex_ids,
			preferred_slice_indices=preferred_slice_indices,
		)

		prompt = _build_final_prompt((item.get("question") or "").strip(), context_text)
		raw_answer = get_invoke(model, prompt)
		prediction = _normalize_answer(raw_answer)

		answer = item.get("answer", "")
		aliases = item.get("answer_alias", []) or []
		final_ground_truth = [answer] + [alias for alias in aliases if alias != answer]
		final_metrics = _evaluate_answer(prediction, final_ground_truth)

		all_f1_scores.append(final_metrics["f1"])
		all_em_scores.append(final_metrics["exact_match"])
		all_match_scores.append(final_metrics["match"])
		all_hit_scores.append(final_metrics["hit"])

		if final_metrics.get("hit", 0.0) == 0.0:
			hit0_cases.append(
				{
					"instance_id": instance_dir.name,
					"question": (item.get("question") or "").strip(),
					"prediction": prediction,
					"ground_truth": final_ground_truth,
					"metrics": final_metrics,
				}
			)

		tqdm.write(
			"\n" + "-" * 72
			+ f"\nQuestion: {item.get('question', '').strip()}"
			+ f"\nLLM Answer: {prediction}"
			+ f"\nStandard Answer: {', '.join(gt for gt in final_ground_truth if gt) or 'N/A'}"
			+ f"\nF1: {final_metrics['f1']:.4f}"
			+ f"\nExact Match: {final_metrics['exact_match']:.4f}"
			+ f"\nMatch: {final_metrics['match']:.4f}"
			+ f"\nHit: {final_metrics['hit']:.4f}"
			+ "\n" + "-" * 72
		)

		instance_report = {
			"instance_id": instance_dir.name,
			"status": "ok",
			"question": (item.get("question") or "").strip(),
			"final": {
				"question": (item.get("question") or "").strip(),
				"prediction": prediction,
				"ground_truth": final_ground_truth,
				"metrics": final_metrics,
				"consistent_context": consistent_indices,
				"inconsistent_context": inconsistent_indices,
			},
		}
		results.append(instance_report)

	summary = {
		"instances_root": str(root.resolve()),
		"dataset_path": str(Path(dataset_path).resolve()),
		"total_instances": len(results),
		"evaluated_instances": sum(1 for item in results if item.get("status") == "ok"),
		"overall_f1": fmean(all_f1_scores) if all_f1_scores else 0.0,
		"overall_exact_match": fmean(all_em_scores) if all_em_scores else 0.0,
		"overall_match": fmean(all_match_scores) if all_match_scores else 0.0,
		"overall_hit": fmean(all_hit_scores) if all_hit_scores else 0.0,
		"final_count": len(all_f1_scores),
	}

	payload = {
		"summary": summary,
		"results": results,
	}

	out_path = Path(output_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

	if hit0_output_path:
		hit0_out_path = Path(hit0_output_path)
	else:
		hit0_out_path = out_path.with_name(f"{out_path.stem}_hit0.json")

	hit0_payload = {
		"summary": {
			"instances_root": str(root.resolve()),
			"dataset_path": str(Path(dataset_path).resolve()),
			"hit0_count": len(hit0_cases),
			"note": "These are final-step cases with hit=0. Review for potential answer_alias expansion.",
		},
		"cases": hit0_cases,
	}
	hit0_out_path.parent.mkdir(parents=True, exist_ok=True)
	hit0_out_path.write_text(json.dumps(hit0_payload, indent=2, ensure_ascii=False), encoding="utf-8")

	print("\n" + "=" * 72)
	print("HotpotQA multi-hop QA results")
	print("=" * 72)
	print(f"Total instances: {summary['total_instances']}")
	print(f"Evaluated instances: {summary['evaluated_instances']}")
	print(f"Overall F1: {summary['overall_f1']:.4f}")
	print(f"Overall Exact Match: {summary['overall_exact_match']:.4f}")
	print(f"Overall Match: {summary['overall_match']:.4f}")
	print(f"Overall Hit: {summary['overall_hit']:.4f}")
	print(f"Saved to: {out_path}")
	print(f"Hit=0 cases saved to: {hit0_out_path}")
	print("=" * 72)

	return payload


def main() -> None:
	parser = argparse.ArgumentParser(description="HotpotQA QA with hyper-simulation slice ranking")
	parser.add_argument("--instances-root", type=str, default=DEFAULT_INSTANCES_ROOT)
	parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
	parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--hit0-output-path", type=str, default="")
	parser.add_argument("--model-name", type=str, default="qwen3.5:9b")
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--limit-instances", type=int, default=0)
	args = parser.parse_args()

	run_hotpotqa_multihop_evaluation(
		instances_root=args.instances_root,
		dataset_path=args.dataset_path,
		output_path=args.output_path,
		hit0_output_path=args.hit0_output_path,
		model_name=args.model_name,
		temperature=args.temperature,
		limit_instances=args.limit_instances or None,
	)


if __name__ == "__main__":
	main()
